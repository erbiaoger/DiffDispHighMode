实施计划：概率场 + 路径提取 + 数值矩阵数据集 + 两阶段训练（先 UNet 去噪）
目标与验收标准
目标：输入一张频散能量矩阵 E(f,c)，输出 K 条模态曲线 c_k(f)（基阶+高阶），并能量化评估误差与稳定性。
验收标准（至少满足这些）
用合成验证集：每个 mode 的 MAE(c)、Hit@tol（例如 ±20 m/s）、曲线连续性（断裂率、跳变率）可自动输出报表。
训练/推理全流程不依赖 PNG 渲染，主数据形态是数值矩阵（.npz/.npy/.pt）。
支持配置驱动的 K_max（可变 K）：样本可缺失部分高阶模态，训练时通过 mask 忽略缺失 mode。
推理输出为可复用的曲线文件（json/csv），而不是“看起来像”的图片。
现状与约束（从仓库探查得到）
你当前的数据生成脚本输出 PNG：gen_data_labels.py (line 1)。
频散能量计算（Park）已经是数值矩阵：dispersion.py (line 1) 的 get_dispersion() 返回 fr, c, img。
当前环境里 python 未安装 torch（需要你用 conda/pip 配好；下面计划会给出环境文件）。
总体架构（推荐落地方式）
Stage A（数据生成）：从物理模型生成 (E_clean, E_noisy, labels)，全部保存为固定网格的数值矩阵。
Stage B（能量图增强，先用 UNet）：训练 Denoiser(E_noisy) -> E_clean（回归/重建）。
Stage C（曲线概率场）：训练 Picker(E_clean or Denoiser(E_noisy)) -> P_k(f,c)（每个 mode 一张概率图）。
Stage D（路径提取）：对每个 P_k 用 DP/Viterbi 抽取曲线 c_k(f)，并加平滑/连续性约束。
Stage E（评估闭环）：合成验证集上按 mode 输出指标；可选做 permutation matching 防止 mode swap 误判。
1) 数据与标签：统一为“固定网格矩阵”
1.1 固定网格定义（256x256）
全局定义（写入 dataset meta.json）
fmin, fmax, F=256，cmin, cmax, C=256
f_axis = linspace(fmin,fmax,F)（Hz）
c_axis = linspace(cmin,cmax,C)（m/s）
所有样本都重采样到该网格：避免 PNG 裁剪/插值带来的隐性域差，并让真实数据走同一预处理。
1.2 单样本保存格式（建议 .npz）
每个样本 sample_XXXXXX.npz 至少包含：

E_clean：float32 [F,C]，归一化前的“干净能量矩阵”（来自无噪合成记录算出来的能量图）。
E_noisy：float32 [F,C]，施加 domain randomization 后的能量矩阵（噪声/缺道/参数扰动等）。
Y_curve_fc：float32 [K_max,F]，每个 mode 的真值曲线（单位 m/s），无效频点用 nan。
mode_mask：uint8 [K_max]，该样本哪些 mode 有效（disba 解不出来就置 0）。
meta：可选字典序列化（或独立 manifest.jsonl），记录生成参数（波子、噪声类型、SNR、缺道率等）。
说明：E_clean 很关键，它让 Stage B（去噪/增强）有监督信号；否则你只能做自监督而难度会上升。

1.3 标签从“线条图”改为“概率场真值”
训练 Picker 时不直接回归 c(f)，而是监督 P(f,c)：

生成 Y_map[k,f,c]：对每个 f，在 c 轴上放一个高斯峰（中心是真值 c_k(f)）
Y_map[k,f,:] = exp(-0.5 * ((c_axis - c_gt)/sigma_c)^2)
sigma_c 推荐用 “像素单位”配置：例如 sigma_px=2~4，换算到 m/s
训练时用 Y_map 做 BCE/Dice/FL（下面给出具体 loss 组合）。
2) Domain Randomization：让合成“更像真数据”
把 randomization 拆成 3 层，尽量在“数值域”完成并被记录进 meta，便于复现实验。

2.1 记录域（dshift）随机化（优先级最高）
在生成 dshift 后、算能量图前做：

子波随机化：Ormsby 四频随机、或 Ricker 主频随机、或混合子波。
噪声
白噪声：按目标 SNR 随机（比如 0–20 dB）。
彩噪：1/f^p（p 随机 0.5–2）。
相干噪：叠加 1–3 条线性事件（随机速度、截距、带宽）。
衰减/频散扰动（近似也行）：对频谱乘随机的频率相关衰减曲线（控制高频更容易被压低）。
阵列与采样
缺道：随机丢掉 r_missing（例如 0–30%），可成段丢失。
非均匀采样：对 x 加小扰动后插值到规则网格（或直接在能量图阶段模拟）。
孔径变化：随机裁剪/抽稀接收道数（模拟短阵列）。
2.2 能量图算法参数随机化
同一条记录用随机参数计算能量图，增强泛化：

fmin/fmax 小范围 jitter（保证仍映射到统一 f_axis）。
cmin/cmax/dc jitter（最终仍重采样到统一 c_axis）。
归一化方式随机：linear/log1p/per-f 归一化（但要与真实数据保持一致的候选集合）。
2.3 能量矩阵域随机化（补充）
在 E_clean 上再做轻量增强（训练去噪器/Picker 都可用）：

对比度/伽马、局部遮挡（模拟能量空洞）、随机竖条缺失（模拟频点缺失）。
3) 两阶段模型（先 UNet 去噪，再 Picker）
你选择“先用普通 UNet 去噪”是对的：先把闭环跑通，之后再把 Stage B 换回扩散会顺很多。

3.1 Stage B：Denoiser（UNet 回归）
输入：E_noisy（[1,F,C]）
输出：E_hat（[1,F,C]），回归到 E_clean
Loss（推荐组合）
L1(E_hat, E_clean) + λ * SSIM（可选）或 Huber
训练策略
先不追求最强模型，追求“能稳定提升 Picker 指标”
训练时保持能量图归一化策略与真实数据一致（否则会引入新域差）
3.2 Stage C：Picker（概率场预测）
输入：E_clean 或 E_hat
输出：logits：[K_max,F,C]（每个 mode 一张 logit map）
激活：sigmoid 得到 P_map（不强制每个 f 只有一个 c）
Loss（推荐组合）
BCEWithLogits(logits, Y_map)（主 loss）
+ α * DiceLoss(sigmoid(logits), Y_map)（缓解正负极度不平衡）
* mode_mask[k]（对缺失 mode 直接屏蔽 loss）
4) 路径提取：DP/Viterbi 从概率场抽曲线
为每个 mode 单独抽一条 c_k(f)，并支持缺失频段。

4.1 状态与代价
状态：速度索引 j ∈ [0,C-1]，额外加一个 NULL 状态表示该频点“不拾取”
观测代价（推荐）
cost_obs(f,j) = -log(P_map(f,j)+eps)
cost_obs(f,NULL) = const_null（可配置，控制允许断裂的程度）
转移代价（平滑/连续性）
cost_tr(j_prev, j) = λ * min(|j-j_prev|, max_jump)^2
NULL 的转移代价单独设置：进入/退出 NULL 给予惩罚，避免无意义断裂
4.2 DP 计算与回溯
递推：dp[f,j] = cost_obs(f,j) + min_{j’ in window} (dp[f-1,j’] + cost_tr(j’,j))
window 限制：|j-j’| <= max_jump（O(FCmax_jump)，256 网格可控）
回溯得到 j*(f)，再映射到 c_axis[j]
4.3 多模态与 mode swap
因为你要 K_max 可变，评估时做“最优匹配”更稳：
对预测曲线集合与真值曲线集合做 assignment（按整体 MAE 最小匹配），避免单纯按通道索引造成 mode swap 误判
训练时仍保持通道=mode（能稳定收敛），评估时允许匹配更公平
5) 评估与报表（必须做，不然很难迭代）
对每个样本、每个 mode 输出：

MAE(m/s)：仅在真值非 nan 的频点统计
Hit@tol：如 tol=20 m/s
Coverage：预测非 NULL 的频点比例
BreakRate：NULL 段数量或 NULL 占比
Smoothness：mean(|Δc|) 或 mean(|Δ²c|)
CrossingCount（可选）：mode 间曲线交叉次数（仅作为诊断，不做硬约束）
输出形式：

每次评估生成 metrics.json + metrics.csv
可选输出对比图（仅用于可视化，不参与训练输入输出）
6) 代码落地：文件与脚本（建议的最小集合）
下面是“实现闭环所需的最少文件”，先别追求工程化过度。

6.1 新增/替换脚本
数据生成（数值矩阵版）：gen_dataset_npz.py (line 1)
取代现有 PNG 输出的 gen_data_labels.py (line 1)（保留它做可视化也行）
训练 denoiser：train_denoiser.py (line 1)
训练 picker：train_picker.py (line 1)
推理+DP 抽取：infer_curves.py (line 1)
评估：eval_picker.py (line 1)
6.2 新增核心模块（建议放新包，避免和 diffseis 混在一起）
数据集与预处理：data.py (line 1)
Domain randomization：augment.py (line 1)
模型（UNet 简化版复用也行）：models.py (line 1)
DP/Viterbi：path.py (line 1)
指标：metrics.py (line 1)
7) 环境与依赖（必须先解决）
增加一个环境文件（两选一）
requirements.txt：torch, torchvision, numpy, scipy, matplotlib（可选，仅画图）, disba, ccfj, pylops
或 environment.yml（推荐 conda，避免 torch 安装踩坑）
验收：python -c "import torch; print(torch.__version__)" 能通过
8) 实施里程碑（按最少返工顺序）
里程碑 1：数值矩阵数据集闭环
完成 gen_dataset_npz.py (line 1)，生成 E_clean/E_noisy/Y_curve_fc/mode_mask，并能加载成 batch。
里程碑 2：Stage C（Picker）先不加 Stage B
直接用 E_noisy -> P_map 训练，先把 DP 抽曲线 + 指标打通（最关键）。
里程碑 3：加入 DP/Viterbi + 评估报表
让训练每 N step 自动跑一小份 val，输出 MAE/Hit/Coverage/BreakRate。
里程碑 4：加入 Stage B（UNet 去噪/增强）
训练 denoiser，比较 “无 denoiser vs 有 denoiser” 对 Picker 指标的提升。
里程碑 5：把 randomization 拉满并做消融
一次只加一种 randomization，观测指标变化，找最有效的 3–5 个保留。
里程碑 6（可选）：Stage B 换回扩散模型
只有在 UNet 去噪收益明确、数据形态稳定后再上扩散，避免系统复杂度爆炸。
默认关键超参（可以直接写进 config）
grid: F=C=256
K_max: 先从 5 起（mode=0..4），用 mode_mask 屏蔽缺失
sigma_px（label 高斯宽度）：3
DP 参数：lambda_smooth=0.5~2.0，max_jump=6~12，const_null 通过验证集调到断裂率合理
归一化：推荐 E = log1p(E) 后做 per-sample 标准化（均值方差或 minmax），并固化成函数供真数据共用
明确假设（本计划基于这些默认）
合成阶段可以计算到 mode=0..K_max-1 的频散曲线；若某 mode 解失败，用 mode_mask 忽略。
真实数据推理时你能把能量图变成与合成一致的 E(f,c) 数值矩阵（同 f_axis/c_axis、同归一化）。
你接受评估时允许做“最优 mode 匹配”（避免 mode swap 造成误判），但训练通道仍对齐 mode 索引。