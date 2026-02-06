## Pick High Mode Dispersion with Diffusion

用扩散模型从频散能量图里自动“挑出”高阶模态的频散曲线，并配套了一套合成数据与标签生成流程。

### 项目要解决什么问题
- 输入：由面波记录计算得到的频散能量图（如 Park 相移法或 F‑J 方法）。
- 输出：高阶模态（多模态）频散曲线的像素级标注/轨迹。
- 核心思路：先用理论层状模型生成面波记录与真值频散曲线，再把频散能量图作为条件输入，让扩散模型生成“干净的”曲线标签图像。

### 总体流程
![](figs/Drawing%202024-09-17%2015.44.21.excalidraw.png)

1. 设定层状介质模型（厚度、Vs 等）并生成理论频散曲线。  
2. 由理论频散曲线合成面波记录。  
3. 用 Park 相移法或 F‑J 计算频散能量图。  
4. 叠加真值曲线生成标签图。  
5. 训练扩散模型（UNet + Gaussian Diffusion），输入能量图，输出曲线标签。  

### 关键实现细节
- 频散曲线生成：使用 `disba.PhaseDispersion` 计算多模态 Rayleigh 波频散曲线，代码参考 `Dispersion/make_label.py`、`diffseis/utools.py`。  
- 合成面波记录：在 `Dispersion/surfacewaves.py`（以及相关脚本/Notebook）中用 Ormsby 子波和频散曲线生成合成记录。  
- 频散能量图提取：Park 相移法见 `Dispersion/dispersionspectra.py`、`Dispersion/Dispersion/dispersion.py`；F‑J 方法用 `ccfj` + `scipy`，示例见 `Dispersion/make_label.py`。  
- 标签构建：将真值曲线绘制到空白图像上作为 label，图像保存到 `diffseis/dataset/demultiple/data_train/labels/`。  
- 扩散模型：UNet 噪声预测器在 `diffseis/unet.py`，Gaussian Diffusion 训练与推理在 `diffseis/diffusion.py`；训练入口是 `diffseis/run.py`（支持 `demultiple / interpolation / denoising`，这里主要用于“曲线提取/去噪”）。  

### 目录结构
- `Dispersion/`：频散曲线、面波合成、能量图提取、标签生成的脚本与 Notebook。关键脚本：`Dispersion/make_label.py`、`Dispersion/dispersionspectra.py`。  
- `diffseis/`：扩散模型实现与训练入口。关键脚本：`diffseis/diffusion.py`、`diffseis/unet.py`、`diffseis/run.py`。  
- `figs/`：流程图与结果图。  

### 结果示例
理论频散曲线生成合成面波记录  
![](figs/Dp2ShotGather.png)

相移法和 F‑J 方法提取频散能量图的区别  
![](figs/diff_Phase_FJ.png)

最终 diffusion 生成结果  
![](figs/output.png)

### 使用方式（建议顺序）
1. **生成数据与标签**：参考 `Dispersion/make_label.py` 或 `Dispersion/Make_Label_*` Notebook；输出路径默认指向 `diffseis/dataset/demultiple/data_train/{data,labels}`。  
2. **训练扩散模型**：运行 `diffseis/run.py`；重要参数包括 `image_size`、`train_batch_size`、`train_lr`、`train_num_steps`。  
3. **推理/可视化**：使用 `diffseis/diffusion.py` 的 `inference()` 或 `diffseis/utools.py` 里的可视化函数。  

### 依赖与环境（核心）
- Python 3.x  
- numpy, scipy, matplotlib, PIL  
- torch, torchvision  
- disba, pylops, ccfj  

### 备注
- 目前主要流程通过 Notebook 和脚本驱动，尚未整理成统一 CLI。  
- 训练与推理默认基于 GPU（`.cuda()`），如果在 CPU 上运行需要修改相关调用。  

### 新版数值矩阵流程（推荐）
为减少 PNG 渲染带来的域差，这里新增了一套“能量矩阵 + 概率场 + DP 抽曲线”的脚本化流程：

- 数据集生成（`.npz`，包含 `E_clean/E_noisy/Y_curve_fc/mode_mask`）  
  - `scripts/gen_dataset_npz.py`  
- Stage B：能量图去噪（UNet 回归）  
  - `scripts/train_denoiser.py`  
- Stage C：曲线概率场预测（UNet 输出 `K_max` 通道）  
  - `scripts/train_picker.py`  
- 推理与曲线导出（DP/Viterbi 抽取 `c_k(f)`，输出 `jsonl/csv`）  
  - `scripts/infer_curves.py`  
- 评估报表（MAE、Hit@tol、Coverage、BreakRate、Smoothness）  
  - `scripts/eval_picker.py`  

**环境依赖**
- `disba`：用于计算多模态理论频散曲线（标签来源）  
- `torch/torchvision`：训练与推理  
- 参考：`environment.yml` 或 `requirements.txt`

**最小运行示例**
```bash
# 1) 生成数值矩阵数据集
python scripts/gen_dataset_npz.py --out data/demultiple/npz --num-train 200 --num-val 50

# 2) 训练 picker（直接 E_noisy -> P_map）
python scripts/train_picker.py --dataset-root data/demultiple/npz --out runs/picker --steps 20000

# 3) 推理导出曲线（jsonl + csv）
python scripts/infer_curves.py --picker-ckpt runs/picker/picker_final.pt --dataset-root data/demultiple/npz --split val --out runs/infer
```
