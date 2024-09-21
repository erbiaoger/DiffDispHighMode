# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import math
import copy
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import random

from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
import os

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        mode,
        channels,
        image_size,
        timesteps=2000,
        loss_type='l1',
    ):
        super().__init__()
        self.mode       = mode
        self.channels   = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.timesteps  = timesteps
        
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # 生成 beta schedule
        betas                         = make_beta_schedule(schedule='linear', n_timestep=timesteps, linear_start=1e-6, linear_end=1e-2)
        betas                         = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas                        = 1. - betas
        alphas_cumprod                = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev           = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # 计算扩散 q(x_t | x_{t-1}) 和其他值
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # 计算后验分布 q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # 上面的公式等于 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # 下面的对数计算被裁剪，因为扩散链开始时后验方差为0
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    def predict_start_from_noise(self, x_t, t, noise):
        # 从噪声预测初始值
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        # 计算后验分布
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            # 如果有条件 x，则将其与 x 合并并使用 denoise_fn 进行去噪
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            # 否则，直接使用 denoise_fn 对 x 进行去噪
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        # 计算模型的均值和后验对数方差
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        # 从模型分布中采样
        model_mean, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()
    
    @torch.no_grad()
    def p_sample_loop(self, x_in, mask=None):
        device = self.betas.device
        x_cond = x_in
        if mask is not None:  
            x_cond = x_in * mask
            
        shape = x_cond.shape
        img = torch.randn(shape, device=device)
        ret_img = x_cond
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, i, condition_x=x_cond)     # 多次经UNet去噪
            if mask is not None:
                img = x_cond + img * (1. - mask)
            
        if mask is not None:
            ret_img = torch.cat([ret_img, x_in], dim=0)
        ret_img = torch.cat([ret_img, img], dim=0)
        return ret_img
    
    @torch.no_grad()
    def inference(self, x_in, mask=None):
        # 推理过程
        return self.p_sample_loop(x_in, mask)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 从后验分布中采样
        return (continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise)


    def p_losses(self, x_cond, x_start, noise=None):
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)    # 随机从 1 ～ T 选择一个 t
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1],
                              self.sqrt_alphas_cumprod_prev[t],size=b)).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)       # t 序列

        noise   = default(noise, lambda: torch.randn_like(x_start))     # 随机噪声
        x_noisy = self.q_sample(x_start=x_start,
                    continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
                    noise=noise)                                        # image + noise

        
        if self.mode == "interpolation":
            # here x_cond -> mask
            x_recon = self.denoise_fn(torch.cat([x_start*x_cond, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            loss    = self.loss_func(noise, x_recon)
        else:
            x_recon = self.denoise_fn(torch.cat([x_cond, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)       # predict noise
            loss    = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
    

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, mode):
        super().__init__()
        self.folder      = folder
        self.image_size  = image_size
        self.mode        = mode
        self.data_files   = [self.folder +"data/"+ f for f in os.listdir(self.folder+"data/") if os.path.isfile(os.path.join(self.folder+"data/", f))]
        self.labels_files = [self.folder +"labels/"+f for f in os.listdir(self.folder+"labels/") if os.path.isfile(os.path.join(self.folder+"labels/", f))]
        

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

        
    def __len__(self):
        dir_path = self.folder+"data/"
        res      = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
        return res
    
    def irregular_mask(self, data, rate=0.5):
        """the mask matrix of random sampling
        Args:
            data: original data patches
            rate: sampling rate,range(0,1)
        """
        n    = data.size()[-1]
        mask = torch.torch.zeros(data.size(),dtype=torch.float64)
        
        v            = round(n*rate)
        TM           = random.sample(range(n),v)
        mask[:,:,TM] = 1 # missing by column
        mask         = mask.type(torch.HalfTensor)
        return  mask

    def __getitem__(self, index):
        # data = self.folder+"data/"+str(index)+".png"
        data     = self.data_files[index]
        img_data = Image.open(data)

        if self.mode == "demultiple":
            # label = self.folder+"labels/"+str(index)+".png"
            label     = self.labels_files[index]
            img_label = Image.open(label)
            return self.transform(img_data), self.transform(img_label)
        elif self.mode == "interpolation":
            return self.irregular_mask(self.transform(img_data)), self.transform(img_data)
        elif self.mode == "denoising":
            img   = self.transform(img_data)
            mean  = torch.mean(img)
            std   = torch.std(img)
            noise = 0.5*torch.normal(mean, std, size =(img.shape[0], img.shape[1], img.shape[2]))
            img_  = img + noise
            
            return img_, img
        

        else:
            print("ERROR MODE")

# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        # 遍历当前模型和移动平均模型的参数
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            # 更新移动平均值
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        # 更新移动平均值的公式
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,       
        mode,
        folder,
        *,
        ema_decay                   = 0.999,
        image_size                  = (128,128),
        train_batch_size            = 32,
        train_lr                    = 3e-6,
        train_num_steps             = 100000,
        gradient_accumulate_every   = 2,
        amp                         = False,
        step_start_ema              = 5000,
        update_ema_every            = 1,
        save_and_sample_every_image = 1000,
        save_and_sample_every_model = 10000
    ):
        """
        初始化一个 Diffusion 对象。

        参数：
            diffusion_model (object)                  :  要使用的扩散模型。
            mode (str)                                :  扩散模式。
            folder (str)                              :  包含数据集的文件夹。
            ema_decay (float, optional)                :  指数移动平均的衰减率。默认值为 0.999。
            image_size (tuple, optional)              :  输入图像的尺寸。默认值为 (128, 128)。
            train_batch_size (int, optional)          :  训练时的批量大小。默认值为 32。
            train_lr (float, optional)                 :  训练时的学习率。默认值为 3e-6。
            train_num_steps (int, optional)           :  训练步骤数。默认值为 100000。
            gradient_accumulate_every (int, optional) :  在更新模型之前累积梯度的步数。默认值为 2。
            amp (bool, optional)                      :  是否使用自动混合精度训练。默认值为 False。
            step_start_ema (int, optional)            :  开始更新指数移动平均的步数。默认值为 5000。
            update_ema_every (int, optional)          :  每次更新指数移动平均之间的步数。默认值为 1。
            save_and_sample_every (int, optional)     :  每次保存和采样之间的步数。默认值为 10000。
        """

        super().__init__()
        self.model            = diffusion_model             # 扩散模型
        self.mode             = mode                        # 模式
        self.folder           = folder                      # 文件夹路径
        self.ema              = EMA(ema_decay)              # 指数移动平均（EMA）
        self.ema_model        = copy.deepcopy(self.model)   # EMA模型的深拷贝
        self.update_ema_every = update_ema_every            # 更新EMA的频率

        self.step_start_ema              = step_start_ema               # 开始应用EMA的步骤
        self.save_and_sample_every_image = save_and_sample_every_image  # 保存和采样的频率
        self.save_and_sample_every_model = save_and_sample_every_model  # 保存和采样的频率

        self.batch_size                = train_batch_size               # 训练批次大小
        self.image_size                = diffusion_model.image_size     # 图像尺寸
        self.gradient_accumulate_every = gradient_accumulate_every      # 梯度累积频率
        self.train_num_steps           = train_num_steps                # 训练步数

        self.ds  = Dataset(self.folder, image_size, mode)               # 数据集
        self.dl  = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, 
                                         shuffle=True, pin_memory=True)) # 数据加载器
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)      # 优化器

        self.step   = 0                         # 当前步骤数
        self.amp    = amp                       # 自动混合精度
        self.scaler = GradScaler(enabled=amp)   # 梯度缩放器
        
        results_folder      = './results_' + str(self.mode) # 结果文件夹路径
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)            # 创建结果文件夹

        self.reset_parameters()                             # 重置参数

    def reset_parameters(self):
        # 从原始模型复制参数到EMA模型
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        # 如果当前步骤小于开始使用EMA的步骤，则重置参数
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        # 更新EMA模型的平均参数
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        # # 将模型状态和其他重要信息保存到文件
        # data = {
        #     'step'   :  self.step,
        #     'model'  :  self.model.state_dict(),
        #     'ema'    :  self.ema_model.state_dict(),
        #     'scaler' :  self.scaler.state_dict()
        # }
        # torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        # 保存模型
        torch.save(self.model, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        # 从文件加载模型状态和其他重要信息
        data      = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        while self.step < self.train_num_steps:  # 当训练步数小于总训练步数时
            for i in range(self.gradient_accumulate_every):  # 进行梯度累积
                img    = next(self.dl)  # 从数据加载器中获取下一个批次的数据
                inputs = img[0].cuda()  # 将输入数据移动到GPU
                gt     = img[1].cuda()  # 将目标数据移动到GPU
                
                # print("input shape: ", inputs.shape)
                # print("gt shape: ", gt.shape)
                
                with autocast(enabled=self.amp):  # 在自动混合精度环境中进行计算
                    loss = self.model(inputs, gt).cuda()  # 计算损失，并移动到GPU   # 单次
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()  # 缩放损失并进行反向传播

                print(f'{self.step}: {loss.item()}')  # 打印当前步数和损失值

            self.scaler.step(self.opt)  # 更新优化器参数
            self.scaler.update()        # 更新缩放器
            self.opt.zero_grad()        # 清空优化器的梯度

            if self.step % self.update_ema_every == 0:  # 每隔一定步数更新EMA模型
                self.step_ema()
            if self.step != 0 and self.step % self.save_and_sample_every_image == 0:  # 每隔一定步数保存和采样
                milestone = self.step // self.save_and_sample_every_image  # 计算里程碑编号
                inputs_   = torch.unsqueeze(inputs[0], dim=0)  # 扩展输入数据的维度
                
                if self.mode == "interpolation":  # 如果模式为插值
                    gt_        = torch.unsqueeze(gt[0], dim=0)  # 扩展目标数据的维度
                    all_images = self.ema_model.inference(x_in=gt_, mask=inputs_)  # 通过EMA模型进行推理
                else:
                    all_images = self.ema_model.inference(x_in=inputs_)  # 通过EMA模型进行推理
                all_images = (all_images + 1) * 0.5  # 将图像像素值归一化到[0, 1]
                utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=6)  # 保存图像
            if self.step != 0 and self.step % self.save_and_sample_every_model == 0:  # 每隔一定步数保存和采样
                self.save(milestone)  # 保存模型

            self.step += 1  # 增加步数

        print('training completed')  # 打印训练完成信息
