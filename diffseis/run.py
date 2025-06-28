# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from diffusion import GaussianDiffusion, Trainer
from unet import UNet

import torch
from torch import nn

mode = "demultiple" #demultiple, interpolation, denoising
folder = "dataset/"+mode+"/data_train/"
image_size = (128,128)

model = UNet(
        in_channel  = 2,        # 4 for demultiple; 3 for data; 1 for label
        out_channel = 1         # 1 for label
).cuda()

# if torch.cuda.device_count() > 1: # 含有多张GPU的卡
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model) # 单机多卡DP训练

diffusion = GaussianDiffusion(
    model,
    mode = mode,
    channels = 1,
    image_size = image_size,
    timesteps = 2000,
    loss_type = 'l1' # L1 or L2
).cuda()

# if torch.cuda.device_count() > 1: # 含有多张GPU的卡
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     diffusion = nn.DataParallel(diffusion) # 单机多卡DP训练


trainer = Trainer(
    diffusion,
    mode                        = mode,
    folder                      = folder,
    image_size                  = image_size,
    train_batch_size            = 16,          #32 for A100; 16 for GTX
    train_lr                    = 2e-5,
    train_num_steps             = 1000000,        # total training steps
    gradient_accumulate_every   = 2,          # gradient accumulation steps 梯度累积步骤
    ema_decay                   = 0.995,      # exponential moving average decay
    amp                         = True,       # turn on mixed precision
    save_and_sample_every_image = 300,
    save_and_sample_every_model = 500
)

trainer.train()