#!/usr/bin/env python3
"""Train diffusion model on generated dispersion images."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from diffseis.diffusion import GaussianDiffusion, Trainer  # noqa: E402
from diffseis.unet import UNet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model.")
    parser.add_argument("--mode", type=str, default="demultiple")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--grad-acc", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")

    args = parser.parse_args()

    folder = Path(args.data_root) / args.mode / "data_train"

    model = UNet(in_channel=2, out_channel=1).cuda()
    if args.no_cuda:
        model = model.cpu()

    diffusion = GaussianDiffusion(
        model,
        mode=args.mode,
        channels=1,
        image_size=(args.image_size, args.image_size),
        timesteps=args.timesteps,
        loss_type=args.loss,
    )

    if not args.no_cuda:
        diffusion = diffusion.cuda()

    trainer = Trainer(
        diffusion,
        mode=args.mode,
        folder=str(folder) + "/",
        image_size=(args.image_size, args.image_size),
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_steps=args.steps,
        gradient_accumulate_every=args.grad_acc,
        amp=args.amp,
        save_and_sample_every_image=300,
        save_and_sample_every_model=500,
    )

    trainer.train()


if __name__ == "__main__":
    main()
