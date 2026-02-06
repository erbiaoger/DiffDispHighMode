#!/usr/bin/env python3
"""Train Stage-B energy denoiser: E_noisy -> E_clean (numeric matrices)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def _require_torch():
    try:
        import torch

        return torch
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing dependency: torch. Create env from environment.yml first.") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True, help="e.g. data/demultiple/npz")
    ap.add_argument("--out", type=str, default="runs/denoiser", help="output dir")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--norm", type=str, default="log1p_minmax")
    args = ap.parse_args()

    torch = _require_torch()
    from torch.utils.data import DataLoader
    from torch import nn
    from torch.optim import AdamW

    from diffdisp.data import DatasetConfig, NPZDispersionDataset
    from diffdisp.models import UNet2D
    from diffdisp.normalize import NormConfig

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train_ds = NPZDispersionDataset(
        DatasetConfig(
            dataset_root=dataset_root,
            split="train",
            use_input="noisy",
            norm=NormConfig(kind=args.norm),
            return_prob_maps=False,
            return_clean_target=True,
        )
    )
    val_ds = NPZDispersionDataset(
        DatasetConfig(
            dataset_root=dataset_root,
            split="val",
            use_input="noisy",
            norm=NormConfig(kind=args.norm),
            return_prob_maps=False,
            return_clean_target=True,
        )
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = UNet2D(in_ch=1, out_ch=1, base=32).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    cfg_dump = {
        "dataset_root": str(dataset_root),
        "norm": args.norm,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
    }
    (out_root / "config.json").write_text(json.dumps(cfg_dump, indent=2) + "\n", encoding="utf-8")

    step = 0
    train_iter = iter(train_loader)

    def run_val() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)  # noisy
                y = batch["x_clean"].to(device)
                yhat = model(x)
                losses.append(float(loss_fn(yhat, y).item()))
        model.train()
        return float(np.mean(losses)) if losses else float("nan")

    model.train()
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch["x"].to(device)
        y = batch["x_clean"].to(device)

        yhat = model(x)
        loss = loss_fn(yhat, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"[step {step}] train_l1={loss.item():.6f}")

        if step != 0 and step % args.val_every == 0:
            v = run_val()
            print(f"[step {step}] val_l1={v:.6f}")

        if step != 0 and step % args.save_every == 0:
            ckpt = {
                "step": step,
                "state_dict": model.state_dict(),
                "config": cfg_dump,
            }
            torch.save(ckpt, out_root / f"denoiser_step_{step:07d}.pt")

        step += 1

    ckpt = {"step": step, "state_dict": model.state_dict(), "config": cfg_dump}
    torch.save(ckpt, out_root / "denoiser_final.pt")
    print(f"Saved final checkpoint to {out_root / 'denoiser_final.pt'}")


if __name__ == "__main__":
    main()
