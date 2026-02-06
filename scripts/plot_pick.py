#!/usr/bin/env python3
"""Plot energy image with predicted and GT curves overlaid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    ap.add_argument("--input-npz", type=str, default="", help="path to a single .npz file")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--random", action="store_true", help="pick a random .npz from dataset split")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n", type=int, default=1, help="number of random samples to plot")
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--picker-ckpt", type=str, required=True)
    ap.add_argument("--denoiser-ckpt", type=str, default="")
    ap.add_argument("--out", type=str, default="runs/plots")
    ap.add_argument("--K-max", type=int, default=5)
    ap.add_argument("--norm", type=str, default="log1p_minmax")
    ap.add_argument("--use-clean", action="store_true", help="plot E_clean instead of E_noisy")
    ap.add_argument("--plot-norm", action="store_true", help="plot normalized energy instead of raw log1p")
    args = ap.parse_args()

    torch = _require_torch()

    from diffdisp.axes import GridSpec
    from diffdisp.io import load_sample_npz
    from diffdisp.models import UNet2D
    from diffdisp.normalize import NormConfig, normalize_energy
    from diffdisp.path import PathConfig, extract_curve_dp

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = json.loads((Path(args.dataset_root) / "meta.json").read_text(encoding="utf-8"))
    grid = GridSpec.from_dict(meta["grid"])
    f_axis = grid.f_axis()
    c_axis = grid.c_axis()

    rng = np.random.default_rng(args.seed)
    if args.random:
        candidates = sorted((Path(args.dataset_root) / args.split).glob("*.npz"))
        if not candidates:
            raise SystemExit(f"No .npz files found under {Path(args.dataset_root) / args.split}")
        n = max(1, min(args.n, len(candidates)))
        picks = rng.choice(candidates, size=n, replace=False)
        input_paths = [Path(p) for p in picks]
    else:
        if not args.input_npz:
            raise SystemExit("--input-npz is required unless --random is set")
        input_paths = [Path(args.input_npz)]

    for input_path in input_paths:
        sample = load_sample_npz(input_path)
        E = sample["E_clean"] if args.use_clean else sample["E_noisy"]
    # Model input normalization
    E_norm = normalize_energy(E, NormConfig(kind=args.norm))
    # Plotting: use log1p + robust stretch, or normalized if requested
    if args.plot_norm:
        E_plot = E_norm.astype(np.float32)
    else:
        E_plot = np.log1p(np.maximum(E, 0.0)).astype(np.float32)
    E_plot = np.nan_to_num(E_plot, nan=0.0, posinf=0.0, neginf=0.0)
    mask = E_plot > 0
    if np.any(mask):
        vmin = np.percentile(E_plot[mask], 2)
        vmax = np.percentile(E_plot[mask], 98)
    else:
        vmin, vmax = float(np.min(E_plot)), float(np.max(E_plot) + 1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    picker_ckpt = torch.load(args.picker_ckpt, map_location="cpu")
    picker = UNet2D(in_ch=1, out_ch=args.K_max, base=32)
    picker.load_state_dict(picker_ckpt["state_dict"])
    picker.to(device).eval()

    denoiser = None
    if args.denoiser_ckpt:
        den_ckpt = torch.load(args.denoiser_ckpt, map_location="cpu")
        denoiser = UNet2D(in_ch=1, out_ch=1, base=32)
        denoiser.load_state_dict(den_ckpt["state_dict"])
        denoiser.to(device).eval()

        x = torch.from_numpy(E_norm[None, None, :, :].astype(np.float32)).to(device)
        with torch.no_grad():
            if denoiser is not None:
                x = denoiser(x).clamp_(0.0, 1.0)
            logits = picker(x)[0].detach().cpu().numpy().astype(np.float32)
        P = 1.0 / (1.0 + np.exp(-logits))

        path_cfg = PathConfig(max_jump=10, lambda_smooth=1.0, const_null=2.0)
        pred_curves = []
        for k in range(args.K_max):
            curve, _ = extract_curve_dp(P[k], c_axis, path_cfg)
            pred_curves.append(curve)

        gt_curves = sample["Y_curve_fc"][: args.K_max]
        gt_mask = sample["mode_mask"][: args.K_max]

        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.imshow(
            E_plot.T,
            origin="lower",
            aspect="auto",
            extent=(f_axis.min(), f_axis.max(), c_axis.min(), c_axis.max()),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        colors = ["#00B3B3", "#F28E2B", "#E15759", "#59A14F", "#B07AA1", "#4E79A7"]
        for k in range(args.K_max):
            if gt_mask[k] == 1:
                ax.plot(f_axis, gt_curves[k], color=colors[k % len(colors)], lw=2.0, label=f"GT {k}")
            ax.plot(f_axis, pred_curves[k], color=colors[k % len(colors)], lw=1.2, ls="--", label=f"Pred {k}")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase Velocity (m/s)")
        ax.set_title(input_path.name)
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)

        out_path = out_root / (input_path.stem + "_pick.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
