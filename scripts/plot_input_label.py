#!/usr/bin/env python3
"""Plot input energy matrix with GT label curves (no model inference)."""

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-npz", type=str, default="", help="path to a single .npz file")
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--random", action="store_true", help="pick a random .npz from dataset split")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n", type=int, default=1, help="number of random samples to plot")
    ap.add_argument("--out", type=str, default="runs/plots")
    ap.add_argument("--use-clean", action="store_true", help="plot E_clean instead of E_noisy")
    ap.add_argument("--per-f", action="store_true", help="normalize each frequency slice for visibility")
    ap.add_argument("--log1p", action="store_true", help="apply log1p before plotting")
    ap.add_argument("--K-max", type=int, default=5)
    args = ap.parse_args()

    from diffdisp.axes import GridSpec
    from diffdisp.io import load_sample_npz

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

    colors = ["#00B3B3", "#F28E2B", "#E15759", "#59A14F", "#B07AA1", "#4E79A7"]

    for input_path in input_paths:
        sample = load_sample_npz(input_path)
        E = sample["E_clean"] if args.use_clean else sample["E_noisy"]
        E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if args.log1p:
            E = np.log1p(np.maximum(E, 0.0))

        if args.per_f:
            Emin = E.min(axis=1, keepdims=True)
            Emax = E.max(axis=1, keepdims=True)
            E = (E - Emin) / (Emax - Emin + 1e-8)

        mask = E > 0
        if np.any(mask):
            vmin = np.percentile(E[mask], 2)
            vmax = np.percentile(E[mask], 98)
        else:
            vmin, vmax = float(np.min(E)), float(np.max(E) + 1e-6)

        gt_curves = sample["Y_curve_fc"][: args.K_max]
        gt_mask = sample["mode_mask"][: args.K_max]

        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.imshow(
            E.T,
            origin="lower",
            aspect="auto",
            extent=(f_axis.min(), f_axis.max(), c_axis.min(), c_axis.max()),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        for k in range(args.K_max):
            if gt_mask[k] == 1:
                ax.plot(f_axis, gt_curves[k], color=colors[k % len(colors)], lw=2.0, label=f"GT {k}")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase Velocity (m/s)")
        ax.set_title(input_path.name + (" (clean)" if args.use_clean else " (noisy)"))
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()

        out_path = out_root / (input_path.stem + "_input_label.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
