#!/usr/bin/env python3
"""Evaluate a trained picker checkpoint on a dataset split and write metrics report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--picker-ckpt", type=str, required=True)
    ap.add_argument("--denoiser-ckpt", type=str, default="")
    ap.add_argument("--out", type=str, default="runs/eval")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--K-max", type=int, default=5)
    ap.add_argument("--norm", type=str, default="log1p_minmax")
    ap.add_argument("--hit-tol", type=float, default=20.0)
    ap.add_argument("--decode", type=str, default="dp", choices=["dp", "softargmax"])
    ap.add_argument("--softargmax-power", type=float, default=1.0)
    args = ap.parse_args()

    torch = _require_torch()

    from diffdisp.data import DatasetConfig, NPZDispersionDataset
    from diffdisp.models import UNet2D
    from diffdisp.normalize import NormConfig
    from diffdisp.path import PathConfig, extract_curve_dp, extract_curve_softargmax
    from diffdisp.metrics import MetricsConfig, compute_metrics_for_matched, match_modes_mae

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ds = NPZDispersionDataset(
        DatasetConfig(
            dataset_root=Path(args.dataset_root),
            split=args.split,
            use_input="noisy",
            norm=NormConfig(kind=args.norm),
            K_max=args.K_max,
            return_prob_maps=False,
            return_clean_target=False,
        )
    )
    c_axis = ds.grid.c_axis()

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

    path_cfg = PathConfig(max_jump=10, lambda_smooth=1.0, const_null=2.0)
    metrics_cfg = MetricsConfig(hit_tol_ms=float(args.hit_tol))

    n = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i in range(n):
            b = ds[i]
            x = b["x"][None, ...].to(device)  # [1,1,F,C]
            if denoiser is not None:
                x = denoiser(x).clamp_(0.0, 1.0)

            logits = picker(x)[0].detach().cpu().numpy().astype(np.float32)  # [K,F,C]
            P = 1.0 / (1.0 + np.exp(-logits))
            pred_curves = np.full((args.K_max, ds.grid.F), np.nan, dtype=np.float32)
            for k in range(args.K_max):
                if args.decode == "softargmax":
                    pred_curves[k] = extract_curve_softargmax(P[k], c_axis, power=args.softargmax_power)
                else:
                    pred_curves[k], _ = extract_curve_dp(P[k], c_axis, path_cfg)

            gt_curves = b["curve"].detach().cpu().numpy().astype(np.float32)[: args.K_max]
            gt_mask = b["mode_mask"].detach().cpu().numpy().astype(np.uint8)[: args.K_max]

            r, c = match_modes_mae(pred_curves, gt_curves, pred_mask_k=np.ones(args.K_max, np.uint8), gt_mask_k=gt_mask)
            for rr, cc in zip(r, c):
                m = compute_metrics_for_matched(pred_curves[int(rr)], gt_curves[int(cc)], metrics_cfg)
                m.update({"sample": b["path"], "pred_mode": int(rr), "gt_mode": int(cc)})
                rows.append(m)

    agg = {}
    if rows:
        for key in ["mae_ms", "hit_at_tol", "coverage", "break_rate", "smoothness"]:
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            agg[key] = float(np.mean(vals)) if vals else float("nan")
    agg["n_pairs"] = len(rows)
    (out_root / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")

    with (out_root / "per_pair.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sample", "pred_mode", "gt_mode", "mae_ms", "hit_at_tol", "coverage", "break_rate", "smoothness"],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Evaluated {n} sample(s); wrote report to {out_root}")


if __name__ == "__main__":
    main()
