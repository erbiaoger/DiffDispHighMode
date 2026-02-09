#!/usr/bin/env python3
"""Run picker (and optional denoiser) and extract curves via DP; save JSON/CSV outputs."""

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
    ap.add_argument("--picker-ckpt", type=str, required=True)
    ap.add_argument("--denoiser-ckpt", type=str, default="")
    ap.add_argument("--dataset-root", type=str, default="", help="e.g. data/demultiple/npz")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--input-npz", type=str, default="", help="single sample .npz (overrides dataset-root)")
    ap.add_argument("--out", type=str, default="runs/infer")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--K-max", type=int, default=5)
    ap.add_argument("--norm", type=str, default="log1p_minmax")
    ap.add_argument("--decode", type=str, default="dp", choices=["dp", "softargmax"])
    ap.add_argument("--softargmax-power", type=float, default=1.0)
    ap.add_argument("--conf-thresh", type=float, default=0.0, help="softargmax confidence threshold")
    args = ap.parse_args()

    torch = _require_torch()

    from diffdisp.data import DatasetConfig, NPZDispersionDataset, load_meta
    from diffdisp.models import UNet2D
    from diffdisp.normalize import NormConfig
    from diffdisp.path import PathConfig, extract_curve_dp, extract_curve_softargmax
    from diffdisp.io import load_sample_npz
    from diffdisp.axes import GridSpec

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

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

    records: List[Dict[str, Any]] = []

    if args.input_npz:
        s = load_sample_npz(Path(args.input_npz))
        meta = load_meta(Path(args.dataset_root)) if args.dataset_root else None
        if meta is None and args.dataset_root == "":
            raise SystemExit("--dataset-root is required when using --input-npz (to get grid axes)")
        grid = GridSpec.from_dict(meta["grid"])
        f_axis = grid.f_axis()
        c_axis = grid.c_axis()
        x = s["E_noisy"]
        from diffdisp.normalize import normalize_energy

        x = normalize_energy(x, NormConfig(kind=args.norm))
        xt = torch.from_numpy(x[None, None, :, :]).to(device)
        with torch.no_grad():
            if denoiser is not None:
                xt = denoiser(xt).clamp_(0.0, 1.0)
            logits = picker(xt)[0].detach().cpu().numpy()
        P = 1.0 / (1.0 + np.exp(-logits))
        curves = []
        for k in range(args.K_max):
            if args.decode == "softargmax":
                curve = extract_curve_softargmax(P[k], c_axis, power=args.softargmax_power, conf_thresh=args.conf_thresh)
            else:
                curve, _ = extract_curve_dp(P[k], c_axis, path_cfg)
            curves.append(curve.tolist())
        rec = {"file": str(args.input_npz), "curves_ms": curves}
        records.append(rec)
    else:
        if not args.dataset_root:
            raise SystemExit("Provide either --input-npz or --dataset-root")
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
        f_axis = ds.grid.f_axis()
        c_axis = ds.grid.c_axis()

        n = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)
        for i in range(n):
            b = ds[i]
            xt = b["x"][None, ...].to(device)  # [1,1,F,C]
            with torch.no_grad():
                if denoiser is not None:
                    xt = denoiser(xt).clamp_(0.0, 1.0)
                logits = picker(xt)[0].detach().cpu().numpy()
            P = 1.0 / (1.0 + np.exp(-logits))
            curves = []
            for k in range(args.K_max):
                if args.decode == "softargmax":
                    curve = extract_curve_softargmax(P[k], c_axis, power=args.softargmax_power, conf_thresh=args.conf_thresh)
                else:
                    curve, _ = extract_curve_dp(P[k], c_axis, path_cfg)
                curves.append(curve.tolist())
            records.append({"file": b["path"], "curves_ms": curves})

    # Write JSONL
    jsonl_path = out_root / "curves.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Write a single CSV with all samples concatenated (one sample per block)
    csv_path = out_root / "curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "f_hz"] + [f"mode_{k}_c_ms" for k in range(args.K_max)]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            for fi, f0 in enumerate(f_axis.tolist()):
                row = {"file": r["file"], "f_hz": f0}
                for k in range(args.K_max):
                    row[f"mode_{k}_c_ms"] = r["curves_ms"][k][fi]
                w.writerow(row)

    print(f"Wrote {len(records)} sample(s) to {out_root}")


if __name__ == "__main__":
    main()
