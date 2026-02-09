#!/usr/bin/env python3
"""Train Stage-C picker: E -> P_k(f,c) and evaluate via DP path extraction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def _require_torch():
    try:
        import torch

        return torch
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing dependency: torch. Create env from environment.yml first.") from e


def dice_loss(prob: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor", eps: float = 1e-6):
    # prob/target: [B,K,F,C], mask: [B,K,F,1]
    prob = prob * mask
    target = target * mask
    inter = (prob * target).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True, help="e.g. data/demultiple/npz")
    ap.add_argument("--out", type=str, default="runs/picker", help="output dir")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-samples", type=int, default=64)

    ap.add_argument("--K-max", type=int, default=5)
    ap.add_argument("--sigma-px", type=float, default=5.0)
    ap.add_argument("--norm", type=str, default="log1p_minmax")

    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--hit-tol", type=float, default=20.0)
    ap.add_argument("--dp-max-jump", type=int, default=10)  # 最大跳跃次数
    ap.add_argument("--dp-lambda-smooth", type=float, default=1.0)
    ap.add_argument("--dp-const-null", type=float, default=2.0)
    ap.add_argument("--dp-null-entry", type=float, default=0.5)
    ap.add_argument("--dp-null-exit", type=float, default=0.5)
    ap.add_argument("--dp-null-stay", type=float, default=0.0)
    ap.add_argument("--decode", type=str, default="dp", choices=["dp", "softargmax"])
    ap.add_argument("--softargmax-power", type=float, default=1.0)

    ap.add_argument("--denoiser-ckpt", type=str, default="", help="optional denoiser checkpoint (.pt)")
    args = ap.parse_args()

    torch = _require_torch()
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    import torch.nn.functional as F

    from diffdisp.data import DatasetConfig, NPZDispersionDataset
    from diffdisp.models import UNet2D
    from diffdisp.normalize import NormConfig
    from diffdisp.path import PathConfig, extract_curve_dp, extract_curve_softargmax
    from diffdisp.metrics import MetricsConfig, compute_metrics_for_matched, match_modes_mae

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train_ds = NPZDispersionDataset(
        DatasetConfig(
            dataset_root=dataset_root,
            split="train",
            use_input="noisy",
            norm=NormConfig(kind=args.norm),
            sigma_px=args.sigma_px,
            K_max=args.K_max,
            return_prob_maps=True,
            return_clean_target=False,
        )
    )
    val_ds = NPZDispersionDataset(
        DatasetConfig(
            dataset_root=dataset_root,
            split="val",
            use_input="noisy",
            norm=NormConfig(kind=args.norm),
            sigma_px=args.sigma_px,
            K_max=args.K_max,
            return_prob_maps=True,
            return_clean_target=False,
        )
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    picker = UNet2D(in_ch=1, out_ch=args.K_max, base=32).to(device)
    opt = AdamW(picker.parameters(), lr=args.lr)

    denoiser = None
    if args.denoiser_ckpt:
        ckpt = torch.load(args.denoiser_ckpt, map_location="cpu")
        denoiser = UNet2D(in_ch=1, out_ch=1, base=32)
        denoiser.load_state_dict(ckpt["state_dict"])
        denoiser.to(device).eval()
        for p in denoiser.parameters():
            p.requires_grad_(False)

    cfg_dump = {
        "dataset_root": str(dataset_root),
        "norm": args.norm,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "K_max": args.K_max,
        "sigma_px": args.sigma_px,
        "denoiser_ckpt": args.denoiser_ckpt,
        "dp": {
            "max_jump": args.dp_max_jump,
            "lambda_smooth": args.dp_lambda_smooth,
            "const_null": args.dp_const_null,
            "null_entry": args.dp_null_entry,
            "null_exit": args.dp_null_exit,
            "null_stay": args.dp_null_stay,
        },
    }
    (out_root / "config.json").write_text(json.dumps(cfg_dump, indent=2) + "\n", encoding="utf-8")

    grid = train_ds.grid
    c_axis = grid.c_axis()

    path_cfg = PathConfig(
        max_jump=args.dp_max_jump,
        lambda_smooth=args.dp_lambda_smooth,
        const_null=args.dp_const_null,
        null_entry=args.dp_null_entry,
        null_exit=args.dp_null_exit,
        null_stay=args.dp_null_stay,
    )
    metrics_cfg = MetricsConfig(hit_tol_ms=float(args.hit_tol))

    train_iter = iter(train_loader)
    step = 0

    def run_val() -> Dict[str, Any]:
        picker.eval()
        rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= args.val_samples:
                    break
                x = batch["x"].to(device)  # [1,1,F,C]
                if denoiser is not None:
                    x = denoiser(x).clamp_(0.0, 1.0)
                logits = picker(x)[0].detach().cpu().numpy().astype(np.float32)  # [K,F,C]
                P = 1.0 / (1.0 + np.exp(-logits))
                pred_curves = np.full((args.K_max, grid.F), np.nan, dtype=np.float32)
                for k in range(args.K_max):
                    if args.decode == "softargmax":
                        pred_curves[k] = extract_curve_softargmax(P[k], c_axis, power=args.softargmax_power)
                    else:
                        pred_curves[k], _ = extract_curve_dp(P[k], c_axis, path_cfg)

                gt_curves = batch["curve"][0].detach().cpu().numpy().astype(np.float32)[: args.K_max]
                gt_mask = batch["mode_mask"][0].detach().cpu().numpy().astype(np.uint8)[: args.K_max]

                r, c = match_modes_mae(pred_curves, gt_curves, pred_mask_k=np.ones(args.K_max, np.uint8), gt_mask_k=gt_mask)
                for rr, cc in zip(r, c):
                    m = compute_metrics_for_matched(pred_curves[int(rr)], gt_curves[int(cc)], metrics_cfg)
                    m.update({"pred_mode": int(rr), "gt_mode": int(cc), "sample": batch["path"][0]})
                    rows.append(m)

        picker.train()
        # aggregate
        agg = {}
        if rows:
            for key in ["mae_ms", "hit_at_tol", "coverage", "break_rate", "smoothness"]:
                vals = [r[key] for r in rows if np.isfinite(r[key])]
                agg[key] = float(np.mean(vals)) if vals else float("nan")
        agg["n_pairs"] = len(rows)
        return {"aggregate": agg, "rows": rows}

    def write_metrics(step_i: int, rep: Dict[str, Any]) -> None:
        out_json = out_root / f"metrics_step_{step_i:07d}.json"
        out_csv = out_root / f"metrics_step_{step_i:07d}.csv"
        out_json.write_text(json.dumps(rep["aggregate"], indent=2) + "\n", encoding="utf-8")

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["sample", "pred_mode", "gt_mode", "mae_ms", "hit_at_tol", "coverage", "break_rate", "smoothness"],
            )
            w.writeheader()
            for row in rep["rows"]:
                w.writerow(row)

    picker.train()
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch["x"].to(device)  # [B,1,F,C]
        if denoiser is not None:
            with torch.no_grad():
                x = denoiser(x).clamp_(0.0, 1.0)

        y = batch["y_map"].to(device)  # [B,K,F,C]
        valid_kf = batch["valid_kf"].to(device).float()  # [B,K,F]
        mode_mask = batch["mode_mask"].to(device).float()[:, : args.K_max]  # [B,K]

        logits = picker(x)  # [B,K,F,C]
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        w_fk = (mode_mask[:, :, None] * valid_kf).unsqueeze(-1)  # [B,K,F,1]
        w_sum = w_fk.sum()
        if float(w_sum.detach().cpu().item()) <= 0.0:
            # No supervised points in this batch (e.g., no valid modes). Skip update.
            if step % 50 == 0:
                print(f"[step {step}] warning: empty supervision mask; skipping batch")
            step += 1
            continue
        bce = (bce * w_fk).sum() / (w_sum + 1e-6)

        prob = torch.sigmoid(logits)
        d = dice_loss(prob, y, w_fk)
        d = (d * mode_mask).sum() / (mode_mask.sum() + 1e-6)

        loss = args.bce_weight * bce + args.dice_weight * d

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"[step {step}] loss={loss.item():.6f} bce={bce.item():.6f} dice={d.item():.6f}")

        if step != 0 and step % args.val_every == 0:
            rep = run_val()
            print(f"[step {step}] val: {rep['aggregate']}")
            write_metrics(step, rep)

        if step != 0 and step % args.save_every == 0:
            ckpt = {"step": step, "state_dict": picker.state_dict(), "config": cfg_dump}
            torch.save(ckpt, out_root / f"picker_step_{step:07d}.pt")

        step += 1

    ckpt = {"step": step, "state_dict": picker.state_dict(), "config": cfg_dump}
    torch.save(ckpt, out_root / "picker_final.pt")
    print(f"Saved final checkpoint to {out_root / 'picker_final.pt'}")


if __name__ == "__main__":
    main()
