#!/usr/bin/env python3
"""Generate numeric dispersion dataset (.npz) with probability-map-ready labels.

Outputs:
  <out>/
    meta.json
    manifest.jsonl
    train/sample_000000.npz ...
    val/sample_000000.npz ...

Each sample contains:
  E_clean   [F,C] float32  (raw, resampled to fixed grid)
  E_noisy   [F,C] float32  (raw, resampled to fixed grid)
  Y_curve_fc [K,F] float32 (m/s, NaN for invalid)
  mode_mask [K] uint8
  meta      json string
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from diffdisp.axes import GridSpec
from diffdisp.augment import (
    EnergyAugmentConfig,
    RecordAugmentConfig,
    apply_energy_randomization,
    apply_record_randomization,
)
from diffdisp.io import append_jsonl, save_json, save_sample_npz
from diffdisp.labels import cpr_to_curve_on_f_axis
from diffdisp.park import park_energy
from diffdisp.resample import resample_fc
from diffdisp.synth import multi_mode_gather, ormsby, ricker


def _require_disba():
    try:
        from disba import PhaseDispersion  # type: ignore

        return PhaseDispersion
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: disba. Install it (see environment.yml/requirements.txt) "
            "or use your conda environment that already has disba."
        ) from e


def _make_wavelet(
    *,
    t_pos: np.ndarray,
    rng: np.random.Generator,
    kind: str,
    f0_min: float,
    f0_max: float,
) -> np.ndarray:
    if kind == "ormsby":
        # random trapezoid within plausible seismic band
        f1 = float(rng.uniform(2.0, 6.0))
        f2 = float(rng.uniform(f1 + 0.5, 10.0))
        f3 = float(rng.uniform(20.0, 45.0))
        f4 = float(rng.uniform(f3 + 0.5, 55.0))
        w, _, _ = ormsby(t_pos, f=(f1, f2, f3, f4), taper=np.hanning)
        # keep only causal-ish part length like original notebook
        w = np.roll(np.fft.ifftshift(w), 20).astype(np.float32)
        return w
    if kind == "ricker":
        f0 = float(rng.uniform(f0_min, f0_max))
        # build symmetric wavelet then shift to causal-ish
        t = np.concatenate((np.flipud(-t_pos[1:]), t_pos), axis=0)
        w = ricker(t, f0).astype(np.float32)
        w = w / (np.max(np.abs(w)) + 1e-12)
        w = np.roll(np.fft.ifftshift(w), 20).astype(np.float32)
        return w
    raise ValueError(f"unknown wavelet kind: {kind}")


def _sample_layer_model(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # Defaults follow existing repo notebooks (km, km/s).
    thick_km = np.array(
        [
            rng.uniform(0.005, 0.015),
            rng.uniform(0.010, 0.030),
            rng.uniform(0.020, 0.050),
            1.0,  # deep layer / half-space proxy used in the original code
        ],
        dtype=np.float32,
    )
    vs_km_s = np.array(
        [
            rng.uniform(0.10, 0.30),
            rng.uniform(0.30, 0.60),
            rng.uniform(0.30, 0.70),
            rng.uniform(0.70, 0.90),
        ],
        dtype=np.float32,
    )
    return thick_km, vs_km_s


def _jitter_params(
    thick_km: np.ndarray,
    vs_km_s: np.ndarray,
    rng: np.random.Generator,
    fluct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t = thick_km * (1 + fluct * (2 * rng.random(len(thick_km)) - 1)).astype(np.float32)
    v = vs_km_s * (1 + fluct * (2 * rng.random(len(vs_km_s)) - 1)).astype(np.float32)
    return t.astype(np.float32), v.astype(np.float32)


def _compute_cpr(
    thick_km: np.ndarray,
    vs_km_s: np.ndarray,
    periods_s_increasing: np.ndarray,
    K_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    PhaseDispersion = _require_disba()
    # disba (numba) can be sensitive to dtype mixing; use float64 consistently here.
    thick64 = np.asarray(thick_km, dtype=np.float64)
    vs64 = np.asarray(vs_km_s, dtype=np.float64)
    vp64 = vs64 * 4.0
    rho64 = np.ones_like(vs64, dtype=np.float64)
    per64 = np.asarray(periods_s_increasing, dtype=np.float64)

    pd = PhaseDispersion(thick64, vp64, vs64, rho64)

    curves_kf = np.full((K_max, len(periods_s_increasing)), np.nan, dtype=np.float32)
    mode_mask = np.zeros((K_max,), dtype=np.uint8)

    for k in range(K_max):
        try:
            out = pd(per64, mode=k, wave="rayleigh")

            # disba>=0.7 typically returns a DispersionCurve object with .period/.velocity.
            if hasattr(out, "period") and hasattr(out, "velocity"):
                per = np.asarray(out.period, dtype=np.float64)
                vel = np.asarray(out.velocity, dtype=np.float64)
            else:
                # older disba versions returned (period, velocity)
                per, vel = out  # type: ignore[misc]
                per = np.asarray(per, dtype=np.float64)
                vel = np.asarray(vel, dtype=np.float64)

            if vel.shape[0] != len(periods_s_increasing):
                # If disba changes the sampling, interpolate back to our period grid.
                curves_kf[k] = np.interp(per64, per, vel, left=np.nan, right=np.nan).astype(np.float32) * 1e3
            else:
                curves_kf[k] = vel.astype(np.float32) * 1e3  # km/s -> m/s
            mode_mask[k] = 1
        except Exception:
            mode_mask[k] = 0

    # Return raw c(period) and mask; mapping to f-axis is done later
    return curves_kf, mode_mask


def _period_curve_to_f_axis(curve_kp_ms: np.ndarray, periods_s_inc: np.ndarray, f_axis_hz: np.ndarray) -> np.ndarray:
    # periods are increasing, so freq is decreasing; flip to increasing for interpolation
    freq_inc = np.flipud(1.0 / (periods_s_inc + 1e-12)).astype(np.float32)
    out = np.full((curve_kp_ms.shape[0], len(f_axis_hz)), np.nan, dtype=np.float32)
    for k in range(curve_kp_ms.shape[0]):
        v_inc = np.flipud(curve_kp_ms[k]).astype(np.float32)
        out[k] = np.interp(f_axis_hz, freq_inc, v_inc, left=np.nan, right=np.nan).astype(np.float32)
    return out


def _fill_nan_1d(x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Fill NaNs in x by linear interpolation over xp (assumed increasing)."""
    x = np.asarray(x, dtype=np.float32)
    xp = np.asarray(xp, dtype=np.float32)
    mask = np.isfinite(x)
    if not np.any(mask):
        return x
    if np.sum(mask) == 1:
        return np.full_like(x, x[mask][0], dtype=np.float32)
    filled = np.interp(xp, xp[mask], x[mask]).astype(np.float32)
    return filled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="output dataset root, e.g. data/demultiple/npz")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-val", type=int, default=200)
    ap.add_argument("--overwrite", action="store_true", help="delete existing split folders before writing")

    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--fmin", type=float, default=3.0)
    ap.add_argument("--fmax", type=float, default=12.0)
    ap.add_argument("--cmin", type=float, default=100.0)
    ap.add_argument("--cmax", type=float, default=900.0)

    ap.add_argument("--K-max", type=int, default=5)
    ap.add_argument("--sigma-px", type=float, default=3.0)

    # synthesis grid
    ap.add_argument("--nt", type=int, default=600)
    ap.add_argument("--dt", type=float, default=0.008)
    ap.add_argument("--nx", type=int, default=201)
    ap.add_argument("--dx", type=float, default=2.0)
    ap.add_argument("--nfft", type=int, default=1024)

    # Park energy params (base; per-sample jitter will be applied)
    ap.add_argument("--dc", type=float, default=2.0)
    ap.add_argument("--param-fluct", type=float, default=0.2)
    ap.add_argument("--f-jitter", type=float, default=0.4, help="± jitter for fmin/fmax used in Park energy")
    ap.add_argument("--c-jitter", type=float, default=40.0, help="± jitter for cmin/cmax used in Park energy")
    ap.add_argument("--dc-choices", type=str, default="1,2,3,5")

    # wavelet
    ap.add_argument("--wavelet", type=str, default="mix", choices=["mix", "ormsby", "ricker"])
    ap.add_argument("--ricker-f0-min", type=float, default=6.0)
    ap.add_argument("--ricker-f0-max", type=float, default=22.0)

    args = ap.parse_args()

    out_root = Path(args.out)
    if args.overwrite and out_root.exists():
        for sub in ["train", "val"]:
            p = out_root / sub
            if p.exists():
                for fp in p.glob("*.npz"):
                    fp.unlink()
        m = out_root / "manifest.jsonl"
        if m.exists():
            m.unlink()
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    grid = GridSpec(
        fmin_hz=float(args.fmin),
        fmax_hz=float(args.fmax),
        F=int(args.F),
        cmin_ms=float(args.cmin),
        cmax_ms=float(args.cmax),
        C=int(args.C),
    )
    f_axis = grid.f_axis()
    c_axis = grid.c_axis()
    periods_inc = np.flipud(1.0 / (f_axis + 1e-12)).astype(np.float32)  # increasing

    meta = {
        "grid": grid.to_dict(),
        "K_max": int(args.K_max),
        "sigma_px": float(args.sigma_px),
        "synth": {"nt": int(args.nt), "dt": float(args.dt), "nx": int(args.nx), "dx": float(args.dx), "nfft": int(args.nfft)},
        "energy": {"method": "park", "dc": float(args.dc), "f_jitter": float(args.f_jitter), "c_jitter": float(args.c_jitter), "dc_choices": args.dc_choices},
    }
    save_json(out_root / "meta.json", meta)

    manifest_path = out_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    rec_aug_cfg = RecordAugmentConfig()
    eng_aug_cfg = EnergyAugmentConfig()

    dc_choices = [float(x) for x in args.dc_choices.split(",") if x.strip()]

    def make_one(split: str, idx: int) -> None:
        thick, vs = _sample_layer_model(rng)
        thick, vs = _jitter_params(thick, vs, rng, args.param_fluct)

        # wavelet
        t_pos = (np.arange(args.nt // 2 + 1, dtype=np.float32) * float(args.dt)).astype(np.float32)
        if args.wavelet == "mix":
            w_kind = "ormsby" if rng.random() < 0.5 else "ricker"
        else:
            w_kind = args.wavelet
        wav = _make_wavelet(t_pos=t_pos, rng=rng, kind=w_kind, f0_min=args.ricker_f0_min, f0_max=args.ricker_f0_max)

        # c(period) for modes
        curve_kp_ms, mode_mask = _compute_cpr(thick, vs, periods_inc, args.K_max)
        # map to fixed f-axis (m/s)
        Y_curve_fc = _period_curve_to_f_axis(curve_kp_ms, periods_inc, f_axis)

        # synth clean gather (use mode weights like existing notebooks)
        modes_fdisp = []
        modes_vdisp = []
        mode_weights = []
        for k in range(args.K_max):
            if mode_mask[k] == 0:
                continue
            # For synthesis, we need fdisp increasing and vdisp in km/s.
            # We use the same period grid, convert to fdisp (increasing).
            fdisp_inc = np.flipud(1.0 / (periods_inc + 1e-12)).astype(np.float32)
            vdisp_km_s_inc = (np.flipud(curve_kp_ms[k]) / 1e3).astype(np.float32)
            vdisp_km_s_inc = _fill_nan_1d(vdisp_km_s_inc, fdisp_inc)
            vdisp_km_s_inc = np.maximum(vdisp_km_s_inc, 0.05).astype(np.float32)
            if not np.all(np.isfinite(vdisp_km_s_inc)):
                continue
            modes_fdisp.append(fdisp_inc)
            modes_vdisp.append(vdisp_km_s_inc)
            mode_weights.append(1.0 / float((k + 1) ** 0.8))

        if not modes_fdisp:
            # If even the fundamental fails, skip by writing an all-zero sample with mask=0.
            E0 = np.zeros((args.F, args.C), dtype=np.float32)
            Y = np.full((args.K_max, args.F), np.nan, dtype=np.float32)
            mm = np.zeros((args.K_max,), dtype=np.uint8)
            save_sample_npz(
                (train_dir if split == "train" else val_dir) / f"sample_{idx:06d}.npz",
                E_clean=E0,
                E_noisy=E0,
                Y_curve_fc=Y,
                mode_mask=mm,
                meta={"error": "no_valid_modes"},
            )
            return

        d_clean = multi_mode_gather(
            nt=args.nt,
            dt=args.dt,
            nx=args.nx,
            dx=args.dx,
            nfft=args.nfft,
            modes_fdisp=modes_fdisp,
            modes_vdisp_km_s=modes_vdisp,
            wav=wav,
            mode_weights=mode_weights,
        )
        if not np.all(np.isfinite(d_clean)):
            # Rarely, synthesis can blow up if vf contains zeros/NaNs. Skip this sample.
            save_sample_npz(
                (train_dir if split == "train" else val_dir) / f"sample_{idx:06d}.npz",
                E_clean=np.zeros((args.F, args.C), dtype=np.float32),
                E_noisy=np.zeros((args.F, args.C), dtype=np.float32),
                Y_curve_fc=Y_curve_fc,
                mode_mask=mode_mask,
                meta={"error": "nonfinite_synth"},
            )
            return

        d_noisy, rec_meta = apply_record_randomization(d_clean, dt_s=args.dt, dx_m=args.dx, rng=rng, cfg=rec_aug_cfg)

        # Park energy params randomization (affects both clean/noisy for realism)
        fmin_e = float(np.clip(args.fmin + rng.uniform(-args.f_jitter, args.f_jitter), 0.1, args.fmax - 0.1))
        fmax_e = float(np.clip(args.fmax + rng.uniform(-args.f_jitter, args.f_jitter), fmin_e + 0.1, 0.5 / args.dt))
        cmin_e = float(max(1.0, args.cmin + rng.uniform(-args.c_jitter, args.c_jitter)))
        cmax_e = float(max(cmin_e + 10.0, args.cmax + rng.uniform(-args.c_jitter, args.c_jitter)))
        dc_e = float(rng.choice(dc_choices))

        f_src, c_src, E_clean_src = park_energy(
            d_clean,
            dx_m=args.dx,
            dt_s=args.dt,
            cmin_ms=cmin_e,
            cmax_ms=cmax_e,
            dc_ms=dc_e,
            fmin_hz=fmin_e,
            fmax_hz=fmax_e,
        )
        _, _, E_noisy_src = park_energy(
            d_noisy,
            dx_m=args.dx,
            dt_s=args.dt,
            cmin_ms=cmin_e,
            cmax_ms=cmax_e,
            dc_ms=dc_e,
            fmin_hz=fmin_e,
            fmax_hz=fmax_e,
        )

        E_clean = resample_fc(E_clean_src, f_src, c_src, f_axis, c_axis, fill_value=0.0)
        E_noisy = resample_fc(E_noisy_src, f_src, c_src, f_axis, c_axis, fill_value=0.0)

        E_noisy, eng_meta = apply_energy_randomization(E_noisy, rng=rng, cfg=eng_aug_cfg)

        sample_meta: Dict[str, Any] = {
            "split": split,
            "idx": idx,
            "model": {"thick_km": thick.tolist(), "vs_km_s": vs.tolist()},
            "wavelet_kind": w_kind,
            "energy_params": {"fmin": fmin_e, "fmax": fmax_e, "cmin": cmin_e, "cmax": cmax_e, "dc": dc_e},
            "record_aug": rec_meta,
            "energy_aug": eng_meta,
        }

        out_path = (train_dir if split == "train" else val_dir) / f"sample_{idx:06d}.npz"
        save_sample_npz(out_path, E_clean=E_clean, E_noisy=E_noisy, Y_curve_fc=Y_curve_fc, mode_mask=mode_mask, meta=sample_meta)

        append_jsonl(manifest_path, {"split": split, "file": out_path.name, **sample_meta})

    for i in range(args.num_train):
        make_one("train", i)
    for i in range(args.num_val):
        make_one("val", i)

    print(f"Dataset written to: {out_root}")


if __name__ == "__main__":
    main()
