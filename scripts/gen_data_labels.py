#!/usr/bin/env python3
"""Generate synthetic dispersion energy images and labels (Park method).

Outputs:
  data/<mode>/data_train/data/*.png
  data/<mode>/data_train/labels/*.png
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

# plotting for image save
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

# project imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from disba import PhaseDispersion  # noqa: E402

# Dispersion / surface-wave utilities
from Dispersion.surfacewaves import ormsby, surfacewavedata  # noqa: E402
from Dispersion.Dispersion.dispersion import get_dispersion  # noqa: E402
import scipy  # noqa: E402


def get_cpr(thick_km, vs_km_s, periods):
    """Compute multi-mode Rayleigh dispersion curves.

    Returns a list of (period, phase_velocity) for modes 0..2.
    """
    true_model = np.vstack([thick_km, vs_km_s * 4, vs_km_s, np.ones_like(vs_km_s)]).T
    pd = PhaseDispersion(*true_model.T)
    cpr = [pd(periods, mode=imode, wave="rayleigh") for imode in range(3)]
    return cpr


def jitter_params(thick_km, vs_km_s, fluctuation=0.1):
    jitter_t = thick_km * (1 + fluctuation * (2 * np.random.rand(len(thick_km)) - 1))
    jitter_v = vs_km_s * (1 + fluctuation * (2 * np.random.rand(len(vs_km_s)) - 1))
    return jitter_t, jitter_v


def synth_dshift(nt, dt, nx, dx, nfft, cpr):
    t = np.arange(nt) * dt

    wav = ormsby(t[: nt // 2 + 1], f=[2, 4, 38, 40], taper=np.hanning)[0][:-1]
    wav = np.roll(np.fft.ifftshift(wav), 20)

    dshifts = []
    for imode in range(3):
        dshift_, _, _ = surfacewavedata(
            nt,
            dt,
            nx,
            dx,
            nfft,
            np.flipud(1 / cpr[imode][0]),
            np.flipud(cpr[imode][1]),
            wav,
        )
        dshifts.append(1.0 / (imode + 1) ** 0.8 * dshift_[np.newaxis])
    return np.concatenate(dshifts).sum(0)


def park_dispersion(dshift, dx, dt, cmin, cmax, dc, fmin, fmax):
    f1, c1, img, _, _ = get_dispersion(dshift.T, dx, dt, cmin, cmax, dc, fmin, fmax)
    return f1, c1, img


def save_energy_image(path, f, c, img, fmin, fmax):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(
        img,
        aspect="auto",
        cmap="gray",
        extent=(f.min(), f.max(), c.min(), c.max()),
        origin="lower",
        interpolation="bilinear",
    )
    ax.margins(0)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(c.min(), c.max())
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_label_image(path, f, c, cpr, fmin, fmax):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(
        np.zeros((len(c), len(f))),
        aspect="auto",
        cmap="gray",
        extent=(f.min(), f.max(), c.min(), c.max()),
        origin="lower",
    )
    for imode in range(3):
        ax.plot(np.flipud(1 / cpr[imode][0]), 1.0e3 * np.flipud(cpr[imode][1]), "white", lw=4)
    ax.margins(0)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(c.min(), c.max())
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate dispersion energy images and labels.")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data"), help="output root")
    parser.add_argument("--mode", type=str, default="demultiple", help="dataset mode name")
    parser.add_argument("--num", type=int, default=200, help="number of samples")
    parser.add_argument("--seed", type=int, default=1234)

    # physics / grid
    parser.add_argument("--nt", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--nfft", type=int, default=1024)

    parser.add_argument("--cmin", type=float, default=100.0)
    parser.add_argument("--cmax", type=float, default=900.0)
    parser.add_argument("--dc", type=float, default=2.0)
    parser.add_argument("--fmin", type=float, default=3.0)
    parser.add_argument("--fmax", type=float, default=12.0)

    parser.add_argument("--fluct", type=float, default=0.2, help="parameter jitter")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    periods = np.flipud(1 / np.linspace(args.fmin, args.fmax, 101))

    out_root = Path(args.out) / args.mode / "data_train"
    data_dir = out_root / "data"
    label_dir = out_root / "labels"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # default parameter ranges (m, m/s)
    h_ranges = [
                np.linspace(5, 15, 3)*1e-3, 
                np.linspace(10, 30, 3)*1e-3, 
                np.linspace(20, 50, 3)*1e-3, 
                np.array([1.0])
                ]
    v_ranges = [
                np.linspace(100, 300, 2)*1e-3, 
                np.linspace(300, 600, 3)*1e-3, 
                np.linspace(300, 700, 3)*1e-3, 
                np.linspace(700, 900, 3)*1e-3
                ]

    # build grid with product instead of nested loops
    import itertools

    h_grid = list(itertools.product(*h_ranges))
    v_grid = list(itertools.product(*v_ranges))
    combos = [(np.array(h), np.array(v)) for h in h_grid for v in v_grid]

    random.shuffle(combos)
    combos = combos[: args.num]

    for idx, (thick, vs) in enumerate(combos):
        thick, vs = jitter_params(thick, vs, fluctuation=args.fluct)
        cpr = get_cpr(thick, vs, periods)
        dshift = synth_dshift(args.nt, args.dt, args.nx, args.dx, args.nfft, cpr)

        f, c, img = park_dispersion(dshift, args.dx, args.dt, args.cmin, args.cmax, args.dc, args.fmin, args.fmax)

        name = f"{idx:06d}.png"
        save_energy_image(data_dir / name, f, c, img, args.fmin, args.fmax)
        save_label_image(label_dir / name, f, c, cpr, args.fmin, args.fmax)

    print(f"Saved {len(combos)} samples to {out_root}")


if __name__ == "__main__":
    main()
