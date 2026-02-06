from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def _tcrop(t: np.ndarray) -> np.ndarray:
    if len(t) % 2 == 0:
        warnings.warn("one sample removed from time axis...", stacklevel=2)
        return t[:-1]
    return t


def ormsby(t_pos: np.ndarray, f: Sequence[float] = (5.0, 10.0, 45.0, 50.0), taper=None):
    """Ormsby wavelet (same convention as Dispersion/surfacewaves.py, but without extra deps)."""

    def numerator(freq: float, t: np.ndarray) -> np.ndarray:
        return (np.sinc(freq * t) ** 2) * ((np.pi * freq) ** 2)

    t_pos = _tcrop(np.asarray(t_pos, dtype=np.float32))
    t = np.concatenate((np.flipud(-t_pos[1:]), t_pos), axis=0)

    f1, f2, f3, f4 = map(float, f)
    pf43 = (np.pi * f4) - (np.pi * f3)
    pf21 = (np.pi * f2) - (np.pi * f1)
    w = (
        (numerator(f4, t) / pf43)
        - (numerator(f3, t) / pf43)
        - (numerator(f2, t) / pf21)
        + (numerator(f1, t) / pf21)
    )
    w = w / (np.max(np.abs(w)) + 1e-12)

    if taper is not None:
        w = w * taper(len(t))

    wcenter = int(np.argmax(np.abs(w)))
    return w.astype(np.float32), t.astype(np.float32), wcenter


def ricker(t: np.ndarray, f0: float) -> np.ndarray:
    """Ricker wavelet centered at t=0."""
    t = np.asarray(t, dtype=np.float32)
    a = (np.pi * f0) ** 2
    return (1.0 - 2.0 * a * t**2) * np.exp(-a * t**2)


def surfacewavedata(
    nt: int,
    dt: float,
    nx: int,
    dx: float,
    nfft: int,
    fdisp: np.ndarray,
    vdisp_km_s: np.ndarray,
    wav: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize surface-wave-only shot gather from a dispersion relation.

    vdisp is in km/s (to match the existing project convention).
    """
    t = np.arange(nt, dtype=np.float32) * float(dt)
    x = np.arange(nx, dtype=np.float32) * float(dx)

    f = np.fft.rfftfreq(nfft, float(dt)).astype(np.float32)
    vf = np.interp(f, fdisp.astype(np.float32), vdisp_km_s.astype(np.float32)).astype(np.float32)
    # Protect against division blowups in time shifts (vf is in km/s).
    vf = np.maximum(vf, 0.05).astype(np.float32)

    data = np.outer(wav.astype(np.float32), np.ones(nx, dtype=np.float32)).T  # [nx, wavlen]
    D = np.fft.rfft(data, n=nfft, axis=1)  # [nx, nf]

    # shifts = x / v(f), but v is km/s, so convert to s/m by 1e-3
    shifts = np.outer(x, 1e-3 / (vf + 1e-12)).astype(np.float32)
    phase = np.exp(-1j * 2 * np.pi * f[np.newaxis, :] * shifts).astype(np.complex64)

    dshift = np.fft.irfft(D * phase, n=nfft, axis=1)[:, :nt].astype(np.float32)
    return dshift, f, vf


def multi_mode_gather(
    *,
    nt: int,
    dt: float,
    nx: int,
    dx: float,
    nfft: int,
    modes_fdisp: Sequence[np.ndarray],
    modes_vdisp_km_s: Sequence[np.ndarray],
    wav: np.ndarray,
    mode_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Sum multiple mode gathers with optional weights."""
    if mode_weights is None:
        mode_weights = [1.0] * len(modes_fdisp)
    if len(mode_weights) != len(modes_fdisp):
        raise ValueError("mode_weights length must match number of modes")

    acc = None
    for k, (fdisp, vdisp) in enumerate(zip(modes_fdisp, modes_vdisp_km_s)):
        dshift, _, _ = surfacewavedata(nt, dt, nx, dx, nfft, fdisp, vdisp, wav)
        w = float(mode_weights[k])
        acc = dshift * w if acc is None else acc + dshift * w
    return acc.astype(np.float32)
