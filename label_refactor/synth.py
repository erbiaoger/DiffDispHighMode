"""Synthesis utilities that mirror the legacy dispersion generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:
    from disba import PhaseDispersion
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hook
        raise ImportError("disba is required for dispersion synthesis. Install via `pip install disba`." ) from exc

from Dispersion.surfacewaves import ormsby, surfacewavedata
from Dispersion.Dispersion.dispersion import get_dispersion as park_transform


@dataclass
class SimulationParams:
    nt: int
    dt: float
    nx: int
    dx: float
    nfft: int
    cmin: float
    cmax: float
    dc: float
    fmin: float
    fmax: float


def _randomize_layers(thickness: np.ndarray, vs: np.ndarray, fluctuation: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    jitter = 1 + fluctuation * (2 * rng.random(size=thickness.shape) - 1)
    jvs = 1 + fluctuation * (2 * rng.random(size=vs.shape) - 1)
    return thickness * jitter, vs * jvs


def _resample_mode(freq_raw: np.ndarray, vel_raw: np.ndarray, target_freq: np.ndarray) -> np.ndarray:
    order = np.argsort(freq_raw)
    freq_sorted = freq_raw[order]
    vel_sorted = vel_raw[order]
    mask = np.isfinite(freq_sorted) & np.isfinite(vel_sorted)
    freq_sorted = freq_sorted[mask]
    vel_sorted = vel_sorted[mask]
    if freq_sorted.size < 2:
        return np.zeros_like(target_freq)
    vel_interp = np.interp(
        target_freq,
        freq_sorted,
        vel_sorted,
        left=float(vel_sorted[0]),
        right=float(vel_sorted[-1]),
    )
    vel_interp = np.clip(vel_interp, a_min=1e-6, a_max=None)
    return vel_interp


def _compute_curves(
    thickness: np.ndarray,
    vs: np.ndarray,
    period: np.ndarray,
    mode_count: int,
    target_freq: np.ndarray,
) -> np.ndarray:
    true_model = np.vstack([thickness, vs * 2, vs, np.ones_like(vs)]).T
    pd = PhaseDispersion(*true_model.T)
    curves = []
    for mode in range(mode_count):
        result        = pd(period, mode=mode, wave="rayleigh")
        per           = result[0]
        vel           = result[1]
        freq          = np.flipud(1 / per)
        vel_ms        = 1e3 * np.flipud(vel)
        vel_resampled = _resample_mode(freq, vel_ms, target_freq)
        curves.append(np.stack([target_freq, vel_resampled], axis=0))
    return np.stack(curves, axis=0)


def _synthetize_data(curves: np.ndarray, params: SimulationParams) -> np.ndarray:
    nt, dt, nx, dx, nfft = params.nt, params.dt, params.nx, params.dx, params.nfft
    t = np.arange(nt) * dt
    wav = ormsby(t[: nt // 2 + 1], f=(2, 4, 38, 40), taper=np.hanning)[0][:-1]
    wav = np.roll(np.fft.ifftshift(wav), 20)

    dshift_sum = np.zeros((nx, nt))
    for mode_idx in range(curves.shape[0]):
        freq = curves[mode_idx, 0]
        vel = curves[mode_idx, 1] * 1e-3  # km/s expected by surfacewavedata
        vel = np.clip(vel, a_min=1e-6, a_max=None)
        dshift, *_ = surfacewavedata(nt, dt, nx, dx, nfft, freq, vel, wav)
        weight = 1.0 / ((mode_idx + 1) ** 0.8)
        dshift_sum += weight * dshift
    return dshift_sum


def _resample_image(
    img: np.ndarray,
    f_src: np.ndarray,
    c_src: np.ndarray,
    f_dst: np.ndarray,
    c_dst: np.ndarray,
) -> np.ndarray:
    f_src = np.asarray(f_src)
    c_src = np.asarray(c_src)
    f_dst = np.asarray(f_dst)
    c_dst = np.asarray(c_dst)

    def _interp_rows(data: np.ndarray) -> np.ndarray:
        if len(f_src) == len(f_dst) and np.allclose(f_src, f_dst):
            return data.copy()
        out = np.empty((data.shape[0], len(f_dst)), dtype=data.dtype)
        for idx, row in enumerate(data):
            out[idx] = np.interp(f_dst, f_src, row, left=0.0, right=0.0)
        return out

    def _interp_cols(data: np.ndarray) -> np.ndarray:
        if len(c_src) == len(c_dst) and np.allclose(c_src, c_dst):
            return data.copy()
        out = np.empty((len(c_dst), data.shape[1]), dtype=data.dtype)
        for col in range(data.shape[1]):
            out[:, col] = np.interp(c_dst, c_src, data[:, col], left=0.0, right=0.0)
        return out

    freq_resampled = _interp_rows(img)
    resampled = _interp_cols(freq_resampled)
    return resampled


def simulate_sample(
    thickness             : np.ndarray,
    vs                    : np.ndarray,
    freq_axis             : np.ndarray,
    vel_axis              : np.ndarray,
    mode_count            : int,
    fluctuation_percentage: float,
    rng                   : np.random.Generator,
    params                : SimulationParams,
) -> Tuple[np.ndarray, np.ndarray]:
    period = np.flipud(1 / freq_axis)
    thick_j, vs_j = _randomize_layers(thickness, vs, fluctuation_percentage, rng)
    curves = _compute_curves(thick_j, vs_j, period, mode_count, freq_axis)
    dshift = _synthetize_data(curves, params)
    f_src, c_src, img, *_ = park_transform(dshift.T, params.dx, params.dt, params.cmin, params.cmax, params.dc, params.fmin, params.fmax)
    spectrum = _resample_image(img, f_src, c_src, freq_axis, vel_axis)
    return curves, spectrum
