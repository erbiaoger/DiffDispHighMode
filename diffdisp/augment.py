from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .synth import ricker


@dataclass(frozen=True)
class RecordAugmentConfig:
    # Missing traces
    missing_prob: float = 0.7
    missing_rate_max: float = 0.3  # 0..1
    missing_block_prob: float = 0.3

    # Additive noise
    white_noise_prob: float = 0.9
    snr_db_min: float = 0.0
    snr_db_max: float = 20.0

    colored_noise_prob: float = 0.5
    colored_p_min: float = 0.5
    colored_p_max: float = 2.0

    # Coherent linear events
    coherent_prob: float = 0.4
    coherent_count_max: int = 3
    coherent_vmin_ms: float = 200.0
    coherent_vmax_ms: float = 1200.0
    coherent_f0_min: float = 5.0
    coherent_f0_max: float = 25.0

    # Frequency dependent attenuation (rough proxy for Q effects)
    atten_prob: float = 0.5
    atten_alpha_min: float = 0.0
    atten_alpha_max: float = 0.015


@dataclass(frozen=True)
class EnergyAugmentConfig:
    gamma_prob: float = 0.3
    gamma_min: float = 0.7
    gamma_max: float = 1.6

    block_prob: float = 0.2
    block_frac_min: float = 0.02
    block_frac_max: float = 0.12

    stripe_prob: float = 0.2
    stripe_rate_max: float = 0.15  # missing frequency rows


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64), dtype=np.float64)))


def apply_record_randomization(
    dshift_xt: np.ndarray,
    *,
    dt_s: float,
    dx_m: float,
    rng: np.random.Generator,
    cfg: RecordAugmentConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply domain randomization in the gather domain."""
    x = dshift_xt.astype(np.float32, copy=True)
    nx, nt = x.shape
    meta: Dict[str, Any] = {}

    # 1) Missing traces (dropouts)
    if rng.random() < cfg.missing_prob:
        rate = float(rng.uniform(0.0, cfg.missing_rate_max))
        nmiss = int(round(rate * nx))
        mask = np.ones(nx, dtype=np.float32)
        if nmiss > 0:
            if rng.random() < cfg.missing_block_prob:
                start = int(rng.integers(0, max(1, nx - nmiss)))
                idx = np.arange(start, start + nmiss)
            else:
                idx = rng.choice(nx, size=nmiss, replace=False)
            mask[idx] = 0.0
        x = (x.T * mask).T
        meta["missing_rate"] = rate
    else:
        meta["missing_rate"] = 0.0

    # 2) Attenuation (frequency dependent)
    if rng.random() < cfg.atten_prob:
        alpha = float(rng.uniform(cfg.atten_alpha_min, cfg.atten_alpha_max))
        nfft = int(2 ** int(np.ceil(np.log2(nt))))
        f = np.fft.rfftfreq(nfft, float(dt_s)).astype(np.float32)
        decay = np.exp(-alpha * f).astype(np.float32)  # stronger decay at higher f
        X = np.fft.rfft(x, n=nfft, axis=1)
        X = X * decay[np.newaxis, :]
        x = np.fft.irfft(X, n=nfft, axis=1)[:, :nt].astype(np.float32)
        meta["atten_alpha"] = alpha
    else:
        meta["atten_alpha"] = 0.0

    # 3) Coherent linear events
    if rng.random() < cfg.coherent_prob:
        n_events = int(rng.integers(1, cfg.coherent_count_max + 1))
        t = np.arange(nt, dtype=np.float32) * float(dt_s)
        xpos = np.arange(nx, dtype=np.float32) * float(dx_m)
        for _ in range(n_events):
            v = float(rng.uniform(cfg.coherent_vmin_ms, cfg.coherent_vmax_ms))
            f0 = float(rng.uniform(cfg.coherent_f0_min, cfg.coherent_f0_max))
            t0 = float(rng.uniform(0.0, 0.25 * nt * dt_s))
            amp = float(rng.uniform(0.1, 0.6))
            for ix in range(nx):
                tau = t - (t0 + xpos[ix] / (v + 1e-6))
                x[ix] += amp * ricker(tau, f0).astype(np.float32)
        meta["coherent_events"] = n_events
    else:
        meta["coherent_events"] = 0

    # 4) Additive white noise by SNR
    if rng.random() < cfg.white_noise_prob:
        snr_db = float(rng.uniform(cfg.snr_db_min, cfg.snr_db_max))
        sig = _rms(x)
        noise_rms = sig / (10 ** (snr_db / 20.0) + 1e-12)
        noise = rng.normal(0.0, noise_rms, size=x.shape).astype(np.float32)
        x = x + noise
        meta["snr_db"] = snr_db
    else:
        meta["snr_db"] = None

    # 5) Colored noise
    if rng.random() < cfg.colored_noise_prob:
        p = float(rng.uniform(cfg.colored_p_min, cfg.colored_p_max))
        nfft = int(2 ** int(np.ceil(np.log2(nt))))
        f = np.fft.rfftfreq(nfft, float(dt_s)).astype(np.float32)
        # Avoid f=0 blowup
        shape = 1.0 / np.maximum(f, f[1] if len(f) > 1 else 1.0) ** (p / 2.0)
        shape = shape / (np.max(shape) + 1e-12)
        W = rng.normal(size=(nx, len(f))).astype(np.float32) + 1j * rng.normal(size=(nx, len(f))).astype(np.float32)
        W = W * shape[np.newaxis, :]
        noise = np.fft.irfft(W, n=nfft, axis=1)[:, :nt].astype(np.float32)
        # Scale colored noise to a fraction of signal RMS
        x = x + 0.2 * _rms(x) * noise / (_rms(noise) + 1e-12)
        meta["colored_p"] = p
    else:
        meta["colored_p"] = None

    return x.astype(np.float32), meta


def apply_energy_randomization(
    E_fc: np.ndarray,
    *,
    rng: np.random.Generator,
    cfg: EnergyAugmentConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply light randomization in energy-matrix domain."""
    x = E_fc.astype(np.float32, copy=True)
    F, C = x.shape
    meta: Dict[str, Any] = {}

    if rng.random() < cfg.gamma_prob:
        g = float(rng.uniform(cfg.gamma_min, cfg.gamma_max))
        x = np.maximum(x, 0.0)
        x = x ** g
        meta["gamma"] = g
    else:
        meta["gamma"] = None

    if rng.random() < cfg.block_prob:
        frac = float(rng.uniform(cfg.block_frac_min, cfg.block_frac_max))
        bh = max(1, int(round(frac * F)))
        bw = max(1, int(round(frac * C)))
        y0 = int(rng.integers(0, max(1, F - bh)))
        x0 = int(rng.integers(0, max(1, C - bw)))
        x[y0 : y0 + bh, x0 : x0 + bw] = 0.0
        meta["block"] = {"frac": frac, "bh": bh, "bw": bw}
    else:
        meta["block"] = None

    if rng.random() < cfg.stripe_prob:
        rate = float(rng.uniform(0.0, cfg.stripe_rate_max))
        nmiss = int(round(rate * F))
        if nmiss > 0:
            idx = rng.choice(F, size=nmiss, replace=False)
            x[idx, :] = 0.0
        meta["stripe_rate"] = rate
    else:
        meta["stripe_rate"] = 0.0

    return x.astype(np.float32), meta

