from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


NormKind = Literal[
    "none",
    "log1p_minmax",
    "log1p_zscore",
    "log1p_per_f_minmax",
]


@dataclass(frozen=True)
class NormConfig:
    kind: NormKind = "log1p_minmax"
    eps: float = 1e-8


def _minmax(x: np.ndarray, eps: float) -> np.ndarray:
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    return (x - x_min) / (x_max - x_min + eps)


def _zscore(x: np.ndarray, eps: float) -> np.ndarray:
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)
    mu = float(np.nanmean(x))
    sig = float(np.nanstd(x))
    return (x - mu) / (sig + eps)


def normalize_energy(E_fc: np.ndarray, cfg: NormConfig) -> np.ndarray:
    """Normalize an energy matrix with shape [F, C]."""
    if cfg.kind == "none":
        return E_fc.astype(np.float32, copy=False)

    x = E_fc.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.log1p(np.maximum(x, 0.0))

    if cfg.kind == "log1p_minmax":
        return _minmax(x, cfg.eps).astype(np.float32, copy=False)
    if cfg.kind == "log1p_zscore":
        return _zscore(x, cfg.eps).astype(np.float32, copy=False)
    if cfg.kind == "log1p_per_f_minmax":
        out = np.empty_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            out[i] = _minmax(x[i], cfg.eps)
        return out

    raise ValueError(f"unknown norm kind: {cfg.kind}")


def ensure_fc(x: np.ndarray) -> np.ndarray:
    """Ensure matrix is [F, C]."""
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {x.shape}")
    return x
