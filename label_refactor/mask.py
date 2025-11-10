"""Utilities for converting structured dispersion curves into supervision masks."""

from __future__ import annotations

import warnings
from typing import Iterable, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from skimage.draw import line_aa
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    line_aa = None


def _interpolate_to_grid(x: np.ndarray, y: np.ndarray, grid_freq: np.ndarray, grid_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map curve coordinates to pixel indices."""

    freq_idx = np.interp(x, (grid_freq.min(), grid_freq.max()), (0, grid_freq.size - 1))
    vel_idx = np.interp(y, (grid_vel.min(), grid_vel.max()), (0, grid_vel.size - 1))
    return freq_idx, vel_idx


def rasterize_modes(
    curves: np.ndarray,
    grid_freq: np.ndarray,
    grid_vel: np.ndarray,
    blur_sigma: float = 1.0,
    antialiased: bool = True,
) -> np.ndarray:
    """Render each mode into its own channel, applying optional Gaussian smoothing."""

    mode_count = curves.shape[0]
    mask = np.zeros((mode_count, grid_vel.size, grid_freq.size), dtype=np.float32)

    use_line_aa = antialiased
    if antialiased and line_aa is None:
        warnings.warn("scikit-image not available; falling back to nearest-neighbour rasterization", RuntimeWarning)
        use_line_aa = False

    for mode in range(mode_count):
        freq = curves[mode, 0]
        vel = curves[mode, 1]
        freq_idx, vel_idx = _interpolate_to_grid(freq, vel, grid_freq, grid_vel)
        if use_line_aa:
            for i in range(len(freq_idx) - 1):
                rr, cc, val = line_aa(
                    int(vel_idx[i]),
                    int(freq_idx[i]),
                    int(vel_idx[i + 1]),
                    int(freq_idx[i + 1]),
                )
                mask[mode, rr, cc] = np.maximum(mask[mode, rr, cc], val)
        else:
            rr = np.clip(np.round(vel_idx).astype(int), 0, grid_vel.size - 1)
            cc = np.clip(np.round(freq_idx).astype(int), 0, grid_freq.size - 1)
            mask[mode, rr, cc] = 1.0
        if blur_sigma > 0:
            mask[mode] = gaussian_filter(mask[mode], sigma=blur_sigma)
    return mask


def save_mask(mask: np.ndarray, out_path: str) -> None:
    np.save(out_path, mask)
