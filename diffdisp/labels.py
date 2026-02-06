from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LabelConfig:
    K_max: int = 5
    sigma_px: float = 3.0
    eps: float = 1e-12


def cpr_to_curve_on_f_axis(
    period_s: np.ndarray,
    v_km_s: np.ndarray,
    f_axis_hz: np.ndarray,
    *,
    v_unit: str = "km/s",
) -> np.ndarray:
    """Convert (period, v) curve to c(f) sampled on f_axis.

    Returns c in m/s, with NaN outside the curve support.
    """
    period_s = np.asarray(period_s, dtype=np.float32)
    v = np.asarray(v_km_s, dtype=np.float32)

    # disba convention in this repo: v is km/s, convert to m/s
    v_ms = v * 1e3

    freq = 1.0 / (period_s + 1e-12)
    # Ensure increasing for np.interp
    freq_inc = np.flipud(freq)
    v_inc = np.flipud(v_ms)

    f_axis_hz = np.asarray(f_axis_hz, dtype=np.float32)
    c = np.interp(f_axis_hz, freq_inc, v_inc, left=np.nan, right=np.nan).astype(np.float32)
    return c


def curves_to_prob_maps(
    curves_kf_ms: np.ndarray,
    c_axis_ms: np.ndarray,
    *,
    sigma_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Make gaussian probability-map labels from curves.

    Args:
      curves_kf_ms: [K, F] in m/s (NaN means invalid at that frequency)
      c_axis_ms: [C]
    Returns:
      Y_kfc: [K, F, C] float32 in [0,1]
      valid_kf: [K, F] uint8 (1 where curve is valid)
    """
    curves = np.asarray(curves_kf_ms, dtype=np.float32)
    c_axis = np.asarray(c_axis_ms, dtype=np.float32)
    K, F = curves.shape
    C = len(c_axis)

    # Convert sigma from pixels to m/s using median spacing on c axis.
    dc = float(np.median(np.diff(c_axis))) if C > 1 else 1.0
    sigma_ms = max(1e-6, float(sigma_px) * dc)

    Y = np.zeros((K, F, C), dtype=np.float32)
    valid = np.zeros((K, F), dtype=np.uint8)

    for k in range(K):
        for i in range(F):
            c0 = curves[k, i]
            if not np.isfinite(c0):
                continue
            valid[k, i] = 1
            z = (c_axis - c0) / sigma_ms
            Y[k, i, :] = np.exp(-0.5 * z * z, dtype=np.float32)

    return Y, valid

