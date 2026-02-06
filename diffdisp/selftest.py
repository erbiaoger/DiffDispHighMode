from __future__ import annotations

import numpy as np

from .axes import GridSpec
from .metrics import MetricsConfig, compute_metrics_for_matched
from .path import PathConfig, extract_curve_dp


def _make_synthetic_prob(F: int, C: int, path_idx: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    c = np.arange(C, dtype=np.float32)
    P = np.zeros((F, C), dtype=np.float32)
    for i in range(F):
        j = int(path_idx[i])
        z = (c - j) / float(sigma)
        P[i] = np.exp(-0.5 * z * z)
    P = P / (np.max(P) + 1e-8)
    return P


def run() -> None:
    grid = GridSpec(fmin_hz=3.0, fmax_hz=12.0, F=64, cmin_ms=100.0, cmax_ms=900.0, C=64)
    c_axis = grid.c_axis()

    gt_idx = np.linspace(10, 50, grid.F).astype(np.int32)
    P = _make_synthetic_prob(grid.F, grid.C, gt_idx)

    curve, idx = extract_curve_dp(P, c_axis, PathConfig(max_jump=4, lambda_smooth=0.5))
    gt_curve = c_axis[gt_idx]

    m = compute_metrics_for_matched(curve, gt_curve, MetricsConfig(hit_tol_ms=20.0))
    assert np.isfinite(m["mae_ms"]) and m["mae_ms"] < 20.0, m
    print("diffdisp.selftest OK", m)


if __name__ == "__main__":
    run()

