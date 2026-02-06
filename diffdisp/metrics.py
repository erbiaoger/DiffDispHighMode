from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricsConfig:
    hit_tol_ms: float = 20.0
    eps: float = 1e-8


def _contiguous_nan_segments(x: np.ndarray) -> int:
    isn = ~np.isfinite(x)
    if isn.size == 0:
        return 0
    # Count transitions 0->1 segments
    return int(np.sum((~isn[:-1]) & (isn[1:])) + (1 if isn[0] else 0))


def curve_mae(pred_f: np.ndarray, gt_f: np.ndarray) -> float:
    pred = np.asarray(pred_f, dtype=np.float32)
    gt = np.asarray(gt_f, dtype=np.float32)
    m = np.isfinite(gt) & np.isfinite(pred)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m])))


def curve_hit_at_tol(pred_f: np.ndarray, gt_f: np.ndarray, tol_ms: float) -> float:
    pred = np.asarray(pred_f, dtype=np.float32)
    gt = np.asarray(gt_f, dtype=np.float32)
    m = np.isfinite(gt) & np.isfinite(pred)
    if not np.any(m):
        return float("nan")
    return float(np.mean((np.abs(pred[m] - gt[m]) <= tol_ms).astype(np.float32)))


def curve_coverage(pred_f: np.ndarray) -> float:
    pred = np.asarray(pred_f, dtype=np.float32)
    return float(np.mean(np.isfinite(pred).astype(np.float32)))


def curve_break_rate(pred_f: np.ndarray) -> float:
    pred = np.asarray(pred_f, dtype=np.float32)
    return float(_contiguous_nan_segments(pred) / max(1, len(pred)))


def curve_smoothness(pred_f: np.ndarray) -> float:
    pred = np.asarray(pred_f, dtype=np.float32)
    m = np.isfinite(pred)
    if np.sum(m) < 3:
        return float("nan")
    idx = np.where(m)[0]
    # only consider consecutive valid samples
    diffs = []
    for i in range(len(idx) - 1):
        if idx[i + 1] == idx[i] + 1:
            diffs.append(abs(float(pred[idx[i + 1]] - pred[idx[i]])))
    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


def match_modes_mae(
    pred_kf: np.ndarray,
    gt_kf: np.ndarray,
    pred_mask_k: Optional[np.ndarray] = None,
    gt_mask_k: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match predicted modes to GT modes by minimizing MAE (Hungarian if available)."""
    pred = np.asarray(pred_kf, dtype=np.float32)
    gt = np.asarray(gt_kf, dtype=np.float32)
    Kp, F = pred.shape
    Kg, _ = gt.shape

    if pred_mask_k is None:
        pred_mask_k = np.ones((Kp,), dtype=np.uint8)
    if gt_mask_k is None:
        gt_mask_k = np.ones((Kg,), dtype=np.uint8)

    cost = np.full((Kp, Kg), 1e9, dtype=np.float32)
    for i in range(Kp):
        if pred_mask_k[i] == 0:
            continue
        for j in range(Kg):
            if gt_mask_k[j] == 0:
                continue
            v = curve_mae(pred[i], gt[j])
            cost[i, j] = 1e8 if not np.isfinite(v) else float(v)

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        row, col = linear_sum_assignment(cost)
        return row.astype(np.int32), col.astype(np.int32)
    except Exception:
        # Greedy fallback
        row = []
        col = []
        used_g = set()
        for i in range(Kp):
            j = int(np.argmin(cost[i]))
            if j in used_g:
                continue
            used_g.add(j)
            row.append(i)
            col.append(j)
        return np.asarray(row, dtype=np.int32), np.asarray(col, dtype=np.int32)


def compute_metrics_for_matched(
    pred_f: np.ndarray,
    gt_f: np.ndarray,
    cfg: MetricsConfig,
) -> Dict[str, Any]:
    return {
        "mae_ms": curve_mae(pred_f, gt_f),
        "hit_at_tol": curve_hit_at_tol(pred_f, gt_f, cfg.hit_tol_ms),
        "coverage": curve_coverage(pred_f),
        "break_rate": curve_break_rate(pred_f),
        "smoothness": curve_smoothness(pred_f),
    }

