from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PathConfig:
    eps: float = 1e-8
    lambda_smooth: float = 1.0
    max_jump: int = 10
    const_null: float = 2.0
    null_entry: float = 0.5
    null_exit: float = 0.5
    null_stay: float = 0.0


def _align_c_axis(c_axis_ms: np.ndarray, C: int) -> np.ndarray:
    c = np.asarray(c_axis_ms, dtype=np.float32).reshape(-1)
    if c.size == C:
        return c
    if c.size == 0:
        return np.linspace(0.0, float(C - 1), C, dtype=np.float32)
    cmin = float(np.nanmin(c))
    cmax = float(np.nanmax(c))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
        cmin = 0.0
        cmax = float(C - 1)
    return np.linspace(cmin, cmax, C, dtype=np.float32)


def extract_curve_dp(
    P_fc: np.ndarray,
    c_axis_ms: np.ndarray,
    cfg: PathConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a single curve c(f) from a probability map using DP/Viterbi.

    Args:
      P_fc: [F, C] probabilities in [0, 1]
    Returns:
      curve_f: [F] in m/s, NaN where NULL is chosen
      path_idx: [F] int indices into c_axis, or -1 for NULL
    """
    P = np.asarray(P_fc, dtype=np.float32)
    if P.ndim != 2:
        raise ValueError(f"P_fc must be [F,C], got {P.shape}")
    F, C = P.shape
    c_axis = _align_c_axis(c_axis_ms, C)

    logp = -np.log(np.clip(P, cfg.eps, 1.0))
    cost_obs = logp  # [F,C]
    cost_obs_null = np.full((F,), float(cfg.const_null), dtype=np.float32)

    # state C is NULL
    S = C + 1
    dp = np.full((F, S), np.inf, dtype=np.float32)
    bp = np.full((F, S), -1, dtype=np.int32)

    dp[0, :C] = cost_obs[0]
    dp[0, C] = cost_obs_null[0]

    # Precompute smooth penalty for delta in [-max_jump..max_jump]
    max_jump = int(cfg.max_jump)
    deltas = np.arange(-max_jump, max_jump + 1)
    smooth = cfg.lambda_smooth * (np.minimum(np.abs(deltas), max_jump) ** 2).astype(np.float32)

    for i in range(1, F):
        # Normal states
        for j in range(C):
            best_cost = np.inf
            best_prev = -1

            # prev normal within window
            j0 = max(0, j - max_jump)
            j1 = min(C - 1, j + max_jump)
            for jp in range(j0, j1 + 1):
                d = j - jp
                pen = smooth[d + max_jump]
                v = dp[i - 1, jp] + pen
                if v < best_cost:
                    best_cost = v
                    best_prev = jp

            # prev NULL -> normal
            v_null = dp[i - 1, C] + float(cfg.null_exit)
            if v_null < best_cost:
                best_cost = v_null
                best_prev = C

            dp[i, j] = cost_obs[i, j] + best_cost
            bp[i, j] = best_prev

        # NULL state
        best_cost = dp[i - 1, C] + float(cfg.null_stay)
        best_prev = C
        # normal -> NULL (take best among normal states)
        j_best = int(np.argmin(dp[i - 1, :C]))
        v_from_norm = dp[i - 1, j_best] + float(cfg.null_entry)
        if v_from_norm < best_cost:
            best_cost = v_from_norm
            best_prev = j_best

        dp[i, C] = cost_obs_null[i] + best_cost
        bp[i, C] = best_prev

    # backtrack
    end_state = int(np.argmin(dp[F - 1]))
    path = np.full((F,), -1, dtype=np.int32)
    s = end_state
    for i in reversed(range(F)):
        path[i] = -1 if s == C else s
        s = int(bp[i, s]) if i > 0 else s

    curve = np.full((F,), np.nan, dtype=np.float32)
    valid = path >= 0
    curve[valid] = c_axis[path[valid]]
    return curve, path


def extract_curve_softargmax(
    P_fc: np.ndarray,
    c_axis_ms: np.ndarray,
    power: float = 1.0,
    eps: float = 1e-8,
    conf_thresh: float = 0.0,
) -> np.ndarray:
    """Continuous curve via soft-argmax on probability map.

    Args:
      P_fc: [F, C] probabilities in [0, 1]
      power: sharpening factor; >1 makes peaks sharper
    Returns:
      curve_f: [F] in m/s
    """
    P = np.asarray(P_fc, dtype=np.float32)
    if P.ndim != 2:
        raise ValueError(f"P_fc must be [F,C], got {P.shape}")
    F, C = P.shape
    c_axis = _align_c_axis(c_axis_ms, C)
    W = np.clip(P, 0.0, 1.0) ** float(power)
    denom = np.sum(W, axis=1, keepdims=True) + eps
    curve = (W @ c_axis) / denom[:, 0]
    curve = curve.astype(np.float32)

    if conf_thresh > 0.0:
        max_conf = np.max(P, axis=1)
        curve[max_conf < float(conf_thresh)] = np.nan

    return curve
