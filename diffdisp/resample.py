from __future__ import annotations

from typing import Optional

import numpy as np


def resample_fc(
    E_src_fc: np.ndarray,
    f_src: np.ndarray,
    c_src: np.ndarray,
    f_tgt: np.ndarray,
    c_tgt: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Resample E(f,c) from (f_src,c_src) to (f_tgt,c_tgt) using separable 1D linear interpolation.

    Shapes:
      E_src_fc: [F_src, C_src]
      output:   [F_tgt, C_tgt]
    """
    if E_src_fc.ndim != 2:
        raise ValueError(f"E_src_fc must be 2D, got {E_src_fc.shape}")

    f_src = np.asarray(f_src, dtype=np.float32)
    c_src = np.asarray(c_src, dtype=np.float32)
    f_tgt = np.asarray(f_tgt, dtype=np.float32)
    c_tgt = np.asarray(c_tgt, dtype=np.float32)

    if not (np.all(np.diff(f_src) > 0) and np.all(np.diff(c_src) > 0)):
        raise ValueError("f_src and c_src must be strictly increasing")

    E_src_fc = E_src_fc.astype(np.float32, copy=False)

    # First interpolate along f for each c-bin.
    tmp = np.empty((len(f_tgt), len(c_src)), dtype=np.float32)
    for j in range(len(c_src)):
        tmp[:, j] = np.interp(f_tgt, f_src, E_src_fc[:, j], left=fill_value, right=fill_value)

    # Then interpolate along c for each f-bin.
    out = np.empty((len(f_tgt), len(c_tgt)), dtype=np.float32)
    for i in range(len(f_tgt)):
        out[i, :] = np.interp(c_tgt, c_src, tmp[i, :], left=fill_value, right=fill_value)

    return out

