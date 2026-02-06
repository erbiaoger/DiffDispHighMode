from __future__ import annotations

import numpy as np


def park_energy(
    dshift_xt: np.ndarray,
    *,
    dx_m: float,
    dt_s: float,
    cmin_ms: float,
    cmax_ms: float,
    dc_ms: float,
    fmin_hz: float,
    fmax_hz: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Park et al. (1998) dispersion energy image.

    Args:
      dshift_xt: gather [nx, nt]
      returns:
        f_vec: [F_src] in Hz (increasing)
        c_vec: [C_src] in m/s (increasing)
        E_fc:  [F_src, C_src]
    """
    if dshift_xt.ndim != 2:
        raise ValueError(f"expected dshift_xt [nx, nt], got {dshift_xt.shape}")

    nx, nt = dshift_xt.shape
    dx_m = float(dx_m)
    dt_s = float(dt_s)

    f_all = np.fft.fftfreq(nt, dt_s).astype(np.float32)
    f_pos = f_all[: nt // 2]
    df = float(f_pos[1] - f_pos[0])

    fmin_idx = int(max(0, np.floor(fmin_hz / df)))
    fmax_idx = int(min(len(f_pos), np.ceil(fmax_hz / df)))
    f_vec = f_pos[fmin_idx:fmax_idx].astype(np.float32)
    if len(f_vec) == 0:
        raise ValueError("empty frequency range; check fmin/fmax and dt/nt")

    c_vec = np.arange(float(cmin_ms), float(cmax_ms), float(dc_ms), dtype=np.float32)
    if len(c_vec) == 0:
        raise ValueError("empty velocity range; check cmin/cmax/dc")

    # FFT along time for each trace
    U = np.fft.fft(dshift_xt.astype(np.float32), axis=1)[:, : nt // 2].astype(np.complex64)  # [nx, nf]

    # Precompute x axis
    x = (np.arange(nx, dtype=np.float32) * dx_m).astype(np.float32)  # [nx]

    E_cf = np.zeros((len(c_vec), len(f_vec)), dtype=np.float32)
    for fi, f0 in enumerate(f_vec):
        spec = U[:, fmin_idx + fi]
        spec = spec / (np.abs(spec) + eps)
        for ci, c0 in enumerate(c_vec):
            k = 2.0 * np.pi * f0 / (c0 + eps)
            phase = np.exp(1.0j * k * x).astype(np.complex64)
            E_cf[ci, fi] = float(np.abs(np.dot(dx_m * phase, spec)))

    # Return as [F, C]
    E_fc = E_cf.T.copy()
    return f_vec, c_vec, E_fc

