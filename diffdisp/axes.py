from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Fixed (f, c) grid definition for numeric matrices.

    - f: frequency axis in Hz, length F
    - c: phase-velocity axis in m/s, length C
    """

    fmin_hz: float
    fmax_hz: float
    F: int
    cmin_ms: float
    cmax_ms: float
    C: int

    def f_axis(self) -> np.ndarray:
        return np.linspace(self.fmin_hz, self.fmax_hz, self.F, dtype=np.float32)

    def c_axis(self) -> np.ndarray:
        return np.linspace(self.cmin_ms, self.cmax_ms, self.C, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fmin_hz": float(self.fmin_hz),
            "fmax_hz": float(self.fmax_hz),
            "F": int(self.F),
            "cmin_ms": float(self.cmin_ms),
            "cmax_ms": float(self.cmax_ms),
            "C": int(self.C),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GridSpec":
        return GridSpec(
            fmin_hz=float(d["fmin_hz"]),
            fmax_hz=float(d["fmax_hz"]),
            F=int(d["F"]),
            cmin_ms=float(d["cmin_ms"]),
            cmax_ms=float(d["cmax_ms"]),
            C=int(d["C"]),
        )

