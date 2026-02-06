from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .axes import GridSpec


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def save_sample_npz(
    path: Path,
    *,
    E_clean: np.ndarray,
    E_noisy: np.ndarray,
    Y_curve_fc: np.ndarray,
    mode_mask: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta is None:
        meta = {}
    np.savez_compressed(
        path,
        E_clean=E_clean.astype(np.float32),
        E_noisy=E_noisy.astype(np.float32),
        Y_curve_fc=Y_curve_fc.astype(np.float32),
        mode_mask=mode_mask.astype(np.uint8),
        meta=json.dumps(meta),
    )


def load_sample_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        meta = z.get("meta")
        meta_obj = json.loads(str(meta)) if meta is not None else {}
        return {
            "E_clean": z["E_clean"].astype(np.float32),
            "E_noisy": z["E_noisy"].astype(np.float32),
            "Y_curve_fc": z["Y_curve_fc"].astype(np.float32),
            "mode_mask": z["mode_mask"].astype(np.uint8),
            "meta": meta_obj,
        }

