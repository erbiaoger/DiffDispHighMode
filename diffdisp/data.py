from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .axes import GridSpec
from .io import load_sample_npz
from .labels import curves_to_prob_maps
from .normalize import NormConfig, normalize_energy

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


def list_npz_files(dataset_root: Path, split: str) -> List[Path]:
    p = Path(dataset_root) / split
    if not p.exists():
        raise FileNotFoundError(f"split dir not found: {p}")
    return sorted([x for x in p.glob("*.npz") if x.is_file()])


def load_meta(dataset_root: Path) -> Dict[str, Any]:
    meta_path = Path(dataset_root) / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class DatasetConfig:
    dataset_root: Path
    split: str = "train"
    use_input: str = "noisy"  # noisy|clean
    norm: NormConfig = NormConfig()
    sigma_px: float = 3.0
    K_max: int = 5
    return_prob_maps: bool = True
    return_clean_target: bool = True


class NPZDispersionDataset(Dataset):
    """Torch dataset for NPZ samples.

    Returns dict with:
      x: [1,F,C] float32
      y_map: [K,F,C] float32 (optional)
      curve: [K,F] float32
      mode_mask: [K] uint8
      valid_kf: [K,F] uint8 (optional)
    """

    def __init__(self, cfg: DatasetConfig):
        if torch is None:
            raise ImportError("torch is required to use NPZDispersionDataset")
        self.cfg = cfg
        self.files = list_npz_files(cfg.dataset_root, cfg.split)
        self.meta = load_meta(cfg.dataset_root)
        self.grid = GridSpec.from_dict(self.meta["grid"])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = load_sample_npz(self.files[idx])
        E_noisy = normalize_energy(s["E_noisy"], self.cfg.norm)
        E_clean = normalize_energy(s["E_clean"], self.cfg.norm)
        x_in = E_noisy if self.cfg.use_input == "noisy" else E_clean
        x = torch.from_numpy(x_in[None, :, :].astype(np.float32))

        curve = s["Y_curve_fc"].astype(np.float32)
        mode_mask = s["mode_mask"].astype(np.uint8)

        out: Dict[str, Any] = {
            "x": x,
            "curve": torch.from_numpy(curve),
            "mode_mask": torch.from_numpy(mode_mask.astype(np.uint8)),
            "path": str(self.files[idx]),
        }

        if self.cfg.return_clean_target:
            out["x_clean"] = torch.from_numpy(E_clean[None, :, :].astype(np.float32))
            out["x_noisy"] = torch.from_numpy(E_noisy[None, :, :].astype(np.float32))

        if self.cfg.return_prob_maps:
            Y, valid = curves_to_prob_maps(curve[: self.cfg.K_max], self.grid.c_axis(), sigma_px=self.cfg.sigma_px)
            out["y_map"] = torch.from_numpy(Y.astype(np.float32))
            out["valid_kf"] = torch.from_numpy(valid.astype(np.uint8))
        return out
