"""PyTorch dataset for dispersion spectra + masks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import apply_spectrum_augmentation
from label_refactor import mask as mask_utils


@dataclass
class DatasetSplit:
    train: Sequence[dict]
    val: Sequence[dict]


def _read_manifest(manifest_path: Path) -> List[dict]:
    entries = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"Manifest {manifest_path} is empty; run label_refactor.cli first")
    return entries


def split_manifest(entries: Sequence[dict], val_split: float, seed: int) -> DatasetSplit:
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    entries = list(entries)
    rng = np.random.default_rng(seed)
    rng.shuffle(entries)
    val_size = max(1, int(len(entries) * val_split))
    train_size = max(1, len(entries) - val_size)
    if train_size + val_size > len(entries):
        val_size = len(entries) - train_size
    train_entries = entries[:train_size]
    val_entries = entries[train_size:train_size + val_size]
    if not val_entries:
        raise ValueError("Validation split ended up empty; adjust val_split or dataset size")
    return DatasetSplit(train=train_entries, val=val_entries)


class DispersionDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[dict],
        mask_dir: Path,
        auto_generate_masks: bool = True,
        blur_sigma: float = 1.5,
        antialiased: bool = False,
        augment_cfg: Dict | None = None,
    ):
        self.entries = list(entries)
        self.mask_dir = Path(mask_dir)
        self.auto_generate_masks = auto_generate_masks
        self.blur_sigma = blur_sigma
        self.antialiased = antialiased
        self.augment_cfg = augment_cfg or {}
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_record(self, entry: dict):
        npz_path = Path(entry["npz"]).expanduser()
        with np.load(npz_path) as data:
            spectrum = np.nan_to_num(data["spectrum"], nan=0.0).astype(np.float32)
            curves = data["curves"].astype(np.float32)
            freq = data["freq"].astype(np.float32)
            phase_velocity = data["phase_velocity"].astype(np.float32)
        return spectrum, curves, freq, phase_velocity

    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        spectrum = np.clip(spectrum, a_min=0.0, a_max=None)
        max_val = spectrum.max()
        if max_val > 0:
            spectrum /= max_val
        return spectrum[None, ...]

    def _materialize_mask(self, entry: dict, curves: np.ndarray, freq: np.ndarray, phase_velocity: np.ndarray) -> np.ndarray:
        mask_path = self.mask_dir / f"{entry['id']}.npy"
        if not mask_path.exists():
            if not self.auto_generate_masks:
                raise FileNotFoundError(f"Missing mask at {mask_path}")
            mask = mask_utils.rasterize_modes(
                curves,
                freq,
                phase_velocity,
                blur_sigma=self.blur_sigma,
                antialiased=self.antialiased,
            )
            mask_utils.save_mask(mask, mask_path.as_posix())
        else:
            mask = np.load(mask_path)
        mask = np.nan_to_num(mask, nan=0.0).astype(np.float32)
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        spectrum_raw, curves, freq, phase_vel = self._load_record(entry)
        spectrum_np = self._normalize_spectrum(spectrum_raw)
        spectrum = torch.from_numpy(spectrum_np)
        spectrum = apply_spectrum_augmentation(spectrum, self.augment_cfg)
        mask_np = self._materialize_mask(entry, curves, freq, phase_vel)
        mask = torch.from_numpy(mask_np)
        return spectrum, mask
