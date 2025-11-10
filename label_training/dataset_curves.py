"""Dataset returning spectra, masks, and curve supervision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

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
    entries = list(entries)
    rng = np.random.default_rng(seed)
    rng.shuffle(entries)
    val_size = max(1, int(len(entries) * val_split))
    train_entries = entries[val_size:]
    val_entries = entries[:val_size]
    return DatasetSplit(train=train_entries, val=val_entries)


class DispersionCurveDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[dict],
        mask_dir: Path,
        auto_generate_masks: bool = True,
        blur_sigma: float = 1.5,
        antialiased: bool = False,
    ):
        self.entries = list(entries)
        self.mask_dir = Path(mask_dir)
        self.auto_generate_masks = auto_generate_masks
        self.blur_sigma = blur_sigma
        self.antialiased = antialiased
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_record(self, entry: dict):
        with np.load(Path(entry["npz"])) as data:
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

    def _ensure_mask(self, entry: dict, curves: np.ndarray, freq: np.ndarray, phase_velocity: np.ndarray) -> np.ndarray:
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
        return np.nan_to_num(mask, nan=0.0).astype(np.float32)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        spectrum, curves, freq, phase_vel = self._load_record(entry)
        mask = self._ensure_mask(entry, curves, freq, phase_vel)
        # curves array shape (modes, 2, freq). Extract velocities component.
        velocities = curves[:, 1]
        return {
            "entry": entry,
            "spectrum": torch.from_numpy(self._normalize_spectrum(spectrum)),
            "mask": torch.from_numpy(mask),
            "velocities": torch.from_numpy(velocities),
            "freq": torch.from_numpy(freq),
            "phase_velocity": torch.from_numpy(phase_vel),
        }
