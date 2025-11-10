"""Typed configuration loader for the label refactor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml


@dataclass
class FrequencyConfig:
    min_hz: float
    max_hz: float
    num_samples: int

    def axis(self) -> np.ndarray:
        return np.linspace(self.min_hz, self.max_hz, self.num_samples, dtype=np.float32)


@dataclass
class PhaseVelocityConfig:
    min_ms: float
    max_ms: float
    num_samples: int

    def axis(self) -> np.ndarray:
        return np.linspace(self.min_ms, self.max_ms, self.num_samples, dtype=np.float32)


@dataclass
class SimulationConfig:
    nt: int
    dt: float
    nx: int
    dx: float
    nfft: int


@dataclass
class PhysicsConfig:
    modes: int
    layer_count: int
    fluctuation_percentage: float
    vs_bounds_kms: Tuple[float, float]
    thickness_bounds_km: Tuple[float, float]

    @property
    def layer_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "thickness": tuple(self.thickness_bounds_km),
            "vs": tuple(self.vs_bounds_kms),
        }


@dataclass
class PathsConfig:
    output_root: Path
    spectra_subdir: str
    curves_subdir: str
    mask_subdir: str
    manifest_path: Path

    @classmethod
    def from_dict(cls, base_dir: Path, data: Dict) -> "PathsConfig":
        output_root = (base_dir / data["output_root"]).resolve()
        manifest_path = (base_dir / data["manifest_path"]).resolve()
        return cls(
            output_root=output_root,
            spectra_subdir=data.get("spectra_subdir", "spectra"),
            curves_subdir=data.get("curves_subdir", "curves"),
            mask_subdir=data.get("mask_subdir", "masks"),
            manifest_path=manifest_path,
        )

    @property
    def curves_dir(self) -> Path:
        return self.output_root / self.curves_subdir

    @property
    def spectra_dir(self) -> Path:
        return self.output_root / self.spectra_subdir

    @property
    def mask_dir(self) -> Path:
        return self.output_root / self.mask_subdir


@dataclass
class MaskConfig:
    channel_count: int
    blur_sigma_px: float
    antialiased: bool


@dataclass
class CLIConfig:
    dry_run_samples: int
    seed: int


@dataclass
class Config:
    frequency     : FrequencyConfig
    phase_velocity: PhaseVelocityConfig
    simulation    : SimulationConfig
    physics       : PhysicsConfig
    paths         : PathsConfig
    mask          : MaskConfig
    cli           : CLIConfig

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        data = yaml.safe_load(path.read_text())
        base_dir = path.parent
        return cls(
            frequency=FrequencyConfig(**data["frequency"]),
            phase_velocity=PhaseVelocityConfig(**data["phase_velocity"]),
            simulation=SimulationConfig(**data["simulation"]),
            physics=PhysicsConfig(**data["physics"]),
            paths=PathsConfig.from_dict(base_dir, data["paths"]),
            mask=MaskConfig(**data["mask"]),
            cli=CLIConfig(**data["cli"]),
        )

    @property
    def frequency_axis(self) -> np.ndarray:
        return self.frequency.axis()

    @property
    def phase_velocity_axis(self) -> np.ndarray:
        return self.phase_velocity.axis()

    @property
    def layer_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self.physics.layer_bounds
