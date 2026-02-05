"""Scaffolding for the new structured dispersion dataset pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from . import augment, manifest as manifest_utils, synth


@dataclass
class LayerModel:
    thickness_km: List[float]
    vs_kms: List[float]
    seed: int


@dataclass
class SampleRecord:
    sample_id: str
    frequency_hz: np.ndarray
    phase_velocity_ms: np.ndarray
    spectra: np.ndarray
    curves: np.ndarray
    metadata: Dict[str, object]


def generate_layer_stack(
    thickness_bounds: Tuple[float, float],
    vs_bounds: Tuple[float, float],
    layer_count: int,
    rng: np.random.Generator,
) -> LayerModel:
    """Create a physically plausible stack; refine constraints per TODO."""

    thickness = rng.uniform(*thickness_bounds, size=layer_count)
    thickness.sort()  # shallow layers stay thinner

    vs = rng.uniform(*vs_bounds, size=layer_count)
    vs.sort()  # velocities increase with depth

    jitter = 1.0 + 0.05 * rng.standard_normal(layer_count)
    thickness = np.clip(thickness * jitter, thickness_bounds[0], thickness_bounds[1])
    vs = np.clip(vs * jitter, vs_bounds[0], vs_bounds[1])
    seed = int(rng.integers(0, 2**31 - 1))
    return LayerModel(thickness.tolist(), vs.tolist(), seed)


def simulate_dispersion(
    layer: LayerModel,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    mode_count: int,
    fluctuation_percentage: float,
    rng: np.random.Generator,
    params: synth.SimulationParams,
    augmentation_settings: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    thickness = np.asarray(layer.thickness_km)
    vs = np.asarray(layer.vs_kms)
    curves, spectrum = synth.simulate_sample(
        thickness,
        vs,
        frequencies,
        velocities,
        mode_count,
        fluctuation_percentage,
        rng,
        params,
    )
    aug_meta: Dict[str, float] = {}
    if augmentation_settings:
        spectrum, aug_meta = augment.augment_spectrum(spectrum, augmentation_settings, rng)
    return curves, spectrum, aug_meta


def save_structured_record(
    out_dir: Path,
    layer: LayerModel,
    spectra: np.ndarray,
    curves: np.ndarray,
    frequencies: np.ndarray,
    phase_velocity: np.ndarray,
) -> SampleRecord:
    sample_id = uuid.uuid4().hex
    record_path = out_dir / f"{sample_id}.npz"
    record_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        record_path,
        spectrum=spectra,
        curves=curves,
        freq=frequencies,
        phase_velocity=phase_velocity,
    )

    param_payload = {
        "thickness_km": layer.thickness_km,
        "vs_kms": layer.vs_kms,
    }
    metadata = {
        "seed": layer.seed,
        "layer_count": len(layer.thickness_km),
        "thickness_km": layer.thickness_km,
        "vs_kms": layer.vs_kms,
        "mode_count": int(curves.shape[0]),
        "freq_min_hz": float(frequencies.min()),
        "freq_max_hz": float(frequencies.max()),
        "phase_velocity_min_ms": float(phase_velocity.min()),
        "phase_velocity_max_ms": float(phase_velocity.max()),
        "param_hash": manifest_utils.hash_parameters(param_payload),
        "npz": record_path.as_posix(),
    }
    return SampleRecord(sample_id, frequencies, phase_velocity, spectra, curves, metadata)


def append_manifest(manifest_path: Path, sample: SampleRecord) -> None:
    manifest_utils.write_manifest_entry(manifest_path, sample.sample_id, sample.metadata)


def batch_generate(
    records_dir: Path,
    manifest_path: Path,
    frequency_axis: np.ndarray,
    phase_velocity_axis: np.ndarray,
    layer_bounds: Dict[str, Tuple[float, float]],
    layer_count: int,
    mode_count: int,
    fluctuation_percentage: float,
    sample_count: int,
    seed: int,
    simulation_params: synth.SimulationParams,
    augmentation_settings: Dict,
) -> Iterable[SampleRecord]:
    """High-level orchestration entry point used by cli.py."""

    rng = np.random.default_rng(seed)
    records_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    attempts = 0
    while generated < sample_count:
        attempts += 1
        layer = generate_layer_stack(layer_bounds["thickness"], layer_bounds["vs"], layer_count, rng)
        try:
            curves, spectra, aug_meta = simulate_dispersion(
                layer,
                frequency_axis,
                phase_velocity_axis,
                mode_count,
                fluctuation_percentage,
                rng,
                simulation_params,
                augmentation_settings,
            )
        except Exception as exc:  # pragma: no cover - modelling failures
            print(f"[warn] simulation failed (attempt {attempts}): {exc}")
            continue

        sample = save_structured_record(
            records_dir,
            layer,
            spectra,
            curves,
            frequency_axis,
            phase_velocity_axis,
        )
        if aug_meta:
            sample.metadata.update({f"aug_{k}": v for k, v in aug_meta.items()})
        append_manifest(manifest_path, sample)
        generated += 1
        yield sample
