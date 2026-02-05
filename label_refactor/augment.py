"""Spectrum augmentation utilities for dispersion data."""

from __future__ import annotations

from typing import Dict

import numpy as np


def add_gaussian_noise(image: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return image
    signal_power = np.mean(image ** 2)
    if signal_power == 0:
        return image
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=image.shape)
    return image + noise


def apply_patch_occlusion(
    image: np.ndarray,
    max_patches: int,
    freq_fraction: float,
    vel_fraction: float,
    attenuation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_patches <= 0 or freq_fraction <= 0 or vel_fraction <= 0:
        return image
    h, w = image.shape
    result = image.copy()
    for _ in range(max_patches):
        if rng.random() < 0.5:
            continue
        f_span = max(1, int(freq_fraction * w * rng.uniform(0.5, 1.0)))
        v_span = max(1, int(vel_fraction * h * rng.uniform(0.5, 1.0)))
        f0 = rng.integers(0, max(1, w - f_span))
        v0 = rng.integers(0, max(1, h - v_span))
        lo, hi = sorted((attenuation * 0.5, attenuation))
        result[v0 : v0 + v_span, f0 : f0 + f_span] *= rng.uniform(lo, hi)
    return result


def warp_axes(image: np.ndarray, max_freq_shift: float, max_vel_shift: float, rng: np.random.Generator) -> np.ndarray:
    if max_freq_shift <= 0 and max_vel_shift <= 0:
        return image
    h, w = image.shape
    warped = image.copy()
    freq_indices = np.arange(w)
    vel_indices = np.arange(h)
    freq_shift = rng.normal(0, max_freq_shift, size=w)
    vel_shift = rng.normal(0, max_vel_shift, size=h)
    for i in range(h):
        shift = vel_shift[i]
        warped[i] = np.interp(freq_indices + shift, freq_indices, warped[i], left=0.0, right=0.0)
    for j in range(w):
        shift = freq_shift[j]
        column = warped[:, j]
        warped[:, j] = np.interp(vel_indices + shift, vel_indices, column, left=0.0, right=0.0)
    return warped


def augment_spectrum(image: np.ndarray, config: Dict, rng: np.random.Generator) -> tuple[np.ndarray, Dict]:
    augmented = image
    metadata: Dict[str, float] = {}
    noise_cfg = config.get("noise", {})
    if noise_cfg.get("enabled", False):
        snr_db = noise_cfg.get("snr_db", 20.0)
        augmented = add_gaussian_noise(augmented, snr_db, rng)
        metadata["noise_snr_db"] = snr_db
    occlusion_cfg = config.get("occlusion", {})
    if occlusion_cfg.get("enabled", False):
        augmented = apply_patch_occlusion(
            augmented,
            max_patches=occlusion_cfg.get("max_patches", 3),
            freq_fraction=occlusion_cfg.get("max_freq_fraction", 0.2),
            vel_fraction=occlusion_cfg.get("max_vel_fraction", 0.2),
            attenuation=occlusion_cfg.get("attenuation", 0.3),
            rng=rng,
        )
        metadata["occlusion_patches"] = occlusion_cfg.get("max_patches", 3)
    warp_cfg = config.get("warp", {})
    if warp_cfg.get("enabled", False):
        augmented = warp_axes(
            augmented,
            max_freq_shift=warp_cfg.get("max_freq_shift", 1.0),
            max_vel_shift=warp_cfg.get("max_vel_shift", 1.0),
            rng=rng,
        )
        metadata["warp_freq"] = warp_cfg.get("max_freq_shift", 1.0)
        metadata["warp_vel"] = warp_cfg.get("max_vel_shift", 1.0)
    augmented = np.clip(augmented, a_min=0.0, a_max=None)
    return augmented, metadata
