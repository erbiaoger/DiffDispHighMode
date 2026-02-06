"""Dataset-time augmentations for dispersion spectra."""

from __future__ import annotations

from typing import Dict

import torch


def apply_spectrum_augmentation(tensor: torch.Tensor, config: Dict) -> torch.Tensor:
    if not config or not config.get("enabled", False):
        return tensor
    x = tensor
    if config.get("gaussian_noise", False):
        std = config.get("noise_std", 0.02)
        noise = torch.randn_like(x) * std
        x = x + noise
    if config.get("cutout", False):
        freq_frac = config.get("cutout_freq_fraction", 0.2)
        vel_frac = config.get("cutout_vel_fraction", 0.2)
        freq_span = max(1, int(x.shape[-1] * freq_frac * torch.rand(1).item()))
        vel_span = max(1, int(x.shape[-2] * vel_frac * torch.rand(1).item()))
        f0 = torch.randint(0, max(1, x.shape[-1] - freq_span + 1), (1,)).item()
        v0 = torch.randint(0, max(1, x.shape[-2] - vel_span + 1), (1,)).item()
        x = x.clone()
        x[..., v0 : v0 + vel_span, f0 : f0 + freq_span] = 0.0
    if config.get("random_gain", False):
        gain = torch.empty(1, 1, x.shape[-1], device=x.device).uniform_(0.8, 1.2)
        x = x * gain
    return x.clamp(min=0.0)
