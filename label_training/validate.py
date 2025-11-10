"""Validation script to assess trained models on dispersion data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .dataset import DispersionDataset, _read_manifest
from .model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dispersion segmentation model")
    parser.add_argument("--config", type=Path, default=Path("label_training/config_example.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("label_training/checkpoints/best.pt"))
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--output-dir", type=Path, default=Path("label_training/validation_plots"))
    parser.add_argument("--override", type=str, default=None, help="JSON overrides for the config")
    return parser.parse_args()


def load_config(path: Path, override: str | None) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if override:
        updates = json.loads(override)
        for key, value in updates.items():
            section, _, subkey = key.partition(".")
            if not subkey:
                cfg[section] = value
            else:
                cfg.setdefault(section, {})[subkey] = value
    return cfg


def load_model(cfg: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> UNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
        dropout=cfg["model"].get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def pick_curves(mask: np.ndarray, freq: np.ndarray, phase_velocity: np.ndarray) -> np.ndarray:
    """Extract dispersion curves by picking the argmax along velocity for each frequency bin."""
    picks = []
    for mode in range(mask.shape[0]):
        mode_mask = mask[mode]
        idx = np.argmax(mode_mask, axis=0)
        velocities = phase_velocity[idx]
        picks.append(np.stack([freq, velocities], axis=0))
    return np.stack(picks, axis=0)


def visualize_sample(entry: dict, spectrum: np.ndarray, mask_pred: np.ndarray, freq: np.ndarray, phase_vel: np.ndarray, output_path: Path) -> None:
    curves = pick_curves(mask_pred, freq, phase_vel)
    fig, ax = plt.subplots(figsize=(6, 4))
    extent = (freq.min(), freq.max(), phase_vel.min(), phase_vel.max())
    ax.imshow(spectrum, extent=extent, origin="lower", aspect="auto", cmap="viridis")
    for mode_idx in range(curves.shape[0]):
        ax.plot(curves[mode_idx, 0], curves[mode_idx, 1], label=f"mode {mode_idx}")
    ax.set_title(entry["id"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(cfg, args.checkpoint, device)
    manifest_entries = _read_manifest(Path(cfg["paths"]["manifest"]))
    dataset_cfg = cfg.get("dataset", {})
    dataset = DispersionDataset(
        manifest_entries,
        Path(cfg["paths"]["mask_dir"]),
        auto_generate_masks=dataset_cfg.get("auto_generate_masks", True),
        blur_sigma=dataset_cfg.get("blur_sigma", 1.5),
        antialiased=dataset_cfg.get("antialiased", False),
    )

    rng = np.random.default_rng(cfg["training"].get("seed", 123))
    sample_indices = rng.choice(len(dataset), size=min(args.samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in sample_indices:
            spectrum, mask_gt = dataset[idx]
            entry = manifest_entries[idx]
            spectrum_batch = spectrum.unsqueeze(0).to(device)
            logits = model(spectrum_batch)
            pred = torch.sigmoid(logits).cpu().squeeze(0).numpy()
            spectrum_np, curves, freq, phase_vel = dataset._load_record(entry)
            visualize_sample(
                entry,
                spectrum_np,
                pred,
                freq,
                phase_vel,
                args.output_dir / f"{entry['id']}.png",
            )

    print(f"Saved {len(sample_indices)} validation plots to {args.output_dir}")


if __name__ == "__main__":
    main()
