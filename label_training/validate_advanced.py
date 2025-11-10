"""Validation for advanced dispersion models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .dataset_curves import DispersionCurveDataset, _read_manifest, split_manifest
from .models import FrequencySequenceModel, GraphCurveModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate advanced dispersion models")
    parser.add_argument("--config", type=Path, default=Path("label_training/config_advanced.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("label_training/checkpoints_advanced/advanced_best.pt"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("label_training/validation_plots_advanced"))
    parser.add_argument("--override", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path, override: str | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = yaml.safe_load(path.read_text())
    override_dict: Dict[str, Any] = {}
    if override:
        override_dict = json.loads(override)
        for key, value in override_dict.items():
            section, _, subkey = key.partition(".")
            if not subkey:
                cfg[section] = value
            else:
                cfg.setdefault(section, {})[subkey] = value
    return cfg, override_dict


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_type = cfg["training"].get("model_type", "sequence")
    if model_type == "sequence":
        params = cfg["sequence_model"]
        return FrequencySequenceModel(
            in_channels=params["in_channels"],
            mode_count=params["mode_count"],
            hidden_dim=params.get("hidden_dim", 256),
            transformer_layers=params.get("transformer_layers", 4),
            transformer_heads=params.get("transformer_heads", 4),
            lstm_layers=params.get("lstm_layers", 2),
            dropout=params.get("dropout", 0.1),
        )
    params = cfg["graph_model"]
    return GraphCurveModel(
        in_channels=params["in_channels"],
        mode_count=params["mode_count"],
        base_channels=params.get("base_channels", 32),
        depth=params.get("depth", 4),
        temperature=params.get("temperature", 1.0),
    )


def resample_axis(axis: np.ndarray, target_len: int) -> np.ndarray:
    if len(axis) == target_len:
        return axis
    return np.linspace(axis.min(), axis.max(), target_len)


def visualize(freq_axis: np.ndarray, phase_axis: np.ndarray, spectrum: np.ndarray, tracks: np.ndarray, entry_id: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    extent = (freq_axis.min(), freq_axis.max(), phase_axis.min(), phase_axis.max())
    ax.imshow(spectrum, extent=extent, origin="lower", aspect="auto", cmap="viridis")
    for mode_idx in range(tracks.shape[0]):
        ax.plot(freq_axis, tracks[mode_idx], label=f"mode {mode_idx}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.set_title(entry_id)
    ax.legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg, overrides = load_config(args.config, args.override)
    device = resolve_device(cfg["training"].get("device", "auto"))

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_cfg = checkpoint.get("config", {})
    if ckpt_cfg:
        ckpt_training = ckpt_cfg.get("training", {})
        if "training.model_type" not in overrides:
            cfg.setdefault("training", {})["model_type"] = ckpt_training.get("model_type", cfg["training"].get("model_type", "sequence"))
        for section in ("sequence_model", "graph_model"):
            if section not in cfg and section in ckpt_cfg:
                cfg[section] = ckpt_cfg[section]

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    manifest = _read_manifest(Path(cfg["paths"]["manifest"]))
    splits = split_manifest(manifest, cfg["training"]["val_split"], cfg["training"]["seed"])
    dataset_kwargs = {
        "mask_dir": Path(cfg["paths"]["mask_dir"]),
        "auto_generate_masks": cfg["dataset"].get("auto_generate_masks", True),
        "blur_sigma": cfg["dataset"].get("blur_sigma", 1.5),
        "antialiased": cfg["dataset"].get("antialiased", False),
    }
    dataset = DispersionCurveDataset(splits.val, **dataset_kwargs)

    rng = np.random.default_rng(cfg["training"].get("seed", 123))
    indices = rng.choice(len(dataset), size=min(args.samples, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[idx]
        entry = sample["entry"]
        spectrum = sample["spectrum"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(spectrum)
        raw_spectrum, _, freq_raw, phase_raw = dataset._load_record(entry)
        if cfg["training"].get("model_type", "sequence") == "sequence":
            tracks = outputs["velocity_tracks"].cpu().squeeze(0).numpy()
            freq_axis = resample_axis(freq_raw, tracks.shape[-1])
            visualize(freq_axis, phase_raw, raw_spectrum, tracks, entry["id"], args.output_dir / f"{entry['id']}.png")
        else:
            logits = outputs["heatmap_logits"]
            vel_axis = torch.from_numpy(phase_raw).float().to(device)
            vel_axis = vel_axis.view(1, 1, -1)
            vel_axis = F.interpolate(vel_axis, size=logits.shape[-2], mode="linear", align_corners=False).squeeze(0).squeeze(0)
            picked = model.pick(logits, vel_axis)
            tracks = picked.cpu().squeeze(0).numpy()
            freq_axis = resample_axis(freq_raw, tracks.shape[-1])
            visualize(freq_axis, phase_raw, raw_spectrum, tracks, entry["id"], args.output_dir / f"{entry['id']}.png")

    print(f"Saved {len(indices)} advanced validation plots to {args.output_dir}")


if __name__ == "__main__":
    main()
