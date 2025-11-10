"""Training entry point for advanced dispersion models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from .dataset_curves import DispersionCurveDataset, _read_manifest, split_manifest
from .models import FrequencySequenceModel, GraphCurveModel, soft_argmax_velocities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advanced dispersion models")
    parser.add_argument("--config", type=Path, default=Path("label_training/config_advanced.yaml"))
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


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def create_dataloaders(cfg: Dict[str, Any]):
    manifest_path = Path(cfg["paths"]["manifest"])
    entries = _read_manifest(manifest_path)
    splits = split_manifest(entries, cfg["training"]["val_split"], cfg["training"]["seed"])
    dataset_kwargs = {
        "mask_dir": Path(cfg["paths"]["mask_dir"]),
        "auto_generate_masks": cfg["dataset"].get("auto_generate_masks", True),
        "blur_sigma": cfg["dataset"].get("blur_sigma", 1.5),
        "antialiased": cfg["dataset"].get("antialiased", False),
    }
    train_ds = DispersionCurveDataset(splits.train, **dataset_kwargs)
    val_ds = DispersionCurveDataset(splits.val, **dataset_kwargs)
    loader_kwargs = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": cfg["training"].get("num_workers", 0),
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_model(cfg: Dict[str, Any]) -> nn.Module:
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
    if model_type == "graph":
        params = cfg["graph_model"]
        return GraphCurveModel(
            in_channels=params["in_channels"],
            mode_count=params["mode_count"],
            base_channels=params.get("base_channels", 32),
            depth=params.get("depth", 4),
            temperature=params.get("temperature", 1.0),
        )
    raise ValueError(f"Unknown model_type {model_type}")


def sequence_loss(outputs: dict, batch: dict, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    target = batch["velocities"].to(device)
    pred = outputs["velocity_tracks"]
    b, m, _ = target.shape
    target = target.view(b * m, 1, -1)
    target = F.interpolate(target, size=pred.shape[-1], mode="linear", align_corners=False)
    target = target.view(b, m, -1)
    loss = F.smooth_l1_loss(pred, target)
    smooth_lambda = cfg["sequence_model"].get("smooth_lambda", 0.0)
    if smooth_lambda > 0:
        diff = pred[:, :, 1:] - pred[:, :, :-1]
        loss = loss + smooth_lambda * diff.abs().mean()
    return loss


def graph_loss(model: GraphCurveModel, outputs: dict, batch: dict, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    logits = outputs["heatmap_logits"]
    mask = batch["mask"].to(device)
    mask = F.interpolate(mask, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    mask_loss = F.binary_cross_entropy_with_logits(logits, mask)
    velocity_axis = batch["phase_velocity"].to(device)
    velocity_axis = velocity_axis.unsqueeze(1)
    velocity_axis = F.interpolate(
        velocity_axis,
        size=logits.shape[-2],
        mode="linear",
        align_corners=False,
    )
    velocity_axis = velocity_axis.squeeze(1)
    pred_curves = soft_argmax_velocities(
        logits,
        velocity_axis,
        temperature=cfg["graph_model"].get("temperature", 1.0),
    )
    target = batch["velocities"].to(device)
    b, m, _ = target.shape
    target = target.view(b * m, 1, -1)
    target = F.interpolate(target, size=pred_curves.shape[-1], mode="linear", align_corners=False)
    target = target.view(b, m, -1)
    curve_loss = F.smooth_l1_loss(pred_curves, target)
    weights = cfg.get("loss", {})
    return weights.get("mask_weight", 1.0) * mask_loss + weights.get("curve_weight", 1.0) * curve_loss


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        spectra = batch["spectrum"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(spectra)
        if cfg["training"].get("model_type", "sequence") == "sequence":
            loss = sequence_loss(outputs, batch, cfg, device)
        else:
            loss = graph_loss(model, outputs, batch, cfg, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * spectra.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, cfg, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            spectra = batch["spectrum"].to(device)
            outputs = model(spectra)
            if cfg["training"].get("model_type", "sequence") == "sequence":
                loss = sequence_loss(outputs, batch, cfg, device)
            else:
                loss = graph_loss(model, outputs, batch, cfg, device)
            total_loss += loss.item() * spectra.size(0)
    return total_loss / len(loader.dataset)


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: Dict[str, Any], val_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    device = resolve_device(cfg["training"].get("device", "auto"))

    train_loader, val_loader = create_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])

    best_loss = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, cfg, device)
        val_loss = evaluate(model, val_loader, cfg, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        save_checkpoint(checkpoint_dir / "advanced_last.pt", model, optimizer, epoch, cfg, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(checkpoint_dir / "advanced_best.pt", model, optimizer, epoch, cfg, val_loss)
    print(f"Training finished. Best val loss {best_loss:.4f}")


if __name__ == "__main__":
    main()
