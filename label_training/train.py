"""Training CLI for dispersion segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from .dataset import DispersionDataset, split_manifest, _read_manifest
from .model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dispersion segmentation model")
    parser.add_argument("--config", type=Path, default=Path("label_training/config_example.yaml"))
    parser.add_argument("--override", type=str, default=None, help="Optional JSON dict to override config")
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def create_dataloaders(cfg: Dict[str, Any]):
    manifest_path = Path(cfg["paths"]["manifest"])
    mask_dir = Path(cfg["paths"]["mask_dir"])
    dataset_cfg = cfg.get("dataset", {})
    augment_cfg = dataset_cfg.get("augment")
    val_augment = augment_cfg if dataset_cfg.get("augment_validation", False) else None
    entries = _read_manifest(manifest_path)
    splits = split_manifest(entries, cfg["training"]["val_split"], cfg["training"]["seed"])
    train_ds = DispersionDataset(
        splits.train,
        mask_dir,
        auto_generate_masks=dataset_cfg.get("auto_generate_masks", True),
        blur_sigma=dataset_cfg.get("blur_sigma", 1.5),
        antialiased=dataset_cfg.get("antialiased", False),
        augment_cfg=augment_cfg,
    )
    val_ds = DispersionDataset(
        splits.val,
        mask_dir,
        auto_generate_masks=dataset_cfg.get("auto_generate_masks", True),
        blur_sigma=dataset_cfg.get("blur_sigma", 1.5),
        antialiased=dataset_cfg.get("antialiased", False),
        augment_cfg=val_augment,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, log_every):
    model.train()
    total_loss = 0.0
    for step, (spectra, masks) in enumerate(loader, start=1):
        spectra = spectra.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(spectra)
        loss = criterion(logits, masks)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * spectra.size(0)
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)} loss={loss.item():.4f}")
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for spectra, masks in loader:
            spectra = spectra.to(device)
            masks = masks.to(device)
            logits = model(spectra)
            loss = criterion(logits, masks)
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
    set_seed(cfg["training"]["seed"])
    device = resolve_device(cfg["training"].get("device", "auto"))

    train_loader, val_loader = create_dataloaders(cfg)
    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg.get("optimization", {}).get("weight_decay", 0.0),
    )

    epochs = cfg["training"]["epochs"]
    grad_clip = cfg.get("optimization", {}).get("grad_clip")
    log_every = cfg.get("logging", {}).get("log_every", 50)

    best_loss = float("inf")
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip, log_every)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, cfg, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, cfg, val_loss)

    print(f"Training complete. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
