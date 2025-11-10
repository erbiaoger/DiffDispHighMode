"""Randomly load a structured dispersion sample and plot spectrum + curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import config as config_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect generated dispersion samples")
    parser.add_argument("--config", type=Path, default=Path("label_refactor/config_example.yaml"))
    parser.add_argument("--sample-id", type=str, default=None, help="Specify a sample id; otherwise choose randomly")
    parser.add_argument("--output", type=Path, default=Path("label_refactor/plots/preview.png"))
    parser.add_argument("--seed", type=int, default=None, help="Seed controlling random selection when sample-id is omitted")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true", help="Display the figure interactively in addition to saving")
    return parser.parse_args()


def _pick_sample(records_dir: Path, sample_id: str | None, seed: int | None) -> Path:
    if sample_id is not None:
        candidate = records_dir / f"{sample_id}.npz"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find sample {candidate}")
        return candidate

    files = sorted(records_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {records_dir}")
    rng = np.random.default_rng(seed)
    return files[int(rng.integers(0, len(files)))]


def _plot_sample(sample_path: Path, output_path: Path, show: bool, dpi: int) -> None:
    data = np.load(sample_path)
    freq = data["freq"]
    phase_velocity = data["phase_velocity"]
    spectrum = data["spectrum"]
    curves = data["curves"]

    fig, ax = plt.subplots(figsize=(7, 4))
    extent = (float(freq.min()), float(freq.max()), float(phase_velocity.min()), float(phase_velocity.max()))
    im = ax.imshow(
        spectrum,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    for mode_idx in range(curves.shape[0]):
        ax.plot(curves[mode_idx, 0], curves[mode_idx, 1], label=f"mode {mode_idx}", linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.set_title(sample_path.name)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Spectral amplitude")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = config_module.Config.from_file(args.config)
    sample_path = _pick_sample(cfg.paths.curves_dir, args.sample_id, args.seed or cfg.cli.seed)
    _plot_sample(sample_path, args.output, args.show, args.dpi)


if __name__ == "__main__":  # pragma: no cover
    main()
