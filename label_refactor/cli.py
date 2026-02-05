"""Entry point to exercise the label refactor pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config as config_module
from . import dataset, manifest, mask, synth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured dispersion labels")
    parser.add_argument("--config", type=Path, default=Path("label_refactor/config_sample.yaml"))
    parser.add_argument("--samples", type=int, default=None, help="Override sample count from config")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed from config")
    parser.add_argument(
        "--validate-manifest",
        action="store_true",
        help="Run manifest deduplication checks after generation",
    )
    return parser.parse_args()


def main() -> None:
    args          = parse_args()
    cfg           = config_module.Config.from_file(args.config)
    freq_axis     = cfg.frequency_axis
    vel_axis      = cfg.phase_velocity_axis
    manifest_path = cfg.paths.manifest_path
    records_dir   = cfg.paths.curves_dir
    mask_dir      = cfg.paths.mask_dir

    sample_count = args.samples if args.samples is not None else cfg.cli.dry_run_samples
    seed = args.seed if args.seed is not None else cfg.cli.seed

    dc = (cfg.phase_velocity.max_ms - cfg.phase_velocity.min_ms) / max(cfg.phase_velocity.num_samples - 1, 1)
    sim_params = synth.SimulationParams(
        nt=cfg.simulation.nt,
        dt=cfg.simulation.dt,
        nx=cfg.simulation.nx,
        dx=cfg.simulation.dx,
        nfft=cfg.simulation.nfft,
        cmin=cfg.phase_velocity.min_ms,
        cmax=cfg.phase_velocity.max_ms,
        dc=dc,
        fmin=cfg.frequency.min_hz,
        fmax=cfg.frequency.max_hz,
    )

    for sample in dataset.batch_generate(
        records_dir=records_dir,
        manifest_path=manifest_path,
        frequency_axis=freq_axis,
        phase_velocity_axis=vel_axis,
        layer_bounds=cfg.layer_bounds,
        layer_count=cfg.physics.layer_count,
        mode_count=cfg.physics.modes,
        fluctuation_percentage=cfg.physics.fluctuation_percentage,
        sample_count=sample_count,
        seed=seed,
        simulation_params=sim_params,
        augmentation_settings=cfg.augmentation_settings,
    ):
        mask_path = mask_dir / f"{sample.sample_id}.npy"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_array = mask.rasterize_modes(
            sample.curves,
            freq_axis,
            vel_axis,
            blur_sigma=cfg.mask.blur_sigma_px,
            antialiased=cfg.mask.antialiased,
        )
        mask.save_mask(mask_array, mask_path.as_posix())

    if args.validate_manifest:
        manifest.validate_manifest(manifest_path)


if __name__ == "__main__":
    main()
