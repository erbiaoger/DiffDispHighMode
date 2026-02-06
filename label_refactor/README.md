# Label Refactor Prototype

This package tracks the refactor of the dispersion label generation pipeline described in `LABEL_REFACTOR_TODO.md`. It focuses on three pillars:

1. Saving structured dispersion curves together with spectra and metadata so training does not rely on heuristic image labels.
2. Rendering mode-specific masks on demand using differentiable-friendly rasterization.
3. Capturing manifests plus CLI utilities that make samples reproducible.

## Repository Layout

- `config_example.yaml` — minimal configuration showing how to set frequency grids, paths, and physical constraints.
- `config.py` — dataclasses plus YAML loader powering the CLI.
- `dataset.py` — helpers to sample layered media, run dispersion synthesis hooks, and assemble `.npz` records.
- `mask.py` — utilities to convert structured curves into smooth multi-channel masks.
- `manifest.py` — tools for emitting JSON/CSV manifests alongside generated data.
- `cli.py` — thin wrapper to run dry/production passes aligned with the TODO list.
- `augment.py` — spectrum-level perturbations (noise, occlusion, axis warps) applied during data generation.
- `inspect.py` — quick-look utility to sample a `.npz` record and visualize spectrum + curves.

Each module currently contains scaffolding and TODO annotations so the work can proceed incrementally.

## How to Run

1. Install dependencies (from the repo root):
   ```bash
   pip install numpy scipy pyyaml matplotlib disba pylops scikit-image
   ```
   `scikit-image` is optional but enables anti-aliased masks; without it the CLI falls back to nearest-neighbour rasterization.
   `disba` and `pylops` are required because the new synthesizer reuses the legacy dispersion modelling kernels.
2. Edit `label_refactor/config_example.yaml` (or copy it) to point `paths.output_root` at your dataset root and tune the physics/grid settings.
3. Launch a dry run:
   ```bash
   python -m label_refactor.cli --config label_refactor/config_example.yaml --samples 4 --seed 123 --validate-manifest
   ```
- `.npz` records land under `<output_root>/<curves_subdir>`
- Mask tensors land under `<output_root>/<mask_subdir>`
- `manifests/labels.jsonl` aggregates metadata plus a `param_hash` for deduplication
- Enable/disable spectrum noise, occlusion, or warping via the `augmentation` block in the config; parameters are stored back into the manifest for provenance.
4. Add `--validate-manifest` whenever you want to ensure there are no duplicate IDs or hashes after generation.

You can override the sample count or seed directly on the CLI; other knobs (frequency grids, physics constraints, blur sigma, etc.) live in the YAML config.

### Building a mixed test set

To create a dedicated test manifest that mixes clean and interference-heavy spectra:

1. Remove any stale test manifest: `rm -f label_refactor/manifests/labels_test.jsonl`.
2. Generate the clean portion:
   ```bash
   python -m label_refactor.cli --config label_refactor/config_test_normal.yaml --samples 800 --validate-manifest
   ```
3. Append interference-rich samples (reuses the same manifest/path layout):
   ```bash
   python -m label_refactor.cli --config label_refactor/config_test_interference.yaml --samples 200 --validate-manifest
   ```

Both configs write records beneath `diffseis/dataset/demultiple/data_test` and append metadata to `label_refactor/manifests/labels_test.jsonl`. Adjust the `--samples` counts (or the `cli.dry_run_samples` values) to control the balance between normal and interference data. The manifest lines include augmentation metadata (e.g. `aug_noise_snr_db`) so downstream loaders know which samples contained simulated interference.

### Quick Visualization

After generating data you can preview a random sample:

```bash
python -m label_refactor.inspect --config label_refactor/config_example.yaml --output label_refactor/plots/sample.png
```

- Use `--sample-id <hex>` to pick a specific record (matching the `.npz` filename).
- Pass `--show` if you have a GUI session and want the plot window in addition to the saved PNG.
- The script overlays every mode curve on top of the spectrum using the same axes stored inside the `.npz` file.
