# Dispersion Training Prototype

This package turns the structured curves/masks emitted by `label_refactor` into a simple training/validation pipeline. It targets semantic segmentation of dispersion spectra (predicting multi-mode masks from the spectral image) and is intentionally lightweight so you can iterate quickly before moving to a larger framework.

## Layout

- `config_example.yaml` — wiring between manifests, mask directories, and hyperparameters.
- `dataset.py` — `torch.utils.data.Dataset` that pairs `.npz` spectra with `.npy` masks and handles train/val splits.
- `dataset.py` can auto-generate masks on the fly if a stored `.npy` is missing (enabled via `dataset.auto_generate_masks`).
- `model.py` — small UNet-style encoder/decoder suitable for 2D spectral masks.
- `train.py` — CLI entry point that runs training plus validation, saving the best checkpoint under `output.checkpoint_dir`.
- `validate.py` — loads a trained checkpoint, runs inference on random samples, and overlays picked curves on the spectra for qualitative review.
- `models/advanced.py` & `train_advanced.py` — experimental Transformer+LSTM and graph-based models with their own trainer.
- `validate_advanced.py` — runs inference with the advanced checkpoints and overlays the predicted velocity tracks.

## Dependencies

```
pip install torch torchvision pyyaml numpy
```
Make sure you have already generated data with `label_refactor.cli`, since the training pipeline expects `.npz` files and masks to exist.

## Quick Start

1. Update `label_training/config_example.yaml` so the `paths.manifest`, `paths.records_dir`, and `paths.mask_dir` values point at your generated dataset (the defaults match the sample config from `label_refactor`).
2. Kick off a dry run:
   ```bash
   python -m label_training.train --config label_training/config_example.yaml
   ```
   - Training/validation splits are derived from the manifest using the `training.val_split` ratio.
   - Checkpoints land under `label_training/checkpoints` by default, with `best.pt` storing the lowest validation loss model and `last.pt` capturing the latest epoch.
3. Adjust hyperparameters (epochs, learning rate, number of channels) inside the YAML and re-run. Set `training.device` to `cuda` if you have a GPU available. If you toggle `dataset.auto_generate_masks` to `false`, be sure masks exist on disk beforehand.

### Post-training validation

Generate qualitative plots from the trained checkpoint:

```bash
python -m label_training.validate --config label_training/config_example.yaml \
       --checkpoint label_training/checkpoints/best.pt --samples 4
```

PNG files with predicted curves land under `label_training/validation_plots/` by default. Use `--output-dir` or `--samples` to change the destination or the number of visualizations.

### Monitoring

`train.py` prints per-epoch losses; for deeper inspection, you can plug the loop into TensorBoard or wandb later. The saved checkpoints include the config and final validation loss for reproducibility.

### Advanced Models (experimental)

Use `label_training/config_advanced.yaml` together with `train_advanced.py` to try the Transformer+LSTM or graph-based variants:

```bash
python -m label_training.train_advanced --config label_training/config_advanced.yaml \
       --override '{"training.model_type":"graph"}'
```

Set `training.model_type` to `sequence` for the encoder+LSTM approach or `graph` for the differentiable picking head. The advanced datasets automatically expose ground-truth curves from each `.npz`, and the trainer optimizes both track-level losses and (optionally) mask heatmaps.

After training, visualize the advanced model’s predictions:

```bash
python -m label_training.validate_advanced --config label_training/config_advanced.yaml \
       --checkpoint label_training/checkpoints_advanced/advanced_best.pt --samples 4
```

Images are written to `label_training/validation_plots_advanced/` (configurable via `--output-dir`).
