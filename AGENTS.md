# Repository Guidelines

## Project Structure & Module Organization
- `diffseis/` holds the diffusion workflow (`run.py`, `unet.py`, `diffusion.py`) plus training data in `dataset*/` and generated results in `results_*/`; keep bulky artifacts untracked.
- `Dispersion/` supplies classical processing code (`dispersionspectra.py`, `inversion.py`, `gen_data.py`), archived checkpoints, and curated samples inside `datasets_json/`, `dataset_png/`, `Test/`, and `save/`.
- `figs/` stores reference visuals, while exploratory `.ipynb` notebooks live beside the code—promote durable logic back into modules for reuse.

## Build, Test, and Development Commands
- `python diffseis/run.py`: launch diffusion training; update the `mode` and dataset path constants before long runs.
- `python Dispersion/gen_data.py`: regenerate dispersion datasets for training or validation refreshes.
- `python Dispersion/dispersionspectra.py --config Dispersion/Test/example.json`: render spectra for a sample configuration and drop outputs under `Dispersion/save/`.

## Coding Style & Naming Conventions
- Use four-space indentation, snake_case functions, and UpperCamelCase classes consistent with `diffusion.py` and `unet.py`.
- Keep device transfers explicit (`.cuda()`, autocast context managers) and seed randomness (`torch.manual_seed`) in new scripts for reproducibility.
- Format Python with `black` (line length 120) and `isort`; arrange imports by standard library, third-party, then local modules.

## Testing Guidelines
- Add lightweight `pytest` modules under `tests/` when extending core logic, mocking loaders with a handful of samples from `diffseis/dataset/demultiple/`.
- Run a short dry pass (`python diffseis/run.py --max-steps 10`, add the flag if missing) to confirm data, model, and checkpoint integration.
- For dispersion utilities, execute the spectra command above and review the plots saved in `Dispersion/save/` before submitting changes.

## Commit & Pull Request Guidelines
- Mirror the concise history style (examples: `update`, `a result`): start with a lowercase imperative verb and keep the subject under 60 characters.
- Reference related issues (`Refs #123`), flag dataset or checkpoint updates, and attach before/after plots when behaviour shifts.
- Pull requests should explain the scientific intent, include reproduction commands, and note any assets that need syncing outside Git.

## Data & Checkpoint Handling
- Store regenerated datasets, `.pth` weights, and large renders outside Git or under Git LFS; avoid committing notebook output noise.
- Document provenance for new data drops in a short README alongside the folder so future runs stay traceable.
