# Label Generation Refactor TODO

## Structured Curve Records
- [ ] Resample each Rayleigh mode curve onto a fixed frequency grid shared across samples.
- [ ] Package dispersion spectra and resampled curves together (e.g., `.npz` with keys `spectrum`, `curves`, `freq`, `phase_velocity`).
- [ ] Persist provenance metadata (thickness/velocity vectors, random seed, mode count) alongside each sample record.

## Multi-Channel Label Masks
- [ ] Rasterize each mode into its own channel using sub-pixel accurate drawing (e.g., `skimage.draw.line_aa`).
- [ ] Apply a small Gaussian blur so supervision covers a corridor instead of single pixels.
- [ ] Provide utilities to convert structured curves into masks on demand during training.

## File Naming & Metadata Manifests
- [ ] Replace float-concatenated filenames with UUIDs or hashed identifiers.
- [ ] Emit a manifest (`labels.jsonl` or CSV) that links filenames to their physical parameters and randomization settings.
- [ ] Validate manifest entries for uniqueness and completeness.

## Model Sampling & Validation
- [ ] Pre-filter random layer stacks to enforce physical constraints before launching expensive simulations.
- [ ] Log unsuccessful simulations with reasons for failure to refine sampling ranges.
- [ ] Add a CLI dry-run mode that generates a handful of samples for quick inspection.

## Training-Time Integration
- [ ] Update data loaders to read the structured records and lazily rasterize masks when needed.
- [ ] Offer switches to choose between raw `.npz` curves and cached mask images.
- [ ] Document example training commands plus visualization scripts under `label_refactor/`.
