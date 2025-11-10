# Advanced Dispersion Modeling TODO

## 1. CNN/Transformer Encoder + LSTM Decoder
- [ ] Design encoder backbone (e.g., ResNet/UNet or ViT-style) that compresses spectra into frequency tokens.
- [ ] Flatten encoder outputs along the frequency axis to form sequences with positional encodings.
- [ ] Implement multi-layer LSTM/GRU decoder that iteratively predicts per-frequency velocity logits for each mode.
- [ ] Add loss tailored to sequence predictions (e.g., MSE on velocities, smoothness regularization) alongside existing mask loss.
- [ ] Provide inference helper that converts decoder outputs into curve arrays compatible with validators.

## 2. Graph-Based Curve Fitting with Differentiable Picking
- [ ] Extend encoder to output energy heatmaps for each mode (multi-channel logits).
- [ ] Implement differentiable soft-argmax / dynamic time warping head that converts heatmaps to continuous velocity curves.
- [ ] Compute losses directly on picked curves (e.g., L1/L2 vs. ground truth curves) and optionally on heatmaps.
- [ ] Explore graph/CRF smoothing over the picked sequences to enforce monotonicity and continuity.
- [ ] Integrate curve outputs with existing manifest/inspection tools for quantitative and qualitative evaluation.

Both branches should live beside the current UNet in a new module (e.g., `label_training/models/advanced.py`) so the baseline remains untouched.
