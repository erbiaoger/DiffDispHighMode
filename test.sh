#!/usr/bin/env zsh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dasQt
rm -f label_refactor/manifests/labels_test.jsonl
python -m label_refactor.cli --config label_refactor/config_test_normal.yaml --samples 80
python -m label_refactor.cli --config label_refactor/config_test_interference.yaml --samples 20

python -m label_training.validate_advanced --config label_training/config_advanced.yaml \
       --checkpoint label_training/checkpoints_advanced/advanced_best.pt \
       --manifest label_refactor/manifests/labels_test.jsonl --samples 4


python -m label_training.validate_advanced --config label_training/config_advanced.yaml \
       --checkpoint label_training/checkpoints_advanced/advanced_best.pt \
       --manifest label_refactor/manifests/labels_test.jsonl --samples 4