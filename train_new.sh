
python scripts/gen_dataset_npz.py --out data/demultiple/npz \
  --num-train 2000 --num-val 200 --overwrite




python scripts/train_picker.py \
  --dataset-root data/demultiple/npz \
  --out runs/picker \
  --steps 50000 \
  --dp-max-jump 6 \
  --dp-lambda-smooth 2.0 \
  --dp-const-null 3.0



python scripts/eval_picker.py --dataset-root data/demultiple/npz --split val \
  --picker-ckpt runs/picker/picker_final.pt --out runs/eval


python scripts/plot_pick.py --dataset-root data/demultiple/npz \
  --picker-ckpt runs/picker/picker_final.pt --random --n 10 