# Microscopy Segmentation Trainer/Inferencer

Patch-based 2D/3D microscopy image segmentation with MONAI- and PyTorch-based
training, single-volume inference, and batch inference across a dataset tree.

## Features

- Unified dataset: one class for 2D/3D and train/inference (`utils/datasets.py`).
- UNet 2D/3D models (see `models/`).
- Training with Dice+BCE loss and Dice metric curves (PNG) saved per run.
- Sliding-window style inference with patch stitching and multi-format output.
- Batch inference over `<input_dir>/images/<subfolder>`.

## Install

1) Create a Python 3.10+ environment.
2) Install requirements:

```
pip install -r requirements.txt
```

## Data Layout

Training expects separate image and mask trees with matching subfolders:

```
<img_root>/01/  <mask_root>/01/
<img_root>/02/  <mask_root>/02/
...
```

Inference/test read volumes from a folder (e.g., a subfolder with TIF stack or
NIfTI) and write the prediction to a chosen output format.

## Training

`train.py` auto-selects 2D vs 3D based on the z-size in `--training_patch_size`
(z>1 → 3D, z==1 → 2D). Example (Windows caret shown; use `\` on Unix):

```
python train.py ^
  --img_path ./datas/c-Fos/LI-WIN_PAPER/training-data/100-3D/images ^
  --mask_path ./datas/c-Fos/LI-WIN_PAPER/training-data/100-3D/masks ^
  --save_path ./datas/c-Fos/LI-WIN_PAPER/weights ^
  --model_name func-3 ^
  --training_epochs 100 ^
  --training_batch_size 64 ^
  --training_patch_size 1 64 64 ^
  --training_overlay 0 16 16 ^
  --training_resize_factor 1 1 1 ^
  --visualize_preview
```

Notes:
- Loss: Dice+BCE (`train/loss.py:dice_bce_loss`). Trainer logs loss and soft Dice (1−dice_loss).
- Curves: saved under `--save_path` as `<model_name>-metrics_curve.png`.

## Single-Volume Inference

`inference.py` runs inference on one volume. The dimensionality is inferred from
`--inference_patch_size` (z>1 → 3D).

```
python inference.py \
  --img_path ./datas/c-Fos/YYC/testing-data/YYC_20230414-1/images \
  --mask_path ./datas/c-Fos/YYC/testing-data/YYC_20230414-1/results \
  --model_path ./datas/c-Fos/YYC/weights/c-Fos_200_LI_AN.pth \
  --output_type scroll-tiff \
  --inference_patch_size 16 64 64 \
  --inference_overlay 2 4 4 \
  --inference_resize_factor 1 1 1
```

Outputs can be `zarr`, `ome-zarr`, `single-tiff`, `scroll-tiff`, `single-nii`, or `scroll-nii`.

## Batch Inference

`test.py` traverses `<input_dir>/images/<subfolder>` and writes predictions
under `<input_dir>/masks_{model_stem}/<subfolder>`.

```
python test.py \
  --input_dir ./datas/c-Fos/LI-WIN_PAPER/testing-data/V60 \
  --model_path ./datas/c-Fos/LI-WIN_PAPER/weights/fun-3.pth \
  --inference_patch_size 16 64 64 \
  --inference_overlay 2 4 4 \
  --output_type scroll-tiff
```

If using `scroll-*` outputs, files are moved up from the temporary
`<volume_name>_scroll` directory to the subfolder root.

## Unified Dataset

Use `MicroscopyDataset` for both training and inference:

```
from utils.datasets import MicroscopyDataset

ds_train = MicroscopyDataset(train_patches, transform=train_transform, spatial_dims=3, with_mask=True)
ds_infer = MicroscopyDataset(infer_patches, transform=inference_transform, spatial_dims=2, with_mask=False)
```

`spatial_dims` chooses 2D vs 3D; `with_mask` controls whether `(image, mask)` or `image` is returned.

## Class Imbalance Tips

For sparse targets, consider Tversky (alpha<beta) or Focal-based losses
(`train/loss.py` provides Dice, BCE combos, and Tversky variants). You can also
increase the fraction of foreground patches in the sampler.

## Troubleshooting

- Windows: Use PowerShell carets `^` for line continuations as shown.
- If figures fail to overwrite on Windows, the trainer also saves epoch-named snapshots.

