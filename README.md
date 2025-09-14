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

## Docker

The repo includes a GPU-enabled Docker workflow so you only need to mount your data; all code is baked into the image.

- Image: built from `Dockerfile` (CUDA 12.9, Ubuntu 22.04). Copies `train.py`, `inference.py`, and the `IO/`, `models/`, `utils/`, `train/`, `inference/` packages into the container.
- Volume: only `./datas` on the host is bind-mounted to `/workspace/datas` inside the container.
- GPU: requires NVIDIA GPU drivers and the NVIDIA Container Toolkit.

Quick start

- Windows PowerShell: `./run.ps1`
- macOS/Linux: `bash run.sh`
- Cross‑platform Python: `python run.py`

What the runner does
- Generates a temporary `docker-compose.yml` with the single volume mount for `datas/`.
- Builds the image and starts an interactive container (`bash`) in `/workspace`.
- On exit (Ctrl+C), stops and removes the container, deletes the generated compose file, and prunes dangling images.

Notes
- Code changes require a rebuild because code is copied into the image. The runners already use `--build` to rebuild as needed.
- From Windows Command Prompt (cmd.exe), run PowerShell or Bash explicitly, e.g.: `powershell -ExecutionPolicy Bypass -File run.ps1` or `bash run.sh`.
- Ensure `datas/` exists (the scripts create it if missing).

Inside the container

Run training/inference exactly as you would locally, using paths under `/workspace/datas`:

```
python train.py \
  --img_path /workspace/datas/<your>/images \
  --mask_path /workspace/datas/<your>/masks \
  --save_path /workspace/datas/<your>/weights \
  --model_name my-model \
  --training_patch_size 1 64 64

python inference.py \
  --img_path /workspace/datas/<your>/testing/images \
  --mask_path /workspace/datas/<your>/testing/results \
  --model_path /workspace/datas/<your>/weights/my-model.pth \
  --output_type scroll-tiff \
  --inference_patch_size 16 64 64
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

## Troubleshooting

- Windows: Use PowerShell carets `^` for line continuations as shown.
- If figures fail to overwrite on Windows, the trainer also saves epoch-named snapshots.
