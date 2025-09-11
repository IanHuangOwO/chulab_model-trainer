"""
Train a 2D/3D U-Net-style segmentation model on microscopy data using patch-based
sampling. Images and masks are loaded as volumes per subfolder, converted into
patches with optional overlap and resize, then split into train/validation sets.
During training, loss and Dice score are tracked and saved to figures; the best
model checkpoint is written to disk.

Expected data layout
- --img_path: directory containing per-volume subfolders with image slices/files
- --mask_path: directory containing matching subfolders with mask slices/files
  Example:
    <img_path>/01/  and  <mask_path>/01/
    <img_path>/02/  and  <mask_path>/02/
  Each subfolder can contain a stack of .tif/.tiff/.nii.gz, etc. The reader
  assembles them into a single volume for patch extraction.

Model and pipeline
- Model: set by `MODEL` (defaults to UNet2D; use UNet3D for 3D)
- Transforms: basic intensity normalization (+ optional commented augmentations)
- Patching: `--training_patch_size z y x`, `--training_overlay z y x`,
            `--training_resize_factor z y x`
- Metrics: loss (Dice+BCE) and soft Dice score are logged; curves saved as PNG.
- Checkpoint: best validation loss saved to `<save_path>/<model_name>.pth`.

Usage 3D:
python train.py \
  --img_path ./datas/c-Fos/YYC/training-data/LI-AN-3D/images \
  --mask_path ./datas/c-Fos/YYC/training-data/LI-AN-3D/masks \
  --save_path ./datas/c-Fos/YYC/weights \
  --model_name c-Fos \
  --training_epochs 100 \
  --training_batch_size 8 \
  --training_patch_size 16 64 64 \
  --training_overlay 0 0 0 \
  --training_resize_factor 1 1 1 \
  --visualize_preview
  
Usage 2D:
python train.py ^
  --img_path ./datas/c-Fos/LI-WIN_PAPER/training-data/100-3D/images ^
  --mask_path ./datas/c-Fos/LI-WIN_PAPER/training-data/100-3D/masks ^
  --save_path ./datas/c-Fos/LI-WIN_PAPER/weights ^
  --model_name fun-3 ^
  --training_epochs 100 ^
  --training_batch_size 48 ^
  --training_patch_size 1 64 64 ^
  --training_overlay 0 8 8 ^
  --training_resize_factor 1 1 1 ^
  --visualize_preview
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import sys
import torch

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.spatial.dictionary import RandFlipd, RandZoomd, RandAffined
from monai.transforms.intensity.dictionary import (
    ScaleIntensityRanged, GaussianSmoothd, NormalizeIntensityd, 
    RandAdjustContrastd, RandBiasFieldd, RandShiftIntensityd, RandScaleIntensityd
)
from monai.transforms.post.dictionary import AsDiscreted
from monai.data.dataloader import DataLoader

from train.trainer import Trainer
from models.UNet_2D_V2 import UNet2D
from models.UNet_3D_V1 import UNet3D
from utils.datasets import MicroscopyDataset
from utils.loader import load_train_data
from utils.visualization import visualize_dataset

train_transform = Compose([
    # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True , channel_wise=True),
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
    # RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    # RandAffined(keys=["image", "mask"], prob=0.5, rotate_range=(1.57, 1.57, 0.1)),
    # RandAdjustContrastd(keys=["image"], prob=0.3),
    # RandBiasFieldd(keys=["image"], prob=0.2),
    # RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.3),
    # RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
    # RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
    GaussianSmoothd(keys=["mask"], sigma=1.0),
    AsDiscreted(keys=["mask"], threshold= 0.5)
])

val_transform = Compose([
    # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True , channel_wise=True),
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
    # RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    # RandAffined(keys=["image", "mask"], prob=0.5, rotate_range=(1.57, 1.57, 0.1)),
    # RandAdjustContrastd(keys=["image"], prob=0.3),
    # RandBiasFieldd(keys=["image"], prob=0.2),
    # RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.3),
    # RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
    # RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
    GaussianSmoothd(keys=["mask"], sigma=1.0),
    AsDiscreted(keys=["mask"], threshold= 0.5)
])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 3D or 2D U-Net for Microscopy Segmentation"
    )
    
    parser.add_argument(
        "--img_path", type=str, required=True,
        help="Path to image data"
    )
    parser.add_argument(
        "--mask_path", type=str, required=True,
        help="Path to mask data"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--model_name", type=str, default="best_model", 
        help="Filename for the saved model"
    )
    
    parser.add_argument(
        "--training_input_channel", type=int, default=1,
        help="Number of channels in the input image (e.g., 1 for grayscale, 3 for RGB)."
    )
    parser.add_argument(
        "--training_output_channel", type=int, default=1,
        help="Number of channels in the model output (e.g., 1 for binary segmentation, >1 for multi-class)."
    )
    parser.add_argument(
        "--training_epochs", type=int, default=30, 
        help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--training_batch_size", type=int, default=8, 
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--training_patch_size", type=int, default=[1, 64, 64], nargs=3, 
        help="Patch size (z, y, x) for model training. The image is processed in patches of this size."
    )
    parser.add_argument(
        "--training_overlay", type=int, default=[0, 0, 0], nargs=3,
        help="Number of voxels to overlap (z, y, x) between adjacent patches during inference to avoid edge artifacts."
    )
    parser.add_argument(
        "--training_resize_factor", type=float, default=[1, 1, 1], nargs=3,
        help="Scaling factor (z, y, x) to resize the input image before training. Use [1,1,1] for no resizing."
    )
    parser.add_argument(
        "--visualize_preview", action="store_true",
        help="If set, saves preview grids of train/val samples after transforms."
    )
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    img_root = args.img_path
    mask_root = args.mask_path
    os.makedirs(args.save_path, exist_ok=True)

    logging.info("Loading image and mask data...")
    
    training_patch_size = tuple(args.training_patch_size)
    training_overlay = tuple(args.training_overlay)
    training_resize_factor = tuple(args.training_resize_factor)

    # Auto-select 2D vs 3D model and dataset based on Z depth
    if training_patch_size[0] > 1:
        SelectedModel = UNet3D
        spatial_dims = 3
        logging.info("Auto-selected 3D model/dataset (patch depth > 1)")
    else:
        SelectedModel = UNet2D
        spatial_dims = 2
        logging.info("Auto-selected 2D model/dataset (patch depth == 1)")
        
    # Load and split patches in one step
    train_patches, val_patches = load_train_data(
        img_path=img_root,
        mask_path=mask_root,
        patch_size=tuple(training_patch_size),
        overlay=tuple(training_overlay),
        resize_factor=tuple(training_resize_factor),
        balance=True,
        val_ratio=0.3,
        seed=100,
    )
    
    train_dataset = MicroscopyDataset(train_patches, transform=train_transform, spatial_dims=spatial_dims, with_mask=True)
    val_dataset = MicroscopyDataset(val_patches, transform=val_transform, spatial_dims=spatial_dims, with_mask=True)

    # Optional preview of transformed samples
    if args.visualize_preview:
        try:
            visualize_dataset(train_dataset, title="Train samples")
        except Exception as e:
            logging.warning("Failed to visualize train preview: %s", str(e))
        try:
            visualize_dataset(val_dataset, title="Val samples")
        except Exception as e:
            logging.warning("Failed to visualize val preview: %s", str(e))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.training_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.training_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    model = SelectedModel(in_channels=args.training_input_channel, out_channels=args.training_output_channel)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader, 
        val_loader=val_loader, 
        save_path=args.save_path,
        model_name=args.model_name,
    )

    logging.info("Starting training...")

    trainer.train(epochs=args.training_epochs)

    logging.info("Training completed. Model saved to %s", args.save_path)

if __name__ == "__main__":
    sys.exit(main())
