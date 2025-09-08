"""
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
  --training_resize_factor 1 1 1 
  
Usage 2D:
python train.py ^
  --img_path ./datas/c-Fos/LI-WIN_PAPER/training-data/LI-AN-3D/images ^
  --mask_path ./datas/c-Fos/LI-WIN_PAPER/training-data/LI-AN-3D/masks ^
  --save_path ./datas/c-Fos/LI-WIN_PAPER/weights ^
  --model_name function-3 ^
  --training_epochs 100 ^
  --training_batch_size 32 ^
  --training_patch_size 1 64 64 ^
  --training_overlay 0 4 4 ^
  --training_resize_factor 1 1 1 
"""

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.spatial.dictionary import RandFlipd, RandZoomd, RandAffined
from monai.transforms.intensity.dictionary import (
    ScaleIntensityRanged, NormalizeIntensityd, 
    RandAdjustContrastd, RandBiasFieldd, RandShiftIntensityd, RandScaleIntensityd, 
    GaussianSmoothd
)
from monai.transforms.post.dictionary import AsDiscreted
from monai.data.dataloader import DataLoader

from train.trainer import Trainer
from utils.reader import FileReader
from utils.cropper import extract_training_batches

# Model Choose
from models.UNet_2D_V2 import UNet2D
from models.UNet_3D_V1 import UNet3D

MODEL = UNet2D

# Dataset Choose
from train.loader import MicroscopyDataset3D, MicroscopyDataset2D

DATASET = MicroscopyDataset2D

# Transform split: preproc (deterministic), aug (random), post (finalize)
preproc_transform = Compose([
    # GaussianSmoothd(keys=["image"], sigma=1.0),
    # ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True , channel_wise=True),
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
])

aug_train_transform = Compose([
    RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    RandAffined(keys=["image", "mask"], prob=0.5, rotate_range=(1.57, 1.57, 0.1)),
    # RandAdjustContrastd(keys=["image"], prob=0.3),
    # RandBiasFieldd(keys=["image"], prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.3),
    RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
    # RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
])

aug_valid_transform = Compose([
    RandAdjustContrastd(keys=["image"], prob=0.3),
    RandBiasFieldd(keys=["image"], prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.3),
    RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
    # RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
])

post_transform = Compose([
    AsDiscreted(keys=["mask"], threshold=0.5),
])

# Compose final transforms
train_transform = Compose([preproc_transform, aug_train_transform, post_transform])
val_transform = Compose([preproc_transform, aug_valid_transform,  post_transform])

# Load data test and mess with how to preprocess image and mask
def load_data(img_path, mask_path, patch_size, overlay, resize_factor):
    """
    Loads and preprocesses 3D image and mask volumes from the specified directories, 
    applies Gaussian smoothing and binarization to the masks, extracts 2D patches, 
    and returns them for training.

    Parameters:
    ----------
    img_path : str
        Path to the folder containing subfolders of image volumes.
        
    mask_path : str
        Path to the folder containing subfolders of corresponding mask volumes.

    patch_size : tuple of int
        The size of the extracted 2D patches, e.g., (256, 256).

    overlay : int
        The number of pixels by which patches should overlap during extraction.

    resize_factor : float
        Factor by which to downsample the image and mask before patch extraction.

    Returns:
    -------
    image_patches : list of np.ndarray
        A list of extracted image patches.

    mask_patches : list of np.ndarray
        A list of corresponding mask patches, preprocessed (binarized and smoothed).
    """
    volume_dirs = sorted([name for name in os.listdir(img_path)])
    
    image_patches = []
    mask_patches = []

    for folder_name in volume_dirs:
        img_folder_path = os.path.join(img_path, folder_name)
        mask_folder_path = os.path.join(mask_path, folder_name)
        
        img_reader = FileReader(img_folder_path)
        img_volume = img_reader.read(z_start=0, z_end=img_reader.volume_shape[0]).astype(np.float32)

        mask_reader = FileReader(mask_folder_path)
        mask_volume = mask_reader.read(z_start=0, z_end=mask_reader.volume_shape[0]).astype(np.float32)
        
        img_p, mask_p = extract_training_batches(
            image=img_volume, mask=mask_volume, 
            patch_size=patch_size, overlay=overlay, 
            resize_factor=resize_factor, balance=True
        )
        
        image_patches.extend(img_p)
        mask_patches.extend(mask_p)
        
    return image_patches, mask_patches


def main():
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net for Microscopy Segmentation"
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
    
    args = parser.parse_args()
    
    img_root = args.img_path
    mask_root = args.mask_path
    os.makedirs(args.save_path, exist_ok=True)

    logging.info("Loading image and mask data...")
    
    training_patch_size = tuple(args.training_patch_size)
    training_overlay = tuple(args.training_overlay)
    training_resize_factor = tuple(args.training_resize_factor)
    
    image_patches, mask_patches = load_data(img_root, mask_root, training_patch_size, training_overlay, training_resize_factor)
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_patches, mask_patches, test_size=0.3, random_state=100
    )
    
    train_patches = []
    for img, mask in zip(train_imgs, train_masks):
        patch = {"image": img, "mask": mask}
        train_patches.append(patch)
        
    val_patches = []
    for img, mask in zip(val_imgs, val_masks):
        patch = {"image": img, "mask": mask}
        val_patches.append(patch)
    
    train_dataset = DATASET(train_patches, transform=train_transform)
    val_dataset = DATASET(val_patches, transform=val_transform)
    
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
    
    model = MODEL(in_channels=args.training_input_channel, out_channels=args.training_output_channel)
    
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

    trainer.save_figure(metric_name="loss")

    logging.info("All figure saved to %s", args.save_path)

if __name__ == "__main__":
    sys.exit(main())