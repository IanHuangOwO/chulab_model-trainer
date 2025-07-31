"""
Usage:
python train.py \
  --img_path ./datas/TH/YYC_20230922/training_data/raw_data \
  --mask_path ./datas/TH/YYC_20230922/training_data/raw_mask \
  --save_path ./datas/TH/YYC_20230922/weights \
  --model_name resize \
  --training_epochs 10 \
  --training_batch_size 8 \
  --training_patch_size 16 128 128 \
  --training_overlay 2 4 4 \
  --training_resize_factor 1 0.5 0.5 
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
from skimage.filters import gaussian
from sklearn.model_selection import train_test_split

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.spatial.dictionary import RandFlipd, RandZoomd, RandAffined
from monai.transforms.intensity.dictionary import ScaleIntensityRanged, RandAdjustContrastd, RandBiasFieldd, RandShiftIntensityd, RandScaleIntensityd
from monai.transforms.post.dictionary import AsDiscreted
from monai.data.dataloader import DataLoader

from train.loader import MicroscopyDataset3D
from train.trainer import Trainer
from utils.reader import FileReader
from utils.cropper import extract_training_batches

from models.UNet_3D import UNet3D

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dict-based transform
train_transform = Compose([
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    RandAffined(keys=["image", "mask"], prob=0.5, rotate_range=(0, 0, 0.1), scale_range=(0.9, 1.1, 1)),
    RandAdjustContrastd(keys=["image"], prob=0.3),
    RandBiasFieldd(keys=["image"], prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
    AsDiscreted(keys=["mask"], threshold=1),
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
])

val_transform = Compose([
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    RandAffined(keys=["image", "mask"], prob=0.5, rotate_range=(0, 0, 0.1), scale_range=(0.9, 1.1, 1)),
    RandAdjustContrastd(keys=["image"], prob=0.3),
    RandBiasFieldd(keys=["image"], prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
    AsDiscreted(keys=["mask"], threshold=1),
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
])

def load_data(img_path, mask_path, patch_size, overlay, resize_factor):
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
        
        mask_volume[mask_volume > 0] = 1
        mask_volume = gaussian(mask_volume, sigma=1.2)
        mask_volume[mask_volume > 0.5] = 1
        
        img_p, mask_p = extract_training_batches(
            image=img_volume, 
            mask=mask_volume, 
            patch_size=patch_size, 
            overlay=overlay,
            resize_factor=resize_factor,
            balance=True
        )
        
        image_patches.extend(img_p)
        mask_patches.extend(mask_p)
        
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_patches, mask_patches, test_size=0.3, random_state=42
    )
    
    train_patches = []
    for img, mask in zip(train_imgs, train_masks):
        patch = {"image": img, "mask": mask}
        train_patches.append(patch)
        
    val_patches = []
    for img, mask in zip(val_imgs, val_masks):
        patch = {"image": img, "mask": mask}
        val_patches.append(patch)
        
    return train_patches, val_patches
    
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
        "--training_epochs", type=int, default=30, 
        help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--training_batch_size", type=int, default=8, 
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--training_patch_size", type=int, default=[16, 64, 64], nargs=3, 
        help="Patch size (z, y, x) for model training. The image is processed in patches of this size."
    )
    parser.add_argument(
        "--training_overlay", type=int, default=[2, 4, 4], nargs=3,
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
    
    train_patches, val_patches = load_data(img_root, mask_root, training_patch_size, training_overlay, training_resize_factor)
    
    train_dataset = MicroscopyDataset3D(train_patches, transform=train_transform)
    val_dataset = MicroscopyDataset3D(val_patches, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.training_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.training_batch_size, shuffle=True, num_workers=4)
    
    model = UNet3D(in_channels=1, out_channels=1)

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