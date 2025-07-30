"""
Usage:
python train.py \
  --img_path ./datas/TH/YYC_20230922/training_data/raw_data \
  --mask_path ./datas/TH/YYC_20230922/training_data/raw_mask \
  --save_path ./datas/TH/YYC_20230922/weights \
  --model_name contrast_bias_shift_scale.pth \
  --epochs 30
  --batch_size 8
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
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

val_transform = Com