"""
Usage:
python inference.py \
  --img_path ./datas/TH/YYC_20230922/testing_data/raw_data \
  --mask_path ./datas/TH/YYC_20230922/testing_data/raw_mask \
  --model_path ./datas/TH/YYC_20230922/weights/contrast_bias_shift_scale.pth \
  --output_type scroll-tiff
"""

import argparse
import os
import sys
import logging
import torch

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.data.dataloader import DataLoader

from inference.loader import MicroscopyDataset3D
from inference.inferencer import Inferencer
from utils.reader import FileReader
from utils.writer import FileWriter
from utils.cropper import extract_patches
from utils.stitcher import stitch_image

from models.UNet_3D import UNet3D

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define transforms for inference
inference_transform = Compose([
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image"], dtype=torch.float32),
])

# Utility Function 
def load_model(model_path):
    """
    Load a PyTorch model from a given file path.

    Parameters:
    ----------
    model_path : str
        Path to the saved model file (.pt or .pth).

    Returns:
    -------
    model : torch.nn.Module
        Loaded PyTorch model object.

    Raises:
    ------
    FileNotFoundError:
        If the model file does not exist at the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = torch.load(model_path, weights_only=False)
    return model

def load_data(data_reader, z_start, patch_size, overlay, resize_factor):
    """
    Load a volume segment and extract inference-ready patches.

    Parameters:
    ----------
    data_reader : FileReader
        Object responsible for reading the image volume.
    z_start : int
        Starting z-index of the volume slice to load.
    patch_size : tuple of int
        Size of each patch (z, y, x) to extract from the volume.
    overlay : tuple of int
        Number of voxels to overlap between adjacent patches (z, y, x).
    resize_factor : tuple of float
        Scaling factor (z, y, x) to resize the data volume before patching.

    Returns:
    -------
    inference_patches : list of dict
        List of dictionaries containing the patch images under the key "image".
    data_position : list of tuple
        List of spatial positions corresponding to each patch.
    """
    data_volume = data_reader.read(z_start=z_start, z_end=z_start + patch_size[0])
        
    data_patches, data_position = extract_patches(array=data_volume, patch_size=patch_size, overlay=overlay, resize_factor=resize_factor, return_positions=True)
    
    inference_patches = [{"image": img} for img in data_patches]
    
    return inference_patches, data_position

def compute_z_plan(volume_depth, patch_depth, z_overlap):
    """
    Compute the list of z-slice starting positions for patch-wise inference.

    Parameters:
    ----------
    volume_depth : int
        Total depth (z-dimension) of the input volume.
    patch_depth : int
        Depth of each patch to extract.
    z_overlap : int
        Overlap in the z-dimension between consecutive patches.

    Returns:
    -------
    patches : list of tuple
        A list of tuples (z_start, z_overlay) indicating the starting z-slice 
        and the overlap for each patch. An overlay of -1 indicates the last patch.
    """
    assert patch_depth > 0 and z_overlap >= 0
    assert patch_depth > z_overlap

    step = patch_depth - z_overlap
    patches = []
    z = 0

    while z + patch_depth <= volume_depth:
        if z + patch_depth == volume_depth:
            patches.append((z, -1))
        else:
            patches.append((z, z_overlap))
            
        z += step
    
    if not patches or patches[-1][1] != -1:
        last_start = volume_depth - patch_depth
        if patches:
            patches[-1] = (patches[-1][0], patches[-1][0] + patch_depth - last_start)
            
        patches.append((last_start, -1))
        
    return patches
    
def main():
    parser = argparse.ArgumentParser(
        description="3D Mask Inference: Applies a trained model to infer masks from 3D images and outputs multi-resolution results."
    )

   