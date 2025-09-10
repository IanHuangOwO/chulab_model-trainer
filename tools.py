"""
Shared inference utilities reused by inference.py and test.py.
"""

import os
import torch
from typing import List, Tuple

from monai.transforms import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.intensity.dictionary import NormalizeIntensityd

from utils.cropper import extract_patches


# Define transforms for inference (shared)
inference_transform = Compose([
    ToTensord(keys=["image"], dtype=torch.float32),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
])


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False)
    return model


def load_data(
    data_reader,
    z_start: int,
    patch_size: Tuple[int, int, int],
    overlay: Tuple[int, int, int],
    resize_factor: Tuple[float, float, float],
):
    data_volume = data_reader.read(z_start=z_start, z_end=z_start + patch_size[0])
    data_patches, data_position = extract_patches(
        array=data_volume,
        patch_size=patch_size,
        overlay=overlay,
        resize_factor=resize_factor,
        return_positions=True,
    )
    inference_patches = [{"image": img} for img in data_patches]
    return inference_patches, data_position


def compute_z_plan(volume_depth: int, patch_depth: int, z_overlap: int) -> List[Tuple[int, int]]:
    assert patch_depth > 0 and z_overlap >= 0
    assert patch_depth > z_overlap

    step = patch_depth - z_overlap
    patches: List[Tuple[int, int]] = []
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

