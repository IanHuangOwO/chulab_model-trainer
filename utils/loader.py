"""
Shared utilities for inference and training.
Includes transforms, model loader, and data tiling helpers.
"""

import os
from typing import List, Tuple

import numpy as np
import torch

from IO.reader import FileReader
from utils.cropper import extract_patches, extract_training_batches


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False)
    return model


def load_inference_data(
    data_reader,
    z_start: int,
    patch_size: Tuple[int, int, int],
    overlay: Tuple[int, int, int],
    resize_factor: Tuple[float, float, float],
):
    """Extract image-only patches for inference from a FileReader window."""
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


def load_train_data(
    img_path: str,
    mask_path: str,
    patch_size: Tuple[int, int, int],
    overlay: Tuple[int, int, int],
    resize_factor: Tuple[float, float, float],
    balance: bool = True,
    *,
    val_ratio: float = 0.3,
    seed: int | None = 42,
):
    """
    Extract paired (image, mask) patches across subfolders for training.

    If split is True, returns a train/val split. If return_dicts is True, each
    split is returned as a list of {"image": np.ndarray, "mask": np.ndarray}.
    Otherwise returns raw patch arrays.
    """
    volume_dirs = sorted([name for name in os.listdir(img_path)])
    image_patches: List[np.ndarray] = []
    mask_patches: List[np.ndarray] = []

    for folder_name in volume_dirs:
        img_folder_path = os.path.join(img_path, folder_name)
        mask_folder_path = os.path.join(mask_path, folder_name)

        img_reader = FileReader(img_folder_path)
        img_volume = img_reader.read(z_start=0, z_end=img_reader.volume_shape[0]).astype(np.float32)

        mask_reader = FileReader(mask_folder_path)
        mask_volume = mask_reader.read(z_start=0, z_end=mask_reader.volume_shape[0]).astype(np.float32)

        img_p, mask_p = extract_training_batches(
            image=img_volume,
            mask=mask_volume,
            patch_size=patch_size,
            overlay=overlay,
            resize_factor=resize_factor,
            balance=balance,
        )

        image_patches.extend(img_p)
        mask_patches.extend(mask_p)

    # train/val split using numpy (avoids sklearn dependency here)
    n = len(image_patches)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_n = int(n * val_ratio)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    def gather(indices):
        imgs = [image_patches[i] for i in indices]
        msks = [mask_patches[i] for i in indices]
        return [{"image": im, "mask": mk} for im, mk in zip(imgs, msks)]

    train_out = gather(train_idx)
    val_out = gather(val_idx)
    
    return train_out, val_out

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
