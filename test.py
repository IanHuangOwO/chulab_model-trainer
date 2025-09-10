"""
Batch inference (2D or 3D) over an images directory tree using the same
patching/stitching pipeline as inference.py.

Example:
  python test.py \
    --input_dir ./datas/V60 \
    --model_path ./datas/c-Fos/LI-WIN_PAPER/weights/NA.pth \
    --inference_patch_size 1 64 64 \
    --inference_overlay 0 16 16 \
    --output_type scroll-tiff

Input layout (auto-discovered subfolders under images/):
  ./datas/V60/images/<any_subfolder>/*.{tif,tiff,nii.gz,...}

Saves outputs to:
  ./datas/V60/masks_{model_stem}/<same_subfolder>/<same_filenames_or_format>
If output_type=scroll-tiff, files are moved up so they land directly under
the subfolder (e.g., .../left/raw0000.tiff).
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from torch.utils.data import DataLoader

from inference.inferencer import Inferencer
from inference.loader import MicroscopyDataset2D, MicroscopyDataset3D
from utils.reader import FileReader
from utils.writer import FileWriter
from utils.cropper import extract_patches
from utils.stitcher import stitch_image


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_inference")


def load_model(model_path: str) -> torch.nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    # Expect a full torch module saved with torch.save
    model = torch.load(model_path, weights_only=False)
    return model


def make_transform():
    return Compose([
        ToTensord(keys=["image"], dtype=torch.float32),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])


def list_subfolders(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_dir()]


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


def load_data(
    data_reader: FileReader,
    z_start: int,
    patch_size: Tuple[int, int, int],
    overlay: Tuple[int, int, int],
    resize_factor: Tuple[float, float, float],
):
    data_volume = data_reader.read(z_start=z_start, z_end=z_start + patch_size[0])
    data_patches, data_position = extract_patches(
        array=data_volume, patch_size=patch_size, overlay=overlay,
        resize_factor=resize_factor, return_positions=True
    )
    inference_patches = [{"image": img} for img in data_patches]
    return inference_patches, data_position


def move_scroll_results_up(mask_dir: Path, volume_name: str) -> None:
    """If writer created '<mask_dir>/<volume_name>_scroll', move files up to '<mask_dir>' and remove the folder."""
    scroll_dir = mask_dir / f"{volume_name}_scroll"
    if not scroll_dir.exists() or not scroll_dir.is_dir():
        return
    for f in scroll_dir.iterdir():
        if f.is_file():
            target = mask_dir / f.name
            try:
                if target.exists():
                    target.unlink()
                f.replace(target)
            except Exception as e:
                logger.warning(f"Could not move {f} to {target}: {e}")
    try:
        scroll_dir.rmdir()
    except OSError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference over images/<subfolder> using inference.py pipeline")
    p.add_argument("--input_dir", type=str, required=True, help="Base dataset dir containing images/<subfolder>")
    p.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth/.pt")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    # Inference params
    p.add_argument("--inference_patch_size", type=int, nargs=3, default=[1, 64, 64])
    p.add_argument("--inference_overlay", type=int, nargs=3, default=[0, 16, 16])
    p.add_argument("--inference_resize_factor", type=float, nargs=3, default=[1, 1, 1])
    p.add_argument("--inference_resize_order", type=int, default=0)
    # Output params mirroring inference.py
    p.add_argument("--output_type", type=str, default='scroll-tiff', choices=['zarr', 'ome-zarr', 'single-tiff', 'scroll-tiff', 'single-nii', 'scroll-nii'])
    p.add_argument("--output_dtype", type=str, default='uint16')
    p.add_argument("--output_chunk_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--output_resize_factor", type=int, default=2)
    p.add_argument("--output_resize_order", type=int, default=0, choices=[0, 1, 3])
    p.add_argument("--output_n_level", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    base = Path(args.input_dir)
    model_stem = Path(args.model_path).stem
    images_root = base / "images"
    masks_root = base / f"masks_{model_stem}"

    logger.info(f"Loading model: {args.model_path}")
    model = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferencer = Inferencer(model, device=device)

    patch_size = tuple(int(x) for x in args.inference_patch_size)
    overlay = tuple(int(x) for x in args.inference_overlay)
    resize_factor = tuple(float(x) for x in args.inference_resize_factor)

    # Choose dataset class based on depth
    DatasetCls = MicroscopyDataset3D if patch_size[0] > 1 else MicroscopyDataset2D

    # Auto-discover subfolders under images/
    if not images_root.exists():
        logger.error(f"Images root does not exist: {images_root}")
        return 1
    subfolders = list_subfolders(images_root)
    if not subfolders:
        logger.error(f"No subfolders found in {images_root}")
        return 1
    logger.info(f"Auto-discovered subfolders: {[p.name for p in subfolders]}")

    for sub in subfolders:
        img_path = str(sub)
        mask_path = str(masks_root / sub.name)
        os.makedirs(mask_path, exist_ok=True)

        logging.info(f"Reading input image from: {img_path}")
        data_reader = FileReader(img_path)
        data_writer = FileWriter(
            output_path=mask_path,
            output_name=data_reader.volume_name,
            output_type=args.output_type,
            output_dtype=args.output_dtype,
            full_res_shape=data_reader.volume_shape,
            file_name=data_reader.volume_files,
            chunk_size=tuple(args.output_chunk_size),
            resize_factor=args.output_resize_factor,
            resize_order=args.output_resize_order,
            n_level=args.output_n_level,
        )

        logging.info("Inferencing ...")
        prev_z_slices = None
        z_plan = compute_z_plan(data_reader.volume_shape[0], patch_size[0], overlay[0])

        for z_start, z_overlay in z_plan:
            inference_patches, data_position = load_data(
                data_reader=data_reader,
                z_start=z_start,
                patch_size=patch_size,
                overlay=overlay,
                resize_factor=resize_factor,
            )

            inference_dataset = DatasetCls(inference_patches, transform=make_transform())
            inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            mask_patches = inferencer.eval(inference_loader)

            stitched_volume, prev_z_slices = stitch_image(
                patches=mask_patches,
                positions=data_position,
                original_shape=(patch_size[0], data_reader.volume_shape[1], data_reader.volume_shape[2]),
                patch_size=patch_size,
                z_overlay=z_overlay,
                prev_z_slices=prev_z_slices,
                resize_factor=resize_factor,
            )

            data_writer.write(stitched_volume, z_start=z_start, z_end=z_start + stitched_volume.shape[0])

        if args.output_type == "ome-zarr":
            data_writer.write_ome_levels()

        if args.output_type == "scroll-tiff":
            move_scroll_results_up(Path(mask_path), data_reader.volume_name)

    logger.info(f"Done. Results under: {masks_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
