"""
Batch inference (2D or 3D) over an images directory tree using the same
patching/stitching pipeline as inference.py.

Usage 3D:
    python test.py \
    --img_path ./datas/c-Fos/LI-WIN_PAPER/testing-data/V60 \
    --model_path ./datas/c-Fos/LI-WIN_PAPER/weights/fun-3.pth \
    --inference_patch_size 16 64 64 \
    --inference_overlay 2 4 4 \
    --output_type scroll-tiff

Usage 2D:
  python test.py ^
    --input_dir ./datas/c-Fos/LI-WIN_PAPER/testing-data/V60 ^
    --model_path ./datas/c-Fos/LI-WIN_PAPER/weights/func-3.pth ^
    --inference_patch_size 1 64 64 ^
    --inference_overlay 0 16 16 ^
    --output_type scroll-tiff

Input layout (auto-discovered subfolders under images/):
  ./datas/V60/images/<any_subfolder>/*.{tif,tiff,nii.gz,...}

Saves outputs to:
  ./datas/V60/masks_{model_stem}/<same_subfolder>/<same_filenames_or_format>
If output_type=scroll-tiff, files are moved up so they land directly under
the subfolder (e.g., .../left/raw0000.tiff).
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.intensity.dictionary import NormalizeIntensityd

from IO.reader import FileReader
from IO.writer import FileWriter
from inference.inferencer import Inferencer
from utils.datasets import MicroscopyDataset
from utils.stitcher import stitch_image
from utils.loader import load_model, compute_z_plan, load_inference_data


inference_transform = Compose([
    ToTensord(keys=["image"], dtype=torch.float32),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
])


def list_subfolders(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_dir()]


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
                logging.warning(f"Could not move {f} to {target}: {e}")
    try:
        scroll_dir.rmdir()
    except OSError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference over images/<subfolder> using inference.py pipeline")
    p.add_argument("--input_dir", type=str, required=True, help="Base dataset dir containing images/<subfolder>")
    p.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth/.pt")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    # Inference params
    p.add_argument("--inference_patch_size", type=int, nargs=3, default=[1, 64, 64])
    p.add_argument("--inference_overlay", type=int, nargs=3, default=[0, 16, 16])
    p.add_argument("--inference_resize_factor", type=float, nargs=3, default=[1, 1, 1])
    p.add_argument("--inference_resize_order", type=int, default=0)
    # Output params mirroring inference.py
    p.add_argument("--output_type", type=str, default='scroll-tiff', choices=['single-tiff', 'scroll-tiff', 'single-nii', 'scroll-nii'])
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

    logging.info(f"Loading model: {args.model_path}")
    model = load_model(args.model_path)
    inferencer = Inferencer(model)

    inference_patch_size = tuple(args.inference_patch_size)
    inference_overlay = tuple(args.inference_overlay)
    inference_resize_factor = tuple(args.inference_resize_factor)

    # Choose spatial dims based on depth
    spatial_dims = 3 if inference_patch_size[0] > 1 else 2

    # Auto-discover subfolders under images/
    if not images_root.exists():
        logging.error(f"Images root does not exist: {images_root}")
        return 1
    subfolders = list_subfolders(images_root)
    if not subfolders:
        logging.error(f"No subfolders found in {images_root}")
        return 1
    logging.info(f"Auto-discovered subfolders: {[p.name for p in subfolders]}")

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
        z_plan = compute_z_plan(data_reader.volume_shape[0], inference_patch_size[0], inference_overlay[0])

        for z_start, z_overlay in z_plan:
            inference_patches, data_position = load_inference_data(
                data_reader=data_reader,
                z_start=z_start,
                patch_size=inference_patch_size, 
                overlay=inference_overlay,
                resize_factor=inference_resize_factor,
            )

            inference_dataset = MicroscopyDataset(
                inference_patches,
                transform=inference_transform,
                spatial_dims=spatial_dims,
                with_mask=False,
            )
            inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            mask_patches = inferencer.eval(inference_loader)

            stitched_volume, prev_z_slices = stitch_image(
                patches=mask_patches,
                positions=data_position,
                original_shape=(inference_patch_size[0], data_reader.volume_shape[1], data_reader.volume_shape[2]),
                patch_size=inference_patch_size,
                z_overlay=z_overlay,
                prev_z_slices=prev_z_slices,
                resize_factor=inference_resize_factor,
            )

            data_writer.write(stitched_volume, z_start=z_start, z_end=z_start + stitched_volume.shape[0])

        if args.output_type in ["scroll-tiff", 'scroll-nii']:
            move_scroll_results_up(Path(mask_path), data_reader.volume_name)

    logging.info(f"Finish results under: {masks_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
