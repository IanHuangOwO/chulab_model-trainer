"""
Single-volume inference with patch-based tiling and stitching.

Runs a trained model on a single 2D/3D volume by splitting the input into
overlapping patches, predicting per patch, and stitching back to the original
shape. Dimensionality is inferred from the z-size of `--inference_patch_size`
(z > 1 → 3D, z == 1 → 2D).

Parameters (CLI)
- --input_path: Path to an image folder or a single volume readable by FileReader.
- --output_path: Directory where predicted masks are written.
- --model_path: Path to a saved torch model file (.pth/.pt).
- --inference_patch_size: Patch size as three integers `z y x`.
- --inference_overlay: Overlap between adjacent patches `z y x`.
- --inference_resize_factor: Optional resize factors `z y x` applied before inference.
- --output_type: Output format, one of OUTPUT_CHOICES (e.g., 'scroll-tiff', 'zarr', 'ome-zarr').
- --output_dtype / --output_chunk_size / --output_n_level / --output_resize_*: Writer options for multi-resolution outputs.

Examples (3D)
  python inference.py \
    --input_path ./datas/c-Fos/YYC/testing-data/YYC_20230414-1/images \
    --output_path ./datas/c-Fos/YYC/testing-data/YYC_20230414-1/results \
    --model_path ./datas/c-Fos/YYC/weights/c-Fos_200_LI_AN.pth \
    --output_type scroll-tiff \
    --inference_patch_size 16 64 64 \
    --inference_overlay 2 4 4 \
    --inference_resize_factor 1 1 1

Examples (2D, Windows caret)
  python inference.py ^
    --input_path ./datas/c-Fos/LI-WIN_PAPER/testing-data/V60/images/left ^
    --output_path ./datas/c-Fos/LI-WIN_PAPER/testing-data/V60 ^
    --model_path ./datas/c-Fos/LI-WIN_PAPER/weights/func-3_LI-AN-32.pth ^
    --output_type ome-zarr ^
    --inference_patch_size 1 32 32 ^
    --inference_overlay 0 16 16 ^
    --inference_resize_factor 1 1 1
"""
# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import sys

import torch
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.intensity.dictionary import ScaleIntensityRanged, NormalizeIntensityd

from IO import FileReader, FileWriter, OUTPUT_CHOICES, TYPE_MAP
from inference.inferencer import Inferencer
from utils.loader import load_model, load_inference_data, compute_z_plan
from utils.stitcher import stitch_image
from utils.datasets import MicroscopyDataset


inference_transform = Compose([
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ToTensord(keys=["image"], dtype=torch.float32),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Mask Inference: Applies a trained model to infer masks from 3D images and outputs multi-resolution results."
    )

    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input 3D image file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to the output mask file (where the inferred mask will be saved)."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the trained 3D segmentation model (.pt or similar)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--num_workers", type=int, default=8
    )

    parser.add_argument(
        "--inference_patch_size", type=int, nargs=3, default=[16, 64, 64],
        help="Patch size (z, y, x) for model inference. The image is processed in patches of this size."
    )
    parser.add_argument(
        "--inference_overlay", type=int, nargs=3, default=[2, 4, 4],
        help="Number of voxels to overlap (z, y, x) between adjacent patches during inference to avoid edge artifacts."
    )
    parser.add_argument(
        "--inference_resize_factor", type=float, nargs=3, default=[1, 1, 1],
        help="Scaling factor (z, y, x) to resize the input image before inference. Use [1,1,1] for no resizing."
    )
    parser.add_argument(
        "--inference_resize_order", type=int, default=0,
        help="Interpolation order for resizing before inference (0=nearest, 1=linear, 3=cubic)."
    )

    parser.add_argument(
        "--output_type", type=str, default='scroll-tiff', choices=OUTPUT_CHOICES,
        help="Output format type. ex: 'scroll-tiff', 'zarr', 'ome-zarr' or other supported formats."
    )
    parser.add_argument(
        "--output_dtype", type=str, default='uint16',
        help="Data type of the output mask (e.g., 'uint8', 'uint16', 'float32')."
    )
    parser.add_argument(
        "--output_chunk_size", type=int, nargs=3, default=[128, 128, 128],
        help="Chunk size (z, y, x) for saving the output file. Useful for formats like Zarr for better I/O performance."
    )
    parser.add_argument(
        "--output_resize_factor", type=int, default=2,
        help="Downsampling factor between pyramid levels in the output (used in multi-resolution formats)."
    )
    parser.add_argument(
        "--output_resize_order", type=int, default=0,
        choices=[0, 1, 3],
        help="Interpolation order for resizing output levels (0=nearest, 1=linear, 3=cubic)."
    )
    parser.add_argument(
        "--output_n_level", type=int, default=5,
        help="Number of levels in the output pyramid (e.g., 5 = base level + 4 downsampled levels)."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    os.makedirs(output_path, exist_ok=True)
    
    logging.info("Loading model from: %s", model_path)
    
    model = load_model(model_path)
    inferencer = Inferencer(model)
    
    inference_patch_size = tuple(args.inference_patch_size)
    inference_overlay = tuple(args.inference_overlay)
    inference_resize_factor = tuple(args.inference_resize_factor)
    
    # Choose spatial dims based on depth
    spatial_dims = 3 if inference_patch_size[0] > 1 else 2
    
    prev_z_slices = None
    
    logging.info(f"Reading input image from: {input_path}")
    
    data_reader = FileReader(input_path)
    output_type = TYPE_MAP.get(args.output_type)
    data_writer = FileWriter(
        output_path=output_path,
        output_name=data_reader.volume_name, 
        output_type=output_type,
        output_dtype=args.output_dtype,
        full_res_shape=data_reader.volume_shape,
        file_name=data_reader.volume_files,
        chunk_size=args.output_chunk_size,
        resize_factor=args.output_resize_factor,
        resize_order=args.output_resize_order,
        n_level=args.output_n_level,
    )
    
    z_plan = compute_z_plan(data_reader.volume_shape[0], inference_patch_size[0], inference_overlay[0])
    
    for z_start, z_overlay in z_plan:
        inference_patches, data_position = load_inference_data(
            data_reader=data_reader, 
            z_start=z_start,
            patch_size=inference_patch_size,
            overlay=inference_overlay,
            resize_factor=inference_resize_factor
        )
        
        inference_dataset = MicroscopyDataset(
            inference_patches,
            transform=inference_transform,
            spatial_dims=spatial_dims,
            with_mask=False,
        )
        inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        logging.info(f"Inferencing ...")
        mask_patches = inferencer.eval(inference_loader)
        
        logging.info(f"Stitching ...")
        stitched_volume, prev_z_slices = stitch_image(
            patches=mask_patches, 
            positions=data_position,
            original_shape=(inference_patch_size[0], data_reader.volume_shape[1], data_reader.volume_shape[2]),
            patch_size=inference_patch_size,
            z_overlay=z_overlay,
            prev_z_slices=prev_z_slices,
            resize_factor=inference_resize_factor,
        )
        
        data_writer.write(stitched_volume, z_start=z_start, z_end=z_start+stitched_volume.shape[0])

    if output_type == "ome-zarr":
        data_writer.complete_ome()
        
    logging.info(f"Inference complete. Output saved to {output_path}")
        
if __name__ == "__main__":
    sys.exit(main())
