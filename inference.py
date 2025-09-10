"""
Usage 3D:
python inference.py \
  --img_path ./datas/c-Fos/YYC/testing-data/YYC_20230414-1/images \
  --mask_path ./datas/c-Fos/YYC/testing-data/YYC_20230414-1/results \
  --model_path ./datas/c-Fos/YYC/weights/c-Fos_200_LI_AN.pth \
  --output_type scroll-tiff \
  --inference_patch_size 16 64 64 \
  --inference_overlay 2 4 4 \
  --inference_resize_factor 1 1 1 
  
Usage 2D:
python inference.py ^
  --img_path ./datas/c-Fos/LI-WIN_PAPER/testing-data/V45/down/raw ^
  --mask_path ./datas/c-Fos/LI-WIN_PAPER/testing-data/V45/down ^
  --model_path ./datas/c-Fos/LI-WIN_PAPER/weights/NA.pth ^
  --output_type scroll-tiff ^
  --inference_patch_size 1 64 64 ^
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

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.data.dataloader import DataLoader

from inference.inferencer import Inferencer
from utils.reader import FileReader
from utils.writer import FileWriter
from utils.cropper import extract_patches
from utils.stitcher import stitch_image

# Dataset Choose
from inference.loader import MicroscopyDataset3D, MicroscopyDataset2D
from tools import inference_transform, load_model, load_data, compute_z_plan

DATASET = MicroscopyDataset2D
    
def main():
    parser = argparse.ArgumentParser(
        description="3D Mask Inference: Applies a trained model to infer masks from 3D images and outputs multi-resolution results."
    )

    parser.add_argument(
        "--img_path", type=str, required=True,
        help="Path to the input 3D image file."
    )
    parser.add_argument(
        "--mask_path", type=str, required=True,
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
        "--output_type", type=str, default='scroll-tiff',
        choices=['zarr', 'ome-zarr', 'single-tiff', 'scroll-tiff', 'single-nii', 'scroll-nii'],
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

    args = parser.parse_args()
    
    img_path = args.img_path
    mask_path = args.mask_path
    model_path = args.model_path
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
        chunk_size=args.output_chunk_size,
        resize_factor=args.output_resize_factor,
        resize_order=args.output_resize_order,
        n_level=args.output_n_level,
    )
    
    logging.info("Loading model from: %s", model_path)
    
    model = load_model(model_path)
    inferencer = Inferencer(model)
    
    inference_patch_size = tuple(args.inference_patch_size)
    inference_overlay = tuple(args.inference_overlay)
    inference_resize_factor = tuple(args.inference_resize_factor)
    
    # Choose dataset class based on depth
    DatasetCls = MicroscopyDataset3D if inference_patch_size[0] > 1 else MicroscopyDataset2D
    
    prev_z_slices = None
    
    z_plan = compute_z_plan(data_reader.volume_shape[0], inference_patch_size[0], inference_overlay[0])
    
    for z_start, z_overlay in z_plan:
        inference_patches, data_position = load_data(
            data_reader=data_reader, 
            z_start=z_start,
            patch_size=inference_patch_size,
            overlay=inference_overlay,
            resize_factor=inference_resize_factor
        )
        
        inference_dataset = DatasetCls(inference_patches, transform=inference_transform)
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

    if args.output_type == "ome-zarr":
        data_writer.write_ome_levels()
        
    logging.info(f"Inference complete. Output saved to {mask_path}")
        
if __name__ == "__main__":
    sys.exit(main())
