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

DATASET = MicroscopyDataset2D

# Define transforms for inference
inference_transform = Compose([
    ToTensord(keys=["image"], dtype=torch.float32),
    NormalizeIntensityd(keys=["image"], nonzero=True , channel_wise=True),
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
        
        inference_dataset = DATASET(inference_patches, transform=inference_transform)
        inference_loader = DataLoader(inference_dataset, batch_size=8, shuffle=False, num_workers=4)
        
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