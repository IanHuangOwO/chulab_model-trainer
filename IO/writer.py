import logging
import zarr
import os
import numpy as np
import tifffile
import nibabel as nib
import dask.array as da

from pathlib import Path
from numcodecs import Blosc
from concurrent.futures import ProcessPoolExecutor
from skimage.transform import resize

# Set up module-level logger
logger = logging.getLogger(__name__)

# Define valid file suffixes
VALID_SUFFIXES = ['zarr', 'ome-zarr', 'single-tiff', 'scroll-tiff', 'single-nii', 'scroll-nii']

def _resize_xy_worker(args):
    """
    args = (slice_xy, target_y, target_x, dtype, order)
    """
    slice_xy, ty, tx, dt, ord = args
    return resize(
        slice_xy,
        (ty, tx),
        order=ord,
        preserve_range=True,
        anti_aliasing=False
    ).astype(dt)


def _resize_xz_worker(args):
    """
    args = (slice_xz, target_z, target_x, dtype, order)
    """
    slice_xz, tz, tx, dt, ord = args
    return resize(
        slice_xz,
        (tz, tx),
        order=ord,
        preserve_range=True,
        anti_aliasing=False
    ).astype(dt)


def two_pass_resize_zarr(
    output_source: zarr.Group,
    level: int,
    current_shape: tuple[int,int,int],
    target_shape: tuple[int,int,int],
    dtype: np.dtype,
    order: int = 1,
    chunk_size: tuple[int,int,int] = (128, 128, 128),
):
    """
    Resize a 3D Zarr array in two passes with an on-disk temp.

    Args:
      input_arr: zarr.Array, shape (Z, Y, X)
      output_arr: zarr.Array, pre-created with shape (target_z, target_y, target_x)
      temp_group: zarr.Group in which to create temp_key dataset
      temp_key: name of the temp dataset (e.g. "temp")
      target_shape: (target_z, target_y, target_x)
      dtype: output dtype
      order: interpolation order for skimage.resize
      chunk_size: slices (for XY pass) and rows (for XZ pass)
      memory_threshold_gb: unused here (always temp→zarr), but kept for signature
    """

    current_z, _, _ = current_shape
    target_z, target_y, target_x = target_shape
    
    temp_arr = output_source.create_dataset(
        "temp", shape=(current_z, target_y, target_x), 
        chunks=chunk_size,
        dtype=dtype, overwrite=True,
        compression=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    )
    
    # Pass 1: XY → temp_arr (unchanged)
    with ProcessPoolExecutor(max_workers=8) as exe:
        for z0 in range(0, current_z, chunk_size[0]):
            z1 = min(z0 + chunk_size[0], current_z)
            block = output_source[str(level - 1)][z0:z1]  # (dz, Y, X)
            args = [
                (block[i], target_y, target_x, dtype, order)
                for i in range(block.shape[0]) # type: ignore
            ]
            resized_slices = list(exe.map(_resize_xy_worker, args))
            arr = np.stack(resized_slices, axis=0)
            logger.info(f"Writing volume to temp z: {z0} - {z1}")
            
            arr_chunk = (z1 - z0, chunk_size[1], chunk_size[2])
            darr = da.from_array(arr, chunks=arr_chunk) # type: ignore
            darr.to_zarr(temp_arr, region=(
                slice(z0, z1),
                slice(0, temp_arr.shape[1]),
                slice(0, temp_arr.shape[2])
            ))

    # Pass 2: XZ → output_arr (now with threaded writes)
    with ProcessPoolExecutor(max_workers=8) as exe:
        for y0 in range(0, target_y, chunk_size[1]):
            y1 = min(y0 + chunk_size[1], target_y)
            block = temp_arr[:, y0:y1, :]  # (Z, dy, X)
            args = [
                (block[:, j, :], target_z, target_x, dtype, order)
                for j in range(block.shape[1])
            ]
            resized_slices = list(exe.map(_resize_xz_worker, args))
            arr = np.stack(resized_slices, axis=1)
            logger.info(f"Writing volume to zarr y: {y0} - {y1}")
            
            arr_chunk = (chunk_size[0], y1 - y0, chunk_size[2])
            darr = da.from_array(arr, chunks=arr_chunk) # type: ignore
            darr.to_zarr(output_source[str(level)], region=(
                slice(0, output_source[str(level)].shape[0]),
                slice(y0, y1),
                slice(0, output_source[str(level)].shape[2])
            ))

    # Clean up
    logger.info(f"Cleaning temp zarr")
    del output_source["temp"]

class FileWriter:
    def __init__(
        self, output_path, output_name, output_type, full_res_shape, output_dtype='uint16', 
        file_name=None, chunk_size=(128, 128, 128), n_level=5, resize_factor=2, resize_order=0
    ):
        self.output_path = Path(output_path)
        self.output_name = output_name
        self.output_type = output_type
        self.full_res_shape = full_res_shape
        self.output_dtype = output_dtype

        logger.info(f"Initialized FileWriter with output: {self.output_path}")

        self.initialize_output(
            file_name=file_name, 
            chunk_size=chunk_size, 
            n_level=n_level, 
            resize_factor=resize_factor, 
            resize_order=resize_order
        )
        
    def initialize_output(self, file_name=None, chunk_size=(128, 128, 128), n_level=5, resize_factor=2, resize_order=0):
        if self.output_type == "zarr":
            self._initialize_zarr(chunk_size=chunk_size)
        elif self.output_type == "ome-zarr":
            self._initialize_ome(chunk_size=chunk_size, n_level=n_level, resize_factor=resize_factor, resize_order=resize_order)
        elif self.output_type == "single-tiff":
            self._initialize_single_tiff()
        elif self.output_type == "scroll-tiff":
            if file_name is None:
                raise ValueError("file_name is required for scroll-tiff output.")
            self._initialize_scroll_tiff(file_name)
        elif self.output_type == "single-nii":
            self._initialize_single_nii()
        elif self.output_type == "scroll-nii":
            if file_name is None:
                raise ValueError("file_name is required for scroll-nii output.")
            self._initialize_scroll_nii(file_name)
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        
    def write(self, array: np.ndarray, z_start=0, z_end=None, y_start=0, y_end=None, x_start=0, x_end=None):
        z0, z1 = z_start, (self.full_res_shape[0] if z_end is None else z_end)
        y0, y1 = y_start, (self.full_res_shape[1] if y_end is None else y_end)
        x0, x1 = x_start, (self.full_res_shape[2] if x_end is None else x_end)

        logger.info(f"Write volume z: {z0} - {z1}, y: {y0} - {y1}, x: {x0} - {x1}")

        if self.output_type == "ome-zarr":
            darr = da.from_array(array.astype(self.output_dtype), chunks=self.chunk_size) # type: ignore
            darr.to_zarr(
                self.store_group['0'], 
                region=(slice(z0, z1),
                        slice(y0, y1),
                        slice(x0, x1))
            )

        elif self.output_type == "zarr":
            darr = da.from_array(array.astype(self.output_dtype), chunks=self.chunk_size) # type: ignore
            darr.to_zarr(
                self.store_array, 
                region=(slice(z0, z1),
                        slice(y0, y1),
                        slice(x0, x1))
            )

        elif self.output_type == "single-tiff":
            output_path = self.output_path / f"{self.output_name}_z{z0}-{z1}.tiff"
            tifffile.imwrite(output_path, array.astype(self.output_dtype), imagej=True)

        elif self.output_type == "scroll-tiff":
            for idx, file_path in enumerate(self.output_file_path[z0:z1]):
                tifffile.imwrite(file_path, array[idx].astype(self.output_dtype), imagej=True)

        elif self.output_type == "single-nii":
            nii_img = nib.Nifti1Image(array.astype(self.output_dtype), affine=np.eye(4)) # type: ignore
            output_path = self.output_path / f"{self.output_name}_z{z0}-{z1}.nii.gz"
            nib.save(nii_img, output_path) # type: ignore

        elif self.output_type == "scroll-nii":
            for idx, file_path in enumerate(self.output_file_path[z0:z1]):
                slice_2d = array[idx]  # idx from 0 to z1 - z0
                nii_img = nib.Nifti1Image(slice_2d.astype(self.output_dtype), affine=np.eye(4))  # type: ignore
                nib.save(nii_img, file_path)  # type: ignore

        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
    
    def write_ome_levels(self):
        if self.output_type != "ome-zarr":
            raise ValueError("generate_remaining_ome_levels can only be called for ome-zarr outputs.")

        if not hasattr(self, "store_group"):
            raise RuntimeError("OME-Zarr group is not initialized.")

        levels = sorted(int(k) for k in self.store_group.keys())

        for i in range(1, len(levels)):
            # Load previous level from Zarr as Dask array
            prev_arr = da.from_zarr(self.store_group[str(i - 1)].store, component=str(i - 1))

            # Get target shape for current level
            target_shape = self.store_group[str(i)].shape
            logger.info(f"Generating level {i} with shape {target_shape} from level {i - 1}...")

            two_pass_resize_zarr(
                output_source=self.store_group,
                level=i,
                current_shape=prev_arr.shape,
                target_shape=target_shape, # type: ignore
                dtype=np.dtype(self.output_dtype),
                order=self.resize_order,
                chunk_size=self.chunk_size,
            )
        
        datasets = []
        for level in range(len(levels)):
            scale_factor = self.resize_factor ** level
            datasets.append({
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [scale_factor] * 3
                    }
                ]
            })

        multiscales = [{
            "version": "0.4",
            "name": "image",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"}
            ],
            "datasets": datasets
        }]

        self.store_group.attrs["multiscales"] = multiscales
    
    def _initialize_ome(self, chunk_size=(128, 128, 128), n_level=5, resize_factor=2, resize_order=0):
        self.output_path = self.output_path / f"{self.output_name}_ome.zarr"
        self.resize_factor = resize_factor
        self.resize_order = resize_order
        self.chunk_size = tuple(chunk_size)
        
        store = zarr.DirectoryStore(self.output_path)
        self.store_group = zarr.group(store=store)
        
        for level in range(n_level):
            if level == 0:
                target_z, target_y, target_x = self.full_res_shape
                
            else:
                prev_shape = self.store_group[str(level - 1)].shape
                current_z, current_y, current_x = prev_shape
                target_z = int(current_z) // resize_factor
                target_y = int(current_y) // resize_factor
                target_x = int(current_x) // resize_factor
            
            if (target_z < 0 or target_y < 0 or target_x < 0):
                logger.info(f"Skipping level {level} due to insufficient shape: {(target_z, target_y, target_x)}")
                break
            
            self.store_group.create_dataset(
                str(level), 
                shape=(target_z, target_y, target_x), 
                chunks=chunk_size,
                dtype=self.output_dtype,
                compression=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
                overwrite=True
            )
        
    def _initialize_zarr(self, chunk_size=(128, 128, 128)):
        self.output_path = self.output_path / f"{self.output_name}.zarr"
        self.chunk_size = tuple(chunk_size)
        
        store = zarr.DirectoryStore(self.output_path)
        
        self.store_array = zarr.open_array(
            store=store,
            mode='w',
            shape=self.full_res_shape,
            chunks=chunk_size,
            dtype=self.output_dtype,
            compressor=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
        )
    
    def _initialize_single_tiff(self):
        os.makedirs(self.output_path, exist_ok=True)
    
    def _initialize_scroll_tiff(self, file_name):
        self.output_path = self.output_path / f"{self.output_name}_scroll"
        os.makedirs(self.output_path, exist_ok=True)
                
        self.output_file_path = [os.path.join(self.output_path, f"{name.stem}.tiff") for name in file_name]
        
    def _initialize_single_nii(self):
        os.makedirs(self.output_path, exist_ok=True)
    
    def _initialize_scroll_nii(self, file_name):
        self.output_path = self.output_path / f"{self.output_name}_scroll"
        os.makedirs(self.output_path, exist_ok=True)
        
        self.output_file_path = [os.path.join(self.output_path, f"{name.stem}.nii.gz") for name in file_name]