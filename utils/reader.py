import logging
import numpy as np

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize logging
logger = logging.getLogger(__name__)

# Define valid file suffixes
VALID_SUFFIXES = [".tif", ".tiff", ".nii.gz", ".gz", ".png", ".jpg"] # Zarr directories will be handled separately at line 64

@staticmethod
def _estimate_size_gb(shape: tuple, dtype: np.dtype) -> float:
    """Estimate in-memory size in GiB for an array of given shape and dtype."""
    return float(np.prod(shape) * np.dtype(dtype).itemsize / (1024 ** 3))

@staticmethod
def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Promote a 2D (y,x) array to 3D (1,y,x); leave >3D untouched."""
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    return arr

@staticmethod
def _swap_array(arr: np.ndarray, transpose: bool) -> np.ndarray:
    """Optionally transpose 3D (z,y,x)->(x,y,z) or 2D (y,x)->(x,y)."""
    if not transpose:
        return arr
    return arr.swapaxes(1, 2)

@staticmethod
def _swap_shape(shape: tuple, transpose: bool) -> tuple:
    """Optionally transpose a shape tuple (z,y,x)->(x,y,z) or (y,x)->(1,x,y)."""
    if not transpose:
        return shape
    if len(shape) == 3:
        z, y, x = shape
        return z, x, y
    else:
        raise ValueError(f"Unsupported shape for transposition: {shape}")

# ——— Readers ———

@staticmethod
def read_tiff(path: Path, read_to_array: bool = True, transpose: bool = False):
    import tifffile

    arr = tifffile.imread(str(path))

    if read_to_array:
        # Read full array
        arr = _ensure_3d(arr)
        return _swap_array(arr, transpose)
    
    # Metadata-only mode
    shape = arr.shape
    dtype = arr.dtype
    # normalize to 3D
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
    elif len(shape) > 3:
        raise ValueError(f"Unsupported NIfTI shape: {shape}")
    shape = _swap_shape(shape, transpose)
    size_gb = _estimate_size_gb(shape, dtype)
    return shape, dtype, size_gb


@staticmethod
def read_nii_gz(path: Path, read_to_array: bool = True, transpose: bool = False):
    import nibabel as nib
    
    img = nib.load(str(path), mmap=True) # type: ignore
    if read_to_array:
        arr = np.asanyarray(img.dataobj) # type: ignore
        arr = arr[..., np.newaxis] if arr.ndim == 2 else arr
        arr = arr.swapaxes(0, 2)
        return _swap_array(arr, transpose)
    
    # metadata‐only
    shape = img.shape # type: ignore
    dtype = img.get_data_dtype() # type: ignore
    # normalize to 3D
    if len(shape) == 2:
        shape = (1, shape[1], shape[0])
    elif len(shape) == 3:
        shape = (shape[2], shape[1], shape[0])
    elif len(shape) > 3:
        raise ValueError(f"Unsupported NIfTI shape: {shape}")
    shape = _swap_shape(shape, transpose)
    size_gb = _estimate_size_gb(shape, dtype)
    return shape, dtype, size_gb

@staticmethod
def read_zarr(path: Path, read_to_array: bool = True, transpose: bool = False):
    import zarr
    
    arr = zarr.open(str(path), mode='r')
    
    # If read_to_array is True, return the array
    if read_to_array:
        return arr
    
    # metadata‐only
    shape, dtype = arr.shape, arr.dtype
    return shape, dtype, 0


@staticmethod
def read_imageio(path: Path, read_to_array: bool = True, transpose: bool = False):
    import imageio.v3 as iio
    
    if read_to_array:
        arr = iio.imread(str(path))
        arr = _ensure_3d(arr)
        return _swap_array(arr, transpose)
    
    # metadata‐only
    arr = iio.imread(str(path))
    shape, dtype = arr.shape, arr.dtype
    shape = _swap_shape(shape, transpose)
    size_gb = _estimate_size_gb(shape, dtype)
    return shape, dtype, size_gb

# ——— Dispatcher ———

@staticmethod
def _read_image(
    file_path: Path,
    suffix: str,
    read_to_array: bool = True,
    transpose: bool = False
):
    """
    Unified reader. Returns either:
      - numpy array (if read_to_array=True)
      - (shape, dtype, size_gb) tuple
    """
    
    if suffix in (".tif", ".tiff"):
        reader = read_tiff
    elif suffix in (".nii", ".nii.gz", ".gz"):
        reader = read_nii_gz
    elif ".zarr" in str(file_path):
        reader = read_zarr
    else:
        reader = read_imageio

    try:
        return reader(file_path, read_to_array=read_to_array, transpose=transpose)
    except Exception as e:
        logger.error(f"Error in read_image({file_path}): {e}")
        raise
    

# ——— FileReader ———
class FileReader:
    def __init__(self, input_path, transpose=False, memory_limit_gb=32):
        self.input_path = Path(input_path)
        self.transpose = transpose
        self.memory_limit_bytes = memory_limit_gb * 1024 ** 3
        
        logger.info(f"Initializing FileReader with path: {self.input_path}")
        
        self.volume_name: str
        self.volume_files = []
        self.volume_sizes = []
        self.volume_types = []
        
        self._get_volume_files()
        
        logger.info(f"Found {len(self.volume_files)} volumes")
        
        self.volume_shape: tuple
        self.volume_dtype: np.dtype
        self.volume_cumulative_z = []
        
        self._get_volume_info()
        
        self._cache = {}
        
        logger.info(f"Volume name: {self.volume_name}")
        logger.info(f"Volume shape: {self.volume_shape}")
        logger.info(f"Volume dtype: {self.volume_dtype}")
    
    def _get_volume_files(self):
        # Get Volume Name
        self.volume_name = self.input_path.stem
        
        # Get Volume Files and Types
        suffix = self.input_path.suffix.lower()
        
        if suffix in VALID_SUFFIXES: # Single file case
            self.volume_files.append(self.input_path)
            self.volume_types.append(suffix)
        
        elif ".zarr" in str(self.input_path): # Zarr directory case
            self.volume_files.append(self.input_path)
            self.volume_types.append(".zarr")
            
        elif suffix == "": # Directory case
            check_suffixes = None
            for file in sorted(self.input_path.iterdir()):
                current_suffix = file.suffix.lower()
                if check_suffixes is None and current_suffix in VALID_SUFFIXES:
                    check_suffixes = current_suffix
                    self.volume_files.append(file)
                    self.volume_types.append(current_suffix)
                elif current_suffix == check_suffixes:
                    self.volume_files.append(file)
                    self.volume_types.append(current_suffix)
                else:
                    logger.warning(f"Files in directory {file} have different suffixes: {current_suffix} vs {check_suffixes}")
                    
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        if not self.volume_files or not self.volume_types:
            raise FileNotFoundError(f"No valid volume files found in {self.input_path}")
    
    def _get_volume_info(self):
        shapes = []
        dtypes = []

        def process(file, suffix):
            shape, dtype, size = _read_image(file, suffix, read_to_array=False, transpose=self.transpose)
            return shape, dtype, size

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process, file, suffix)
                for file, suffix in zip(self.volume_files, self.volume_types)
            ]

            with tqdm(total=len(futures), desc="Gathering volume info", leave=False) as pbar:
                for idx, future in enumerate(as_completed(futures)):
                    try:
                        shape, dtype, size = future.result()
                        shapes.append(shape[1:])  # Only y, x
                        dtypes.append(dtype)
                        self.volume_sizes.append(size)

                        if self.volume_cumulative_z:
                            self.volume_cumulative_z.append(shape[0] + self.volume_cumulative_z[-1])
                        else:
                            self.volume_cumulative_z.append(shape[0])

                    except Exception as e:
                        file, suffix = self.volume_files[idx], self.volume_types[idx]
                        raise RuntimeError(f"Error reading file {file.name} with suffix {suffix}: {e}")

                    pbar.update(1)

        # Ensure all (x, y) shapes match
        if len(set(shapes)) > 1:
            raise ValueError(f"Mismatch in XY dimensions across slices: {set(shapes)}")

        self.volume_shape = (self.volume_cumulative_z[-1], *shapes[0])  # (z, y, x)

        # Ensure all dtypes match
        if len(set(dtypes)) > 1:
            raise ValueError(f"Mismatch in data types across volume: {set(dtypes)}")

        self.volume_dtype = dtypes[0]
    
    def read(self, z_start=0, z_end=None, y_start=0, y_end=None, x_start=0, x_end=None):
        # 1) defaults
        z0, z1 = z_start, (self.volume_shape[0] if z_end is None else z_end)
        y0, y1 = y_start, (self.volume_shape[1] if y_end is None else y_end)
        x0, x1 = x_start, (self.volume_shape[2] if x_end is None else x_end)
        
        logger.info(f"Reading volume z: {z0} - {z1}, y: {y0} - {y1} x: {x0} - {x1}")
        dz = z1 - z0
        dy = y1 - y0
        dx = x1 - x0

        # 2) find which files overlap this Z-range
        prev_cum = [0] + self.volume_cumulative_z[:-1]
        needed = [i for i, (cum, prev) in enumerate(zip(self.volume_cumulative_z, prev_cum))
                if prev < z1 and cum > z0]

        # 3) memory check
        mem_limit = self.memory_limit_bytes / (1024**3)
        total_to_load = sum(self.volume_sizes[i] for i in needed)
        if total_to_load * 2 > mem_limit:
            raise MemoryError(f"Need {total_to_load*2:.2f}GiB but limit is {mem_limit:.2f}GiB")

        # 4) pre-allocate output
        out = np.empty((dz, dy, dx), dtype=self.volume_dtype)

        # 5) stream each file
        offset = 0
        for i in needed:
            base = prev_cum[i]
            file_z0 = max(0, z0 - base)
            file_z1 = min(self.volume_cumulative_z[i] - base, z1 - base)
            length = file_z1 - file_z0

            # load just this file (can use mmap for npy, nibabel, etc)
            arr = _read_image(
                self.volume_files[i],
                self.volume_types[i],
                True,
                self.transpose
            )
            # slice out only [file_z0:file_z1, y0:y1, x0:x1]
            slab = arr[file_z0:file_z1, y0:y1, x0:x1]
            out[offset:offset+length, :, :] = slab

            # drop references immediately
            del arr, slab
            offset += length

        return out