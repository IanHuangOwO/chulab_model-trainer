"""
Command-line entry point for converting large 3D volumes into various
output formats (OME-Zarr pyramids, flat Zarr, TIFF/NIfTI volumes, and
per-slice "scroll" exports).

Example
  python converter.py ^
    --input_path D:\iansaididontcare\Chulab\others\CHL\M38\M38_anno_ome.zarr\2 ^
    --output_path D:\iansaididontcare\Chulab\others\CHL ^
    --output_type OME-Zarr ^
    --chunk-size 128 ^
    --resize-shape 2250 10240 7400 ^
    --resize-order 0

The CLI streams the input volume using `IO.reader.FileReader` and writes
results incrementally via `IO.writer.FileWriter` to keep memory bounded.
"""
import argparse
import logging
from pathlib import Path
import numpy as np

from IO import FileReader, FileWriter, OUTPUT_CHOICES, TYPE_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    """Parse CLI arguments describing the input volume and desired outputs.

    Returns:
        argparse.Namespace: Parsed flags including input/output paths,
            output type, chunk size, pyramid parameters, transpose order,
            and optional resize settings.
    """
    parser = argparse.ArgumentParser(description="Convert image volume to multiscale OME-Zarr or other formats.")

    # Positional arguments
    parser.add_argument("--input_path", type=str, required=True, help="Input file or directory path")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for result files")
    parser.add_argument("--output_type", type=str, required=True, choices=OUTPUT_CHOICES,
                        help="Specify the output format: OME-Zarr, Zarr, Tif, Scroll-Tif, Nifti, Scroll-Nifti.")

    # File Reader options
    parser.add_argument("--transpose", type=int, nargs=3, metavar=("AXIS0", "AXIS1", "AXIS2"),
        help="Axis order to apply with np.transpose, e.g. '--transpose 1 0 2'.",
    )

    # File Writer options
    parser.add_argument("--resize-shape", type=int, nargs=3, metavar=("Z", "Y", "X"),
                        help="Override the full-resolution volume shape")
    parser.add_argument("--resize-order", type=int, default=0,
                        help="Interpolation order for resizing: 0=nearest, 1=bilinear etc.")

    # OME options
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="Chunk size for Zarr storage")
    parser.add_argument("--downscale-factor", type=int, default=2,
                        help="Downsampling factor per pyramid level")
    parser.add_argument("--levels", type=int, default=5,
                        help="Number of pyramid levels to generate")

    # Scroll option
    parser.add_argument("--scroll-axis", type=int, default=0, choices=[0, 1, 2],
                        help="Axis to scroll and save 2D slices along (0=z, 1=y, 2=x). Default is 0 (z-axis).")

    # Memory limit
    parser.add_argument("--memory-limit", type=int, default=64,
                        help="Maximum memory (in GB) for temp buffers")

    return parser.parse_args()

def _write_pyramid(reader: FileReader, args, full_res_shape, chunk_tuple, io_output_type: str) -> bool:
    """Stream the full-resolution volume into a multiscale Zarr layout.

    Args:
        reader (FileReader): Source volume reader.
        args (argparse.Namespace): Parsed CLI arguments.
        full_res_shape (tuple[int, int, int]): Target full-resolution shape (Z, Y, X).
        chunk_tuple (tuple[int, int, int]): Chunk size used for Zarr arrays.
        io_output_type (str): Either "ome-zarr" or "zarr".

    Returns:
        bool: True when the streaming write completes (even if downsampling
            finalization later fails).
    """

    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(full_res_shape),
        output_dtype=reader.volume_dtype,
        chunk_size=chunk_tuple,
        n_level=args.levels,
        resize_factor=args.downscale_factor,
        resize_order=args.resize_order,
        input_shape=tuple(reader.volume_shape),
    )

    z_max = reader.volume_shape[0]
    for z0 in range(0, z_max, args.chunk_size):
        z1 = min(z0 + args.chunk_size, z_max)
        arr = reader.read(z_start=z0, z_end=z1)
        writer.write(arr, z_start=z0, z_end=z1)
        del arr

    try:
        writer.complete_resize()
    except Exception as e:
        logging.error(f"Failed to finalize resize into output: {e}")

    if io_output_type == "ome-zarr":
        writer.complete_ome()

    return True

def _write_single_volume(reader: FileReader, args, full_res_shape, io_output_type: str) -> bool:
    """Write a single full-resolution output volume for TIFF or NIfTI targets.

    Args:
        reader (FileReader): Source volume reader.
        args (argparse.Namespace): Parsed CLI args.
        full_res_shape (tuple[int, int, int]): Requested full-res output shape.
        io_output_type (str): One of "single-tiff" or "single-nii".

    Returns:
        bool: False if a mismatched resize is requested (unsupported here),
            True after writing otherwise.
    """
    if tuple(full_res_shape) != tuple(reader.volume_shape):
        logging.error("resize-shape currently not supported with IO.writer for single outputs. Use input shape or switch back to utils.* writer for resizing.")
        return False

    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(full_res_shape),
        output_dtype=reader.volume_dtype,
        input_shape=tuple(reader.volume_shape),
    )

    arr = reader.read(z_start=0, z_end=reader.volume_shape[0])
    writer.write(arr, z_start=0, z_end=reader.volume_shape[0])
    del arr
    return True

def _write_scroll_slices(reader: FileReader, args, full_res_shape, io_output_type: str) -> bool:
    """Emit individual 2D slices along the selected axis for scroll outputs.

    Args:
        reader (FileReader): Source volume reader.
        args (argparse.Namespace): Parsed CLI args; uses ``scroll_axis``.
        full_res_shape (tuple[int, int, int]): Expected (Z, Y, X) shape.
        io_output_type (str): "scroll-tiff" or "scroll-nii".

    Returns:
        bool: False if resize-shape mismatches the input; True otherwise.
    """
    if tuple(full_res_shape) != tuple(reader.volume_shape):
        logging.error("resize-shape currently not supported with IO.writer for single outputs. Use input shape or switch back to utils.* writer for resizing.")
        return False
    axis_char = ["z", "y", "x"][args.scroll_axis]
    num_slices = reader.volume_shape[args.scroll_axis]
    base = Path(args.output_path) / f"{reader.volume_name}_scroll"
    file_names = [base / f"{reader.volume_name}_{axis_char}{i:05d}" for i in range(num_slices)]

    writer = FileWriter(
        output_path=args.output_path,
        output_name=reader.volume_name,
        output_type=io_output_type,
        full_res_shape=tuple(reader.volume_shape),
        output_dtype=reader.volume_dtype,
        file_name=[Path(n) for n in file_names],
        input_shape=tuple(reader.volume_shape),
    )

    axis = args.scroll_axis
    axis_length = reader.volume_shape[axis]

    axis_handlers = {
        0: lambda start, end: reader.read(z_start=start, z_end=end),
        1: lambda start, end: np.transpose(reader.read(y_start=start, y_end=end), (1, 0, 2)),
        2: lambda start, end: np.transpose(reader.read(x_start=start, x_end=end), (2, 0, 1)),
    }

    handler = axis_handlers[axis]
    step = args.chunk_size

    for start in range(0, axis_length, step):
        end = min(start + step, axis_length)
        arr = handler(start, end)
        writer.write(arr, z_start=start, z_end=end)
        del arr

    return True

def main():
    """Entry point that orchestrates reading, conversion, and writing.

    The function determines the correct IO pipeline based on ``--output_type``
    and streams data in chunks to keep memory usage predictable.
    """
    args = parse_args()

    logging.info("Starting conversion process.")
    logging.info(f"Input path: {args.input_path}")
    logging.info(f"Output path: {args.output_path}")
    logging.info(f"Output type: {args.output_type}")
    logging.info(f"Memory limit: {args.memory_limit} GB")
    if args.transpose:
        logging.info(f"Transpose order: {tuple(args.transpose)}")

    reader = FileReader(
        input_path=args.input_path,
        memory_limit_gb=args.memory_limit,
        transpose_order=tuple(args.transpose) if args.transpose else None,
    )

    full_res_shape = tuple(args.resize_shape) if args.resize_shape else reader.volume_shape
    logging.info(f"Full-resolution shape: {full_res_shape}")

    io_output_type = TYPE_MAP.get(args.output_type)
    if io_output_type is None:
        logging.error(f"Unsupported output_type: {args.output_type}")
        return

    # Ensure output directory exists
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    chunk_tuple = (args.chunk_size, args.chunk_size, args.chunk_size)

    if io_output_type in ["ome-zarr", "zarr"]:
        if not _write_pyramid(reader, args, full_res_shape, chunk_tuple, io_output_type):
            return
    elif io_output_type in ["single-tiff", "single-nii"]:
        if not _write_single_volume(reader, args, full_res_shape, io_output_type):
            return
    elif io_output_type in ["scroll-tiff", "scroll-nii"]:
        _write_scroll_slices(reader, args, full_res_shape, io_output_type)
    else:
        logging.error(f"Unsupported output_type: {args.output_type}")
        return

    logging.info("Conversion complete.")

if __name__ == "__main__":
    main()
