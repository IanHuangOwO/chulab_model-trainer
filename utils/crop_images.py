import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from skimage import io as skio


def center_crop(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Center-crop ndarray to target (H, W) over the last two dims.

    Supports shapes: (H, W), (H, W, C), (Z, H, W), (Z, H, W, C).
    """
    th, tw = target_hw
    *leading, h, w = arr.shape
    if h < th or w < tw:
        raise ValueError(f"Image smaller than target: {(h, w)} < {(th, tw)}")

    y0 = (h - th) // 2
    x0 = (w - tw) // 2
    slices = tuple([slice(None)] * len(leading) + [slice(y0, y0 + th), slice(x0, x0 + tw)])
    return arr[slices]


def center_pad(arr: np.ndarray, target_hw: Tuple[int, int], pad_value=0) -> np.ndarray:
    """Center-pad ndarray to target (H, W) over the last two dims with constant value.

    Supports shapes: (H, W), (H, W, C), (Z, H, W), (Z, H, W, C).
    """
    th, tw = target_hw
    *leading, h, w = arr.shape

    out_shape = tuple(leading + [th, tw])
    out = np.full(out_shape, pad_value, dtype=arr.dtype)

    y0 = max((th - h) // 2, 0)
    x0 = max((tw - w) // 2, 0)

    yh = min(h, th)
    xw = min(w, tw)

    in_y0 = max((h - th) // 2, 0)
    in_x0 = max((w - tw) // 2, 0)

    in_slices = tuple([slice(None)] * len(leading) + [slice(in_y0, in_y0 + yh), slice(in_x0, in_x0 + xw)])
    out_slices = tuple([slice(None)] * len(leading) + [slice(y0, y0 + yh), slice(x0, x0 + xw)])
    out[out_slices] = arr[in_slices]
    return out


def save_image(path: Path, arr: np.ndarray) -> None:
    """Save image/stack to path as TIFF when extension is .tif/.tiff, else use skimage.io.imsave.

    For multipage arrays (ndim >= 3 and not RGB), attempts to save as stack.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        try:
            import tifffile as tiff
            tiff.imwrite(str(path), arr)
            return
        except Exception:
            pass
    # Fallback: skimage
    skio.imsave(str(path), arr)


def main():
    parser = argparse.ArgumentParser(description="Crop all images in a directory to a fixed size (center crop).")
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--output", required=True, help="Output directory for cropped images")
    parser.add_argument("--width", type=int, default=2000, help="Target width (default: 2000)")
    parser.add_argument("--height", type=int, default=2000, help="Target height (default: 1600)")
    parser.add_argument("--pattern", default="**/*", help="Glob pattern (default: **/* for all)")
    parser.add_argument("--extensions", nargs="*", default=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
                        help="List of file extensions to include")
    parser.add_argument("--pad_if_small", action="store_true", help="Pad images smaller than target instead of skipping")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    target = (args.height, args.width)
    exts = set(e.lower() for e in args.extensions)

    files = [p for p in in_dir.glob(args.pattern) if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print("No images found matching pattern and extensions.")
        return 0

    for src in tqdm(files, desc="Cropping"):
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        try:
            img = skio.imread(str(src))
            h, w = img.shape[-2], img.shape[-1]

            if h < target[0] or w < target[1]:
                if args.pad_if_small:
                    out = center_pad(img, target)
                else:
                    print(f"Skipping (smaller than target): {src} -> {(w, h)}")
                    continue
            else:
                out = center_crop(img, target)

            save_image(dst.with_suffix(src.suffix), out)
        except Exception as e:
            print(f"Error processing {src}: {e}")

    print(f"Done. Saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

