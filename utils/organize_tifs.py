import argparse
from pathlib import Path
import shutil


def organize_tifs(
    input_dir: Path,
    output_dir: Path | None = None,
    start_index: int = 1,
    recursive: bool = False,
    copy: bool = False,
) -> int:
    output_dir = output_dir or input_dir

    pattern = "**/*.tif" if recursive else "*.tif"
    pattern2 = "**/*.tiff" if recursive else "*.tiff"
    files = sorted(list(input_dir.glob(pattern)) + list(input_dir.glob(pattern2)))

    if not files:
        print("No .tif/.tiff files found.")
        return 0

    width = max(2, len(str(start_index + len(files) - 1)))

    for idx, src in enumerate(files, start=start_index):
        folder_name = f"{idx:0{width}d}"
        dest_dir = output_dir / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / src.name
        if copy:
            shutil.copy2(src, dest)
        else:
            # If moving within same directory tree, ensure we don't try to move a file onto itself
            if src.resolve() == dest.resolve():
                # Already in place
                continue
            shutil.move(str(src), str(dest))

        print(f"{'Copied' if copy else 'Moved'} {src} -> {dest}")

    print(f"Done. Organized {len(files)} files into numbered folders at: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Move/copy each .tif image into its own numbered folder (01, 02, ...)."
    )
    parser.add_argument("--input", required=True, help="Input directory containing .tif/.tiff files")
    parser.add_argument("--output", help="Output directory (defaults to input directory)")
    parser.add_argument("--start", type=int, default=1, help="Starting index (default: 1)")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for TIFFs")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving")

    args = parser.parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.output) if args.output else None

    if not in_dir.is_dir():
        print(f"Input directory not found: {in_dir}")
        return 1

    return organize_tifs(
        input_dir=in_dir,
        output_dir=out_dir,
        start_index=args.start,
        recursive=args.recursive,
        copy=args.copy,
    )


if __name__ == "__main__":
    raise SystemExit(main())

