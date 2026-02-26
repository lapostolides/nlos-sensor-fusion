"""
Convert all .heic images in a directory to .jpg.
Usage: python convert_heic.py <path> [--output <dir>] [--delete-originals]
"""

import argparse
import sys
from pathlib import Path

try:
    import pillow_heif
    from PIL import Image
    pillow_heif.register_heif_opener()
except ImportError:
    print("Missing dependency. Install with: pip install pillow pillow-heif")
    sys.exit(1)


def convert_heic_to_jpg(
    directory: Path, output: Path = None, delete_originals: bool = False
) -> None:
    heic_files = list(directory.glob("*.[Hh][Ee][Ii][Cc]"))

    if not heic_files:
        print(f"No .heic files found in {directory}")
        return

    out_dir = output if output else directory
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(heic_files)} .heic file(s) in {directory}")
    print(f"Output directory: {out_dir}")

    for heic_path in sorted(heic_files):
        jpg_path = out_dir / heic_path.with_suffix(".jpg").name
        try:
            with Image.open(heic_path) as img:
                rgb = img.convert("RGB")
                rgb.save(jpg_path, "JPEG", quality=95)
            print(f"  {heic_path.name} -> {jpg_path.name}")
            if delete_originals:
                heic_path.unlink()
        except Exception as e:
            print(f"  ERROR converting {heic_path.name}: {e}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Convert HEIC images to JPG.")
    parser.add_argument("path", type=Path, help="Directory containing .heic files")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for .jpg files (default: same as input)")
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original .heic files after conversion",
    )
    args = parser.parse_args()

    if not args.path.is_dir():
        print(f"Error: {args.path} is not a directory")
        sys.exit(1)

    convert_heic_to_jpg(args.path, output=args.output, delete_originals=args.delete_originals)


if __name__ == "__main__":
    main()
