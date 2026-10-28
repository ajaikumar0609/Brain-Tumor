"""Report image counts for a four-class brain MRI dataset without changing files."""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
KNOWN_CLASSES = {"glioma", "meningioma", "pituitary", "notumor", "no_tumor", "no tumor"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root containing Training/Testing or class folders")
    args = parser.parse_args()
    root = args.data_dir.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Dataset directory not found: {root}")

    print(f"Dataset root: {root}")
    total = 0
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit("No subfolders found. Extract the dataset first and point --data-dir at the extracted root.")

    for split_or_class in folders:
        children = sorted(p for p in split_or_class.iterdir() if p.is_dir())
        if children:
            split_total = 0
            print(f"\n[{split_or_class.name}]")
            for class_dir in children:
                n = count_images(class_dir)
                split_total += n
                print(f"  {class_dir.name:18s} {n:6d} images")
            print(f"  {'TOTAL':18s} {split_total:6d} images")
            total += split_total
        else:
            n = count_images(split_or_class)
            print(f"{split_or_class.name:24s} {n:6d} images")
            total += n

    print(f"\nTotal images found: {total}")
    if total == 0:
        print("WARNING: no image files were found. Check the extracted folder structure.")
    print("\nNote: counts validate the folder layout only; they do not verify labels, duplicates, or image quality.")


if __name__ == "__main__":
    main()
