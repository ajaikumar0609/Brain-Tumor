"""Shared GAN data pipeline: archive/Training only, 64x64 grayscale, [-1, 1].

    python src/data_pipeline.py            # prints counts, label map, one sample batch

Leakage guard: any root whose path contains a 'Testing' component is rejected.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "archive" / "Training"
CLASS_TO_IDX = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ARCHIVE = PROJECT_ROOT / "archive"


def rel_path(p) -> str:
    """Path relative to archive/Training when possible (keeps earlier hashes unchanged), else relative to archive/."""
    try:
        return str(Path(p).relative_to(DEFAULT_TRAIN_DIR))
    except ValueError:
        return str(Path(p).relative_to(ARCHIVE))


def _check_train_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if any(part.lower() == "testing" for part in root.parts):
        raise ValueError(f"Refusing to use a Testing directory for GAN training: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"Training directory not found: {root}")
    found = {p.name for p in root.iterdir() if p.is_dir()}
    if found != set(CLASS_TO_IDX):
        raise ValueError(f"Class folders {sorted(found)} do not match expected {sorted(CLASS_TO_IDX)}")
    return root


def scan_images(root: Path, validate: bool = True):
    """Return (samples, skipped). samples = [(path, label)]; skipped = [(path, reason)]."""
    samples, skipped = [], []
    for name, idx in CLASS_TO_IDX.items():
        for p in sorted((root / name).iterdir()):
            if p.name.startswith(".") or not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                skipped.append((p, "unsupported extension"))
                continue
            if validate:
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception as exc:
                    skipped.append((p, f"corrupt: {exc}"))
                    continue
            samples.append((p, idx))
    return samples, skipped


class MRIDataset(Dataset):
    """(image[1,S,S] in [-1,1], label) pairs from the Training split."""

    def __init__(self, root: Path = DEFAULT_TRAIN_DIR, image_size: int = 64, validate: bool = True):
        self.root = _check_train_root(root)
        self.samples, self.skipped = scan_images(self.root, validate)
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                      # [0, 1]
            transforms.Normalize([0.5], [0.5]),         # [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            return self.tf(im), label

    def class_counts(self) -> dict[str, int]:
        c = Counter(label for _, label in self.samples)
        return {IDX_TO_CLASS[i]: c[i] for i in sorted(IDX_TO_CLASS)}


class ManifestDataset(Dataset):
    """(image[1,S,S] in [-1,1], label) pairs for ONE partition of a cluster-aware split manifest (representatives only).

    Used by the de-duplicated experiment; the manifest comes from src/dedup_split.py. Only rows with partition == `partition`
    and is_representative == True are used, so copies of an image never enter a partition twice."""

    def __init__(self, manifest, partition: str = "train", image_size: int = 64, validate: bool = True):
        self.root = ARCHIVE
        self.partition = partition
        self.samples, self.skipped = [], []
        for r in csv.DictReader(open(manifest)):
            if r["partition"] != partition or r["is_representative"] != "True":
                continue
            p = ARCHIVE / r["source_path"]
            label = int(r["class_idx"])
            assert CLASS_TO_IDX[r["class_name"]] == label
            if validate:
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception as exc:
                    self.skipped.append((p, f"corrupt: {exc}"))
                    continue
            self.samples.append((p, label))
        self.tf = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((image_size, image_size)),
                                      transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            return self.tf(im), label

    def class_counts(self) -> dict[str, int]:
        c = Counter(label for _, label in self.samples)
        return {IDX_TO_CLASS[i]: c[i] for i in sorted(IDX_TO_CLASS)}


def make_loader(batch_size: int = 16, shuffle: bool = True, image_size: int = 64,
                root: Path = DEFAULT_TRAIN_DIR, num_workers: int = 0, validate: bool = True,
                drop_last: bool = True) -> DataLoader:
    ds = MRIDataset(root, image_size, validate)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    args = ap.parse_args()

    loader = make_loader(args.batch_size, True, args.image_size, args.train_dir)
    ds = loader.dataset
    print(f"train_dir      : {ds.root}")
    print(f"label mapping  : {CLASS_TO_IDX}")
    print(f"valid images   : {len(ds)}   skipped: {len(ds.skipped)}")
    for p, why in ds.skipped[:10]:
        print(f"  skipped {p}: {why}")
    print(f"per-class count: {ds.class_counts()}")
    x, y = next(iter(loader))
    print(f"batch images   : shape={tuple(x.shape)} dtype={x.dtype} min={x.min():.3f} max={x.max():.3f}")
    print(f"batch labels   : {y.tolist()}  shape={tuple(y.shape)}")
    print("data loader OK")


if __name__ == "__main__":
    main()
