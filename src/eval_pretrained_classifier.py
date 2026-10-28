"""Evaluate the pretrained EfficientNet-B3 brain-tumor classifier on the Testing split.

Inference only: no training, no parameter changes, no weight downloads.
The architecture mirrors external_models/brain_tumor_classifier/src/model.py
(rebuilt here with weights=None so torchvision does not download ImageNet weights).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b3

# Checkpoint class order (BrainTumorDataset.CLASSES in the source repo).
CKPT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
# Explicit mapping from the on-disk folder names to checkpoint indices.
FOLDER_TO_IDX = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
MEAN, STD = 0.1856, 0.1892
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_model(num_classes: int = 4) -> nn.Module:
    backbone = efficientnet_b3(weights=None)
    old = backbone.features[0][0]
    backbone.features[0][0] = nn.Conv2d(
        1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=False,
    )
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(backbone.classifier[1].in_features, num_classes)
    )

    class Wrapper(nn.Module):  # keys must start with "backbone."
        def __init__(self):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            return self.backbone(x)

    return Wrapper()


class TestDataset(Dataset):
    def __init__(self, root: Path):
        missing = [n for n in FOLDER_TO_IDX if not (root / n).is_dir()]
        found = sorted(p.name for p in root.iterdir() if p.is_dir())
        if missing or set(found) != set(FOLDER_TO_IDX):
            raise SystemExit(f"Unexpected class folders in {root}: found {found}, expected {sorted(FOLDER_TO_IDX)}")
        self.samples = [
            (p, FOLDER_TO_IDX[name])
            for name in CKPT_CLASSES
            for p in sorted((root / name).iterdir())
            if p.suffix.lower() in IMAGE_EXTS
        ]
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            with Image.open(path) as im:
                return self.tf(im), label, str(path)
        except Exception as exc:  # corrupt / unsupported image
            return None, label, f"{path}: {exc}"


def collate(batch):
    ok = [b for b in batch if b[0] is not None]
    bad = [b[2] for b in batch if b[0] is None]
    if not ok:
        return None, None, bad
    return torch.stack([b[0] for b in ok]), torch.tensor([b[1] for b in ok]), bad


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-dir", type=Path, default=root / "archive" / "Testing")
    ap.add_argument("--checkpoint", type=Path,
                    default=root / "external_models/brain_tumor_classifier/weights/best_model.pth")
    ap.add_argument("--out-dir", type=Path, default=root / "outputs" / "pretrained_classifier")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = ap.parse_args()

    if args.test_dir.name.lower() == "training":
        raise SystemExit("Refusing to evaluate on a Training directory.")
    device = torch.device("mps" if args.device in ("auto", "mps") and torch.backends.mps.is_available() else "cpu")

    model = build_model()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    ds = TestDataset(args.test_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    print(f"device={device} images={len(ds)} test_dir={args.test_dir}")

    y_true, y_pred, skipped = [], [], []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for x, y, bad in loader:
            skipped += bad
            if x is None:
                continue
            pred = model(x.to(device)).argmax(1).cpu()
            y_true += y.tolist()
            y_pred += pred.tolist()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    if not y_true:
        raise SystemExit("No images could be evaluated.")

    from sklearn.metrics import classification_report, confusion_matrix

    labels = list(range(len(CKPT_CLASSES)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=CKPT_CLASSES,
                                   digits=4, zero_division=0)
    rep = classification_report(y_true, y_pred, labels=labels, target_names=CKPT_CLASSES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = np.bincount(y_true, minlength=4).tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
        "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in CKPT_CLASSES},
        "images_evaluated": len(y_true),
        "images_evaluated_per_class": dict(zip(CKPT_CLASSES, support)),
        "images_skipped": len(skipped),
        "skipped_details": skipped[:20],
        "device": str(device),
        "inference_seconds": round(elapsed, 3),
        "images_per_second": round(len(y_true) / elapsed, 2),
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "checkpoint": str(args.checkpoint.relative_to(root)) if args.checkpoint.is_relative_to(root) else str(args.checkpoint),
        "test_dir": str(args.test_dir.relative_to(root)) if args.test_dir.is_relative_to(root) else str(args.test_dir),
        "class_order": CKPT_CLASSES,
        "preprocessing": f"Grayscale(1) -> Resize(224,224) -> ToTensor -> Normalize({MEAN},{STD})",
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "classification_report.txt").write_text(
        f"Pretrained EfficientNet-B3 on {metrics['test_dir']} ({len(y_true)} images, {len(skipped)} skipped)\n\n{report}")
    with open(args.out_dir / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + CKPT_CLASSES)
        for name, row in zip(CKPT_CLASSES, cm.tolist()):
            w.writerow([name] + row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, CKPT_CLASSES, rotation=30, ha="right")
    ax.set_yticks(labels, CKPT_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Pretrained classifier — acc {metrics['accuracy']:.4f}")
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix.png", dpi=150)

    print(report)
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"weighted_f1={metrics['weighted_f1']:.4f} skipped={len(skipped)} "
          f"time={elapsed:.1f}s device={device}")
    print(f"results written to {args.out_dir}")


if __name__ == "__main__":
    main()
