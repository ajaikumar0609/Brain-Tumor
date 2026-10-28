import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for the Brain Tumor MRI dataset.

    Folder structure expected under `root_dir`:
        root_dir/
            glioma/       *.jpg | *.jpeg | *.png | *.bmp
            meningioma/
            notumor/
            pituitary/

    Labels are returned as integer class indices (0-3), which is what
    nn.CrossEntropyLoss expects directly - no one-hot conversion needed.
    (this version cleaned some of the heavy work I did in the notebook)
    """

    CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        self.samples = self._load_samples()

    # ------------------------------------------------------------------
    def _load_samples(self) -> list[tuple[str, int]]:
        """Walk root_dir and return a list of (image_path, class_index) pairs."""
        samples = []
        for class_name in self.CLASSES:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Expected class directory not found: '{class_dir}'\n"
                    f"Make sure your data/ folder matches the structure in README.md."
                )
            for file_name in sorted(os.listdir(class_dir)):  # sorted -> reproducible order
                if os.path.splitext(file_name)[1].lower() in self.VALID_EXTENSIONS:
                    samples.append((
                        os.path.join(class_dir, file_name), # image path only (no processed images)
                        self.class_to_idx[class_name],   # integer label (class index)
                    ))
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")   # force single-channel grayscale
        if self.transform:
            image = self.transform(image)
        return image, label
