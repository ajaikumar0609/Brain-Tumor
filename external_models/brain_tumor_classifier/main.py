"""
main.py — Training entrypoint for the Brain Tumor MRI Classifier.

All hyperparameters are declared at the top of this file.
The heavy logic lives in src/ and is imported here.

Usage:
    python main.py
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset import BrainTumorDataset
from src.model import BrainTumorEfficientNet
from src.train import run_phase


# ======================================================================
#  CONFIG  -  edit only this section
# ======================================================================

DATASET_PATH = "data"          # root folder that contains Training/ and Testing/
WEIGHTS_DIR  = "weights"       # where .pth checkpoints are saved

BATCH_SIZE   = 32
NUM_CLASSES  = 4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Two-phase training
PHASE1_EPOCHS = 10             # backbone frozen  — head warm-up
PHASE2_EPOCHS = 30             # full fine-tuning — backbone + head

PHASE1_LR    = 1e-3
PHASE2_LR    = 1e-4            # 10× smaller to avoid overwriting pretrained weights
WEIGHT_DECAY = 1e-4

# Pre-computed dataset statistics (grayscale)
MEAN = 0.1856
STD  = 0.1892

# ======================================================================
#  TRANSFORMS
# ======================================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),              # random crop -> plain resize to 224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        shear=5,
    ),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]),
    transforms.RandomErasing(p=0.2),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]),
])


# ======================================================================
#  DATA
# ======================================================================

train_dataset = BrainTumorDataset(
    root_dir=os.path.join(DATASET_PATH, "Training"),
    transform=train_transform,
)
test_dataset = BrainTumorDataset(
    root_dir=os.path.join(DATASET_PATH, "Testing"),
    transform=test_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"\033[94mDevice : {DEVICE}\033[0m")
print(
    f"\033[92mTrain  : {len(train_dataset)} samples | "
    f"Test : {len(test_dataset)} samples\033[0m"
)

# ======================================================================
#  LOSS
# ======================================================================

# label_smoothing=0.1 prevents overconfidence and helps on smaller datasets
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ======================================================================
#  TRAINING
# ======================================================================

model = BrainTumorEfficientNet(num_classes=NUM_CLASSES).to(DEVICE)

# -- Phase 1: head-only warm-up -----------------------------------------
print("\n\033[94m══ PHASE 1: Head-only training (backbone frozen) ══\033[0m")
model.freeze_backbone()

_, best1 = run_phase(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    device=DEVICE,
    lr=PHASE1_LR,
    weight_decay=WEIGHT_DECAY,
    epochs=PHASE1_EPOCHS,
    phase_name="Phase 1",
    save_path=os.path.join(WEIGHTS_DIR, "best_model_phase_1.pth"),
    start_epoch=0,
)

# -- Phase 2: full fine-tuning -----------------------------------------
print("\n\033[94m══ PHASE 2: Full fine-tuning (backbone + head) ══\033[0m")
model.unfreeze_backbone()

# start Phase 2 from the best Phase 1 checkpoint
model.load_state_dict(
    torch.load(
        os.path.join(WEIGHTS_DIR, "best_model_phase_1.pth"),
        map_location=DEVICE,
    )
)

_, best2 = run_phase(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    device=DEVICE,
    lr=PHASE2_LR,
    weight_decay=WEIGHT_DECAY,
    epochs=PHASE2_EPOCHS,
    phase_name="Phase 2",
    save_path=os.path.join(WEIGHTS_DIR, "best_model_phase_2.pth"),
    start_epoch=PHASE1_EPOCHS,
)

# -- Summary -----------------------------------------
print(
    f"\n\033[92m"
    f"══ Training complete ══\n"
    f"   Phase 1 best test acc : {best1:.2f}%\n"
    f"   Phase 2 best test acc : {best2:.2f}%\n"
    f"   Overall best          : {max(best1, best2):.2f}%"
    f"\033[0m"
)
