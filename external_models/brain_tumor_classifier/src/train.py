"""
Training utilities: one-epoch train loop, evaluation loop, and phase runner.
All functions are pure (no global state) — they receive everything they need
as arguments, which makes them easy to test and reuse in a notebook.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# ======================================================================
#  SINGLE-EPOCH LOOPS
# ======================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        avg_loss  (float) — mean loss over all batches
        accuracy  (float) — percentage of correct predictions (0 - 100)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # Progress bar with epoch info and live loss updates
    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch + 1}/{total_epochs} [train]",
        leave=False,
    )

    for batch_idx, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)

        # classic training step: zero_grad -> forward -> loss -> backward -> step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)          # labels are integer indices
        loss.backward()
        optimizer.step()

        # keeping track of running loss and accuracy for live updates in the progress bar
        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{running_loss / (batch_idx + 1):.4f}"})

    # final average loss and accuracy for the epoch
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on a data loader (no gradient updates).

    Returns:
        avg_loss  (float)
        accuracy  (float) — percentage (0-100)
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # accumulate loss and accuracy stats for the entire loader
            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ======================================================================
#  PHASE RUNNER
# ======================================================================

def run_phase(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    phase_name: str,
    save_path: str,
    start_epoch: int = 0,
) -> tuple[dict, float]:
    """
    Train the model for one phase (either head-only or full fine-tuning).

    Optimizer : AdamW (only trains parameters where requires_grad=True)
    Scheduler : CosineAnnealingLR — smooth LR decay beats a flat rate by 1-2%

    Saves the best checkpoint (by test accuracy) to `save_path`.

    Returns:
        history   (dict)  — keys: train_loss, train_acc, test_loss, test_acc
        best_acc  (float) — best test accuracy achieved in this phase
    """
    # Only optimize unfrozen parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # CosineAnnealingLR: starts at `lr` and decays to 0 over `epochs` epochs, then restarts if more epochs are run
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # history dict to store loss and accuracy for each epoch, used for plotting later
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
    }
    best_acc = 0.0
    total_epochs = start_epoch + epochs

    # main training loop for the phase
    for i in range(epochs):
        epoch = start_epoch + i

        tl, ta = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, total_epochs
        )
        vl, va = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_loss"].append(vl)
        history["test_acc"].append(va)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\n[{phase_name}] Epoch {epoch + 1}/{total_epochs} | LR: {current_lr:.2e}\n"
            f"  Train  → Loss: {tl:.4f} | Acc: {ta:.2f}%\n"
            f"  Test   → Loss: {vl:.4f} | Acc: {va:.2f}%"
        )

        if va > best_acc:
            best_acc = va
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"\033[92m  ✓ New best ({phase_name}): {best_acc:.2f}% — saved to {save_path}\033[0m")

    return history, best_acc
