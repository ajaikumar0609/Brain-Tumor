"""Check Python, PyTorch, torchvision, and available compute device."""
from __future__ import annotations
import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Activate the project venv and run: pip install -r requirements.txt"
        ) from exc

    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Selected device: {device}")
    if device.type == "mps":
        print("Apple Silicon GPU acceleration is available through MPS.")
    elif device.type == "cuda":
        print("NVIDIA CUDA GPU acceleration is available.")
    else:
        print("GPU acceleration was not detected; CPU mode will work but GAN training may be slower.")

    # A small smoke test catches basic device/tensor issues.
    try:
        x = torch.randn(2, 3, 64, 64, device=device)
        y = (x * 2).mean()
        print(f"Tensor smoke test: passed (result={y.item():.5f})")
    except Exception as exc:  # keep the diagnostic informative on less common setups
        raise SystemExit(f"Tensor/device smoke test failed: {exc}") from exc


if __name__ == "__main__":
    main()
