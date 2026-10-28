"""Side-by-side DCGAN vs WGAN-GP smoke samples from the SAME fixed latent vectors (+ real training images).

    python src/compare_smoke_samples.py
Crude diversity numbers only (mean per-pixel std across samples; mean pairwise L2). Not FID/KID.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import torch
from torchvision.utils import make_grid

from data_pipeline import MRIDataset
from dcgan_train import Generator, Z_DIM

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
out = root / "outputs" / "compare_smoke"
out.mkdir(parents=True, exist_ok=True)
z = torch.randn(64, Z_DIM, generator=torch.Generator().manual_seed(42))  # same fixed_z as both trainers


def load_g(path: Path) -> Generator:
    g = Generator()
    g.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["G"])
    return g.eval()


def diversity(x: torch.Tensor) -> dict:
    flat = x.flatten(1)
    return {"mean_pixel_std_across_samples": round(x.std(0).mean().item(), 4),
            "mean_pairwise_l2": round(torch.cdist(flat, flat).sum().item() / (len(flat) * (len(flat) - 1)), 3),
            "pixel_min": round(x.min().item(), 3), "pixel_max": round(x.max().item(), 3)}


with torch.no_grad():
    imgs = {"DCGAN (32 G / 32 D updates)": load_g(root / "outputs/dcgan_smoke/checkpoint.pth")(z),
            "WGAN-GP (32 G / 160 critic updates)": load_g(root / "outputs/wgangp_smoke/checkpoint.pth")(z)}
ds = MRIDataset(root / "archive" / "Training", 64, validate=False)
imgs["real (training images, reference)"] = torch.stack([ds[i * 87][0] for i in range(64)])

stats = {k: diversity(v) for k, v in imgs.items()}
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
for ax, (k, v) in zip(axes, imgs.items()):
    ax.imshow(make_grid(v, nrow=8, normalize=True, value_range=(-1, 1), padding=2).permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title(k)
    ax.axis("off")
fig.tight_layout()
fig.savefig(out / "dcgan_vs_wgangp.png", dpi=120)
(out / "diversity.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
