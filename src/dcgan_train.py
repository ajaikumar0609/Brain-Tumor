"""DCGAN baseline (64x64 grayscale MRI). Trains on archive/Training only.

    python src/dcgan_train.py --subset-size 512 --epochs 1 --out-dir outputs/dcgan_smoke

Objective: standard non-saturating DCGAN loss (BCE-with-logits; real=1, fake=0, no label smoothing).
Optimizers: Adam(lr=2e-4, betas=(0.5, 0.999)) for both nets; weights ~ N(0, 0.02); z_dim=100.
No class conditioning. No CPU fallback is enabled; a failing op raises.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from data_pipeline import DEFAULT_TRAIN_DIR, IDX_TO_CLASS, ManifestDataset, MRIDataset, rel_path

Z_DIM = 100


class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ch * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(True),   # 4
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(True),  # 8
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.ReLU(True),  # 16
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),         # 32
            nn.ConvTranspose2d(ch, 1, 4, 2, 1, bias=False), nn.Tanh(),                                      # 64
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """Outputs logits (sigmoid folded into BCEWithLogits)."""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                                  # 32
            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, True),    # 16
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.LeakyReLU(0.2, True),  # 8
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 8), nn.LeakyReLU(0.2, True),  # 4
            nn.Conv2d(ch * 8, 1, 4, 1, 0, bias=False),                                                       # 1
        )

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def all_finite_grads(module: nn.Module) -> bool:
    return all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in module.parameters())


def snapshot(module: nn.Module):
    return [p.detach().cpu().clone() for p in module.parameters()]


def params_changed(before, module: nn.Module) -> bool:
    return any(not torch.equal(a, b.detach().cpu()) for a, b in zip(before, module.parameters()))


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def sample_grid(G: nn.Module, z: torch.Tensor):
    was = G.training
    G.eval()
    out = G(z).cpu()
    G.train(was)
    return make_grid(out, nrow=8, normalize=True, value_range=(-1, 1), padding=2)


def save_before_after(before, after, path: Path, note: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    for ax, g, title in zip(axes, (before, after), ("before training (init)", note)):
        ax.imshow(g.permute(1, 2, 0).numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_grid(grid, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=None, help="stop after this many batches in total")
    ap.add_argument("--subset-size", type=int, default=512, help="0 = full training set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sample-every", type=int, default=0, help="save samples_step_NNNN.png (8x8 fixed-latent grid) every N batches (0 = off)")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="save checkpoint_step_NNNN.pth every N batches (0 = off)")
    ap.add_argument("--manifest", type=Path, default=None, help="train on the representatives of partition 'train' of a dedup split_manifest.csv instead of archive/Training")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/dcgan_smoke"))
    args = ap.parse_args()

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        print("ERROR: PYTORCH_ENABLE_MPS_FALLBACK is set; unset it so ops cannot silently run on CPU.", file=sys.stderr)
        return 2
    device = torch.device("mps" if args.device in ("auto", "mps") and torch.backends.mps.is_available() else "cpu")
    if args.device == "mps" and device.type != "mps":
        print("MPS requested but unavailable", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    full = ManifestDataset(args.manifest, 'train', 64, True) if args.manifest else MRIDataset(DEFAULT_TRAIN_DIR, image_size=64, validate=True)
    if full.skipped:
        print(f"WARNING: {len(full.skipped)} unreadable images skipped")
    assert args.manifest or all("Training" in Path(p).parts and "Testing" not in Path(p).parts for p, _ in full.samples)
    if args.subset_size and args.subset_size < len(full):
        idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(args.seed))[: args.subset_size].sort().values.tolist()
        ds = Subset(full, idx)
        chosen = [full.samples[i] for i in idx]
    else:
        ds, chosen = full, full.samples
    subset_hash = hashlib.sha256("\n".join(rel_path(p) for p, _ in chosen).encode()).hexdigest()
    subset_counts = {IDX_TO_CLASS[k]: v for k, v in sorted(Counter(l for _, l in chosen).items())}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "image_size": 64, "z_dim": Z_DIM, "optimizer": "Adam", "betas": [0.5, 0.999],
              "loss": "non-saturating BCE-with-logits (real=1, fake=0)", "weight_init": "N(0,0.02)",
              "train_dir": str(DEFAULT_TRAIN_DIR), "images_used": len(chosen), "subset_class_counts": subset_counts,
              "subset_file_list_sha256": subset_hash, "torch_version": str(torch.__version__), "device": str(device),
              "conditional": False}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"device={device} images={len(chosen)} {subset_counts} batches/epoch={len(loader)}")

    G, D = Generator().to(device), Discriminator().to(device)
    G.apply(init_weights)
    D.apply(init_weights)
    opt_g = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.999))
    g0, d0 = snapshot(G), snapshot(D)
    fixed_z = torch.randn(64, Z_DIM, generator=torch.Generator().manual_seed(args.seed)).to(device)
    grid_before = sample_grid(G, fixed_z)
    if args.sample_every:
        save_grid(grid_before, args.out_dir / "samples_step_0000.png", "DCGAN, step 0 (fixed latents)")

    rows, finite = [], True
    done = 0
    t_start = time.perf_counter()
    try:
        for epoch in range(args.epochs):
            for x, _ in loader:
                if args.max_batches is not None and done >= args.max_batches:
                    break
                x = x.to(device)
                b = x.size(0)
                ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
                z = torch.randn(b, Z_DIM, device=device)
                fake = G(z)

                opt_d.zero_grad()
                d_loss = F.binary_cross_entropy_with_logits(D(x), ones) + \
                    F.binary_cross_entropy_with_logits(D(fake.detach()), zeros)
                d_loss.backward()
                if not (torch.isfinite(d_loss).item() and all_finite_grads(D)):
                    finite = False
                    raise FloatingPointError(f"non-finite D loss/grad at batch {done}")
                opt_d.step()

                opt_g.zero_grad()
                g_loss = F.binary_cross_entropy_with_logits(D(fake), ones)
                g_loss.backward()
                if not (torch.isfinite(g_loss).item() and all_finite_grads(G)):
                    finite = False
                    raise FloatingPointError(f"non-finite G loss/grad at batch {done}")
                opt_g.step()

                done += 1
                rows.append((epoch, done, d_loss.item(), g_loss.item()))
                if done % (100 if args.sample_every else 8) == 0 or done == 1:
                    print(f"epoch {epoch} batch {done}: D={rows[-1][2]:.4f} G={rows[-1][3]:.4f} [{time.perf_counter() - t_start:.0f}s]")
                if args.sample_every and done % args.sample_every == 0:
                    save_grid(sample_grid(G, fixed_z), args.out_dir / f"samples_step_{done:04d}.png", f"DCGAN, step {done} (fixed latents)")
                if args.checkpoint_every and done % args.checkpoint_every == 0:
                    save_atomic({"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(),
                                 "opt_d": opt_d.state_dict(), "batches": done, "config": config,
                                 "elapsed_seconds": time.perf_counter() - t_start}, args.out_dir / f"checkpoint_step_{done:04d}.pth")
            if args.max_batches is not None and done >= args.max_batches:
                break
        if device.type == "mps":
            torch.mps.synchronize()
    except Exception as exc:
        print(f"TRAINING FAILED after {done} batches: {type(exc).__name__}: {exc}", file=sys.stderr)
        (args.out_dir / "metrics.json").write_text(json.dumps({"status": "failed", "error": str(exc),
                                                               "batches_completed": done, "device": str(device),
                                                               "all_losses_and_grads_finite": finite}, indent=2))
        return 1
    runtime = time.perf_counter() - t_start

    with open(args.out_dir / "losses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "batch", "d_loss", "g_loss"])
        w.writerows(rows)
    grid_after = sample_grid(G, fixed_z)
    save_before_after(grid_before, grid_after, args.out_dir / "samples.png", f"after {done} batches")

    ckpt = {"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
            "batches": done, "config": config}
    save_atomic(ckpt, args.out_dir / "checkpoint.pth")

    # verify: checkpoint reloads into fresh models and reproduces the same samples
    back = torch.load(args.out_dir / "checkpoint.pth", map_location="cpu", weights_only=True)
    G2 = Generator()
    G2.load_state_dict(back["G"])
    Discriminator().load_state_dict(back["D"])
    G.eval()
    G2.eval()
    with torch.no_grad():
        ckpt_roundtrip = bool(torch.allclose(G2(fixed_z.cpu()), G(fixed_z).cpu(), atol=1e-4))

    metrics = {
        "status": "ok", "device": str(device), "seed": args.seed, "image_size": 64, "batch_size": args.batch_size,
        "images_used": len(chosen), "subset_class_counts": subset_counts, "batches_completed": done,
        "runtime_seconds": round(runtime, 3), "avg_seconds_per_batch": round(runtime / max(done, 1), 4),
        "initial_d_loss": rows[0][2], "initial_g_loss": rows[0][3], "final_d_loss": rows[-1][2],
        "final_g_loss": rows[-1][3], "all_losses_and_grads_finite": finite,
        "generator_params_changed": params_changed(g0, G), "discriminator_params_changed": params_changed(d0, D),
        "checkpoint_reload_matches_samples": ckpt_roundtrip, "trained_on_testing_dir": False,
        "train_dir": str(DEFAULT_TRAIN_DIR), "mps_fallback_env_set": False,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
