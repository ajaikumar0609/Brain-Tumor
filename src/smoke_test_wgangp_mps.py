"""WGAN-GP compatibility smoke test (1 critic step + 1 generator step). NOT training.

    python src/smoke_test_wgangp_mps.py [--device auto|mps|cpu] [--batch-size 4]

Everything (model, data, autograd incl. the double-backward of the gradient penalty)
runs on the selected device; there is no silent CPU fallback. Exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import torch
import torch.nn as nn

from data_pipeline import make_loader

Z_DIM = 100


class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ch * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(True),   # 4x4
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.ReLU(True),  # 16x16
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),         # 32x32
            nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Tanh(),                                                  # 64x64
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Critic(nn.Module):
    """No BatchNorm (invalid with per-sample gradient penalty)."""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2, True),            # 32
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.LeakyReLU(0.2, True),       # 16
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.LeakyReLU(0.2, True),   # 8
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1), nn.LeakyReLU(0.2, True),   # 4
            nn.Conv2d(ch * 8, 1, 4, 1, 0),                                 # 1
        )

    def forward(self, x):
        return self.net(x).view(-1)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interp)
    grads, = torch.autograd.grad(scores.sum(), interp, create_graph=True)
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def grads_finite(module: nn.Module) -> tuple[bool, int]:
    gs = [p.grad for p in module.parameters() if p.requires_grad]
    n = sum(g is not None for g in gs)
    return all(g is not None and torch.isfinite(g).all().item() for g in gs), n


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        raise RuntimeError(f"check failed: {name} {detail}")


def run(device: torch.device, batch_size: int, lambda_gp: float = 10.0) -> None:
    torch.manual_seed(0)
    x_real, _ = next(iter(make_loader(batch_size=batch_size, shuffle=True, validate=False)))
    x_real = x_real.to(device)
    check("real batch", x_real.shape == (batch_size, 1, 64, 64) and x_real.min() >= -1 and x_real.max() <= 1,
          f"shape={tuple(x_real.shape)} device={x_real.device}")

    G, D = Generator().to(device), Critic().to(device)
    opt_g = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.9))
    g_before = [p.detach().clone() for p in G.parameters()]
    d_before = [p.detach().clone() for p in D.parameters()]

    # ---- critic step ----
    t0 = time.perf_counter()
    z = torch.randn(batch_size, Z_DIM, device=device)
    with torch.no_grad():
        fake = G(z)
    check("generator forward", fake.shape == (batch_size, 1, 64, 64) and torch.isfinite(fake).all().item(),
          f"shape={tuple(fake.shape)}")
    d_real, d_fake = D(x_real), D(fake)
    check("critic scores", torch.isfinite(d_real).all().item() and torch.isfinite(d_fake).all().item(),
          f"real_mean={d_real.mean().item():.4f} fake_mean={d_fake.mean().item():.4f}")
    gp = gradient_penalty(D, x_real, fake)
    check("gradient penalty (autograd, create_graph)", torch.isfinite(gp).item(), f"gp={gp.item():.4f}")
    d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
    check("wasserstein critic loss finite", torch.isfinite(d_loss).item(), f"d_loss={d_loss.item():.4f}")
    opt_d.zero_grad()
    d_loss.backward()                      # double-backward through the penalty
    ok, n = grads_finite(D)
    check("critic backward (incl. GP double-backward)", ok, f"{n} param grads finite")
    opt_d.step()
    sync(device)
    t_critic = time.perf_counter() - t0

    # ---- generator step ----
    t0 = time.perf_counter()
    z = torch.randn(batch_size, Z_DIM, device=device)
    g_loss = -D(G(z)).mean()
    check("generator loss finite", torch.isfinite(g_loss).item(), f"g_loss={g_loss.item():.4f}")
    opt_g.zero_grad()
    g_loss.backward()
    ok, n = grads_finite(G)
    check("generator backward", ok, f"{n} param grads finite")
    opt_g.step()
    sync(device)
    t_gen = time.perf_counter() - t0

    check("critic params updated", any(not torch.equal(a, b) for a, b in zip(d_before, D.parameters())))
    check("generator params updated", any(not torch.equal(a, b) for a, b in zip(g_before, G.parameters())))
    print(f"timing: critic_step={t_critic:.3f}s generator_step={t_gen:.3f}s (first step includes MPS warm-up)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    want_mps = args.device in ("auto", "mps")
    device = torch.device("mps" if want_mps and torch.backends.mps.is_available() else "cpu")
    if args.device == "mps" and device.type != "mps":
        print("MPS requested but unavailable", file=sys.stderr)
        return 2
    print(f"torch {torch.__version__} | device: {device} | batch_size: {args.batch_size}")
    t0 = time.perf_counter()
    try:
        run(device, args.batch_size)
    except Exception:
        traceback.print_exc()
        print(f"SMOKE TEST FAILED on {device}. Isolate with --device cpu to see if the failure is MPS-specific.",
              file=sys.stderr)
        return 1
    print(f"SMOKE TEST PASSED on {device} (all ops ran on {device}, no CPU fallback) in "
          f"{time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
