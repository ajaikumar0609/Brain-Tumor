"""Hybrid Conditional WGAN-GP (proposed model): class-conditioned generator + projection critic
+ Wasserstein objective + gradient penalty. 64x64 grayscale MRI, archive/Training only.

    python src/cwgangp_train.py --subset-size 512 --gen-steps 32 --out-dir outputs/cwgangp_smoke        # smoke test
    python src/cwgangp_train.py --subset-size 0 --gen-steps 500 --sample-every 100 --checkpoint-every 100 \\
        --out-dir outputs/cwgangp_500                                                                    # periodic run
    python src/cwgangp_train.py --subset-size 0 --gen-steps 1000 --sample-every 100 --out-dir outputs/cwgangp_500 \\
        --resume outputs/cwgangp_500/checkpoint_step_0500.pth                                            # resume

Conditioning
  Generator  G(z, y): label embedding E_g(y) in R^50 is concatenated with z in R^100 -> ConvT stack -> 1x64x64 Tanh.
  Critic     D(x, y): conv trunk (no BatchNorm) -> sum-pooled feature h(x) in R^512;
             score = w^T h(x) + b + <E_d(y), h(x)>   (projection critic; unbounded scalar, no sigmoid).
Loss (both MINIMISED; real and fake in a critic step share the SAME labels y from the real batch)
  critic     L_D = E_{z,y}[D(G(z,y), y)] - E_{(x,y)}[D(x, y)] + lambda * E[(||grad_xhat D(xhat, y)||_2 - 1)^2]
             xhat = eps*x + (1-eps)*G(z,y), eps~U(0,1), y held fixed.   w_dist = E[D(x,y)] - E[D(G(z,y),y)]
  generator  L_G = -E_{z, y~U{0..3}}[D(G(z,y), y)]
n_critic=5, lambda=10, Adam(lr=1e-4, betas=(0.5, 0.9)), z_dim=100, batch 16. No classifier guidance (not yet added).
No CPU fallback is enabled. --sample-every > 0 selects the periodic mode (samples_step_NNNN.png, checkpoint_step_NNNN.pth).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from data_pipeline import CLASS_TO_IDX, DEFAULT_TRAIN_DIR, IDX_TO_CLASS, ManifestDataset, MRIDataset, rel_path
from dcgan_train import Z_DIM, all_finite_grads, init_weights, params_changed, save_atomic, set_seed, snapshot

NUM_CLASSES = 4
EMB_DIM = 50
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CondGenerator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, emb_dim: int = EMB_DIM, ch: int = 64):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, emb_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + emb_dim, ch * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ch * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.ConvTranspose2d(ch, 1, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.embed(y)], dim=1)
        return self.net(h.view(h.size(0), -1, 1, 1))


class ProjectionCritic(nn.Module):
    """Projection critic without BatchNorm: D(x,y) = psi(h(x)) + <embed(y), h(x)>."""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1), nn.LeakyReLU(0.2, True),   # B x 512 x 4 x 4
        )
        self.psi = nn.Linear(ch * 8, 1)
        self.embed = nn.Embedding(NUM_CLASSES, ch * 8)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x).sum(dim=(2, 3))
        return self.psi(h).squeeze(1) + (self.embed(y) * h).sum(dim=1)


def init_cond(m: nn.Module) -> None:
    init_weights(m)
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)


def gradient_penalty(critic: ProjectionCritic, real, fake, y) -> torch.Tensor:
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    xhat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grads, = torch.autograd.grad(critic(xhat, y).sum(), xhat, create_graph=True)   # label y fixed
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def class_grid(G: CondGenerator, z: torch.Tensor, device) -> torch.Tensor:
    """Returns (4, n, 1, 64, 64): for each class, images from the SAME latents z (eval-mode BN)."""
    was = G.training
    G.eval()
    outs = [G(z.to(device), torch.full((z.size(0),), c, dtype=torch.long, device=device)).cpu() for c in range(NUM_CLASSES)]
    G.train(was)
    return torch.stack(outs)


def to_grid(x: torch.Tensor, nrow: int):
    return make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2).permute(1, 2, 0).numpy()


def conditioning_stats(imgs: torch.Tensor) -> dict:
    """imgs (4,n,1,64,64). Label effect = mean |G(z,a)-G(z,b)| for a!=b (same z); latent effect = same label, different z."""
    pairs = list(itertools.combinations(range(NUM_CLASSES), 2))
    label = {f"{IDX_TO_CLASS[a]}_vs_{IDX_TO_CLASS[b]}": round((imgs[a] - imgs[b]).abs().mean().item(), 5) for a, b in pairs}
    n = imgs.size(1)
    latent = sum((imgs[c][i] - imgs[c][j]).abs().mean().item()
                 for c in range(NUM_CLASSES) for i in range(n) for j in range(i + 1, n)) / (NUM_CLASSES * n * (n - 1) / 2)
    return {"mean_abs_pixel_diff_same_z_different_label": label,
            "mean_label_effect": round(sum(label.values()) / len(label), 5),
            "mean_abs_pixel_diff_same_label_different_z": round(latent, 5),
            "class_mean_pixel_value": {IDX_TO_CLASS[c]: round(imgs[c].mean().item(), 5) for c in range(NUM_CLASSES)}}


def write_csvs(out_dir: Path, gen_rows, crit_rows) -> None:
    with open(out_dir / "losses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen_step", "critic_loss_mean_of_5", "grad_penalty_mean_of_5", "w_dist_estimate_mean_of_5",
                    "gen_loss", "critic_phase_seconds", "gen_phase_seconds"])
        w.writerows(gen_rows)
    with open(out_dir / "critic_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["critic_update", "gen_step", "critic_loss", "grad_penalty", "w_dist_estimate"])
        w.writerows(crit_rows)


def save_step_grid(imgs: torch.Tensor, path: Path, step: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [IDX_TO_CLASS[c] for c in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.imshow(to_grid(imgs.flatten(0, 1), nrow=imgs.size(1)), cmap="gray")
    ax.set_yticks([2 + 66 * r + 33 for r in range(NUM_CLASSES)], names)
    ax.set_xticks([])
    ax.set_title(f"cWGAN-GP samples, step {step} (rows = class label; columns = the same 8 latents)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_progress(out_dir: Path, steps: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    cols = 3
    rows = (len(steps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, st in zip(axes.flat, steps):
        ax.imshow(mpimg.imread(out_dir / f"samples_step_{st:04d}.png"))
    fig.tight_layout()
    fig.savefig(out_dir / "progress_all_steps.png", dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gen-steps", type=int, default=32, help="TOTAL number of generator updates (target when resuming)")
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--lambda-gp", type=float, default=10.0)
    ap.add_argument("--subset-size", type=int, default=512, help="0 = full training set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sample-every", type=int, default=0, help="periodic mode: save a sample grid every N generator updates (0 = smoke mode)")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="save checkpoint_step_NNNN.pth every N generator updates (periodic mode)")
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from (loaded with weights_only=True)")
    ap.add_argument("--train-list", type=Path, default=None,
                    help="JSON list of archive/Training-relative file paths to train on (e.g. only the CNN training partition)")
    ap.add_argument("--manifest", type=Path, default=None, help="train on the representatives of partition 'train' of a dedup split_manifest.csv instead of archive/Training")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/cwgangp_smoke"))
    args = ap.parse_args()
    periodic = args.sample_every > 0

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        print("ERROR: PYTORCH_ENABLE_MPS_FALLBACK is set; unset it.", file=sys.stderr)
        return 2
    device = torch.device("mps" if args.device in ("auto", "mps") and torch.backends.mps.is_available() else "cpu")
    if args.device == "mps" and device.type != "mps":
        print("MPS requested but unavailable", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    full = ManifestDataset(args.manifest, 'train', 64, True) if args.manifest else MRIDataset(DEFAULT_TRAIN_DIR, image_size=64, validate=True)
    assert args.manifest or all("Training" in Path(p).parts and "Testing" not in Path(p).parts for p, _ in full.samples)
    if full.skipped:
        print(f"WARNING: {len(full.skipped)} unreadable training images skipped")
    train_list_sha = None
    if args.train_list:
        raw = Path(args.train_list).read_bytes()
        train_list_sha = hashlib.sha256(raw).hexdigest()
        wanted = set(json.loads(raw))
        idx = [i for i, (p, _) in enumerate(full.samples) if rel_path(p) in wanted]
        assert len(idx) == len(wanted), f"train list has {len(wanted)} entries but only {len(idx)} were found in archive/Training"
        ds, chosen = Subset(full, idx), [full.samples[i] for i in idx]
    elif args.subset_size and args.subset_size < len(full):
        idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(args.seed))[: args.subset_size].sort().values.tolist()
        ds, chosen = Subset(full, idx), [full.samples[i] for i in idx]
    else:
        ds, chosen = full, full.samples
    subset_hash = hashlib.sha256("\n".join(rel_path(p) for p, _ in chosen).encode()).hexdigest()
    subset_counts = {IDX_TO_CLASS[k]: v for k, v in sorted(Counter(l for _, l in chosen).items())}
    hash_matches = None
    ref = PROJECT_ROOT / "outputs/dcgan_smoke/config.json"
    if args.subset_size == 512 and ref.exists():
        hash_matches = json.loads(ref.read_text())["subset_file_list_sha256"] == subset_hash
        if not hash_matches:
            print("ERROR: subset hash differs from the DCGAN/WGAN-GP smoke subset", file=sys.stderr)
            return 2
    if not args.subset_size and not args.train_list and not args.manifest and (len(chosen) != 5600 or set(subset_counts.values()) != {1400}):
        print(f"ERROR: expected 5,600 training images (1,400 per class), found {len(chosen)} {subset_counts}", file=sys.stderr)
        return 2
    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, generator=loader_gen)

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "image_size": 64, "z_dim": Z_DIM, "emb_dim": EMB_DIM, "num_classes": NUM_CLASSES,
              "class_to_idx": CLASS_TO_IDX, "optimizer": "Adam", "betas": [0.5, 0.9],
              "loss": "cWGAN-GP: L_D = E[D(G(z,y),y)] - E[D(x,y)] + lambda*GP(y fixed); L_G = -E[D(G(z,y),y)]",
              "generator_conditioning": "embedding(50) concatenated with z", "critic_conditioning": "projection (no BatchNorm)",
              "train_dir": str(DEFAULT_TRAIN_DIR), "images_used": len(chosen), "subset_class_counts": subset_counts,
              "subset_file_list_sha256": subset_hash, "subset_hash_matches_dcgan_wgangp_smoke": hash_matches,
              "batches_per_epoch": len(loader), "train_list_sha256": train_list_sha, "torch_version": str(torch.__version__), "device": str(device),
              "classifier_guidance": False, "testing_dir_used": False}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"device={device} images={len(chosen)} {subset_counts} subset_sha={subset_hash[:12]} matches_baselines={hash_matches}")

    G, D = CondGenerator().to(device), ProjectionCritic().to(device)
    G.apply(init_cond)
    D.apply(init_cond)
    opt_g = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.9))
    g0, d0 = snapshot(G), snapshot(D)
    all_z = torch.randn(64, Z_DIM, generator=torch.Generator().manual_seed(args.seed))   # same latents as baselines
    fixed_z = all_z[:8]

    crit_rows, gen_rows, snapshots = [], [], []
    finite, critic_updates, gen_updates = True, 0, 0
    t_critic_total = t_gen_total = 0.0
    elapsed_prev = 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        for k in ("lr", "n_critic", "lambda_gp", "batch_size", "seed"):
            if ck["config"][k] != config[k]:
                print(f"ERROR: resume config mismatch for {k}: {ck['config'][k]} vs {config[k]}", file=sys.stderr)
                return 2
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        gen_updates, critic_updates = ck["gen_updates"], ck["critic_updates"]
        crit_rows, gen_rows, snapshots = [tuple(r) for r in ck["crit_rows"]], [tuple(r) for r in ck["gen_rows"]], ck["snapshots"]
        elapsed_prev, t_critic_total, t_gen_total = ck["elapsed_seconds"], ck["t_critic_total"], ck["t_gen_total"]
        torch.set_rng_state(ck["rng"]["torch"])
        loader_gen.set_state(ck["rng"]["loader"])
        if device.type == "mps" and ck["rng"].get("mps") is not None:
            torch.mps.set_rng_state(ck["rng"]["mps"])
        print(f"resumed from {args.resume} at generator update {gen_updates} (critic updates {critic_updates}); "
              f"note: data order restarts at an epoch boundary, so resumption is not bit-identical")
        g0, d0 = snapshot(G), snapshot(D)   # 'parameters changed' is judged relative to the resumed state

    def take_snapshot(step: int) -> None:
        imgs = class_grid(G, fixed_z, device)
        st = conditioning_stats(imgs)
        save_step_grid(imgs, args.out_dir / f"samples_step_{step:04d}.png", step)
        snapshots.append({"step": step, "mean_label_effect": st["mean_label_effect"],
                          "mean_latent_effect_same_label": st["mean_abs_pixel_diff_same_label_different_z"],
                          "per_label_pair": st["mean_abs_pixel_diff_same_z_different_label"],
                          "class_mean_pixel_value": st["class_mean_pixel_value"],
                          "pixel_min": round(imgs.min().item(), 4), "pixel_max": round(imgs.max().item(), 4)})
        print(f"  [sample step {step}] label_effect={st['mean_label_effect']:.5f} latent_effect={st['mean_abs_pixel_diff_same_label_different_z']:.5f}")

    def save_ckpt(path: Path) -> None:
        save_atomic({"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                     "gen_updates": gen_updates, "critic_updates": critic_updates, "config": config,
                     "crit_rows": [list(r) for r in crit_rows], "gen_rows": [list(r) for r in gen_rows], "snapshots": snapshots,
                     "elapsed_seconds": elapsed_prev + time.perf_counter() - t_start, "t_critic_total": t_critic_total,
                     "t_gen_total": t_gen_total,
                     "rng": {"torch": torch.get_rng_state(), "loader": loader_gen.get_state(),
                             "mps": torch.mps.get_rng_state() if device.type == "mps" else None}}, path)

    stats_before, before_imgs = None, None
    if not args.resume:
        before_imgs = class_grid(G, fixed_z, device)
        stats_before = conditioning_stats(before_imgs)
        if periodic:
            take_snapshot(0)

    data = infinite(loader)
    t_start = time.perf_counter()
    try:
        for g_step in range(gen_updates + 1, args.gen_steps + 1):
            for p in D.parameters():
                p.requires_grad_(True)
            tc = time.perf_counter()
            step_rows = []
            for _ in range(args.n_critic):
                x, y = next(data)
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    fake = G(torch.randn(x.size(0), Z_DIM, device=device), y)     # same labels as the real batch
                d_real, d_fake = D(x, y), D(fake, y)
                gp = gradient_penalty(D, x, fake, y)
                d_loss = d_fake.mean() - d_real.mean() + args.lambda_gp * gp
                opt_d.zero_grad()
                d_loss.backward()
                if not (torch.isfinite(d_loss).item() and all_finite_grads(D)):
                    finite = False
                    raise FloatingPointError(f"non-finite critic loss/grad at gen step {g_step}")
                opt_d.step()
                critic_updates += 1
                step_rows.append((critic_updates, g_step, d_loss.item(), gp.item(), (d_real.mean() - d_fake.mean()).item()))
            sync(device)
            dt_c = time.perf_counter() - tc
            t_critic_total += dt_c
            crit_rows += step_rows

            for p in D.parameters():
                p.requires_grad_(False)
            tg = time.perf_counter()
            yg = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=device)
            g_loss = -D(G(torch.randn(args.batch_size, Z_DIM, device=device), yg), yg).mean()
            opt_g.zero_grad()
            g_loss.backward()
            if not (torch.isfinite(g_loss).item() and all_finite_grads(G)):
                finite = False
                raise FloatingPointError(f"non-finite generator loss/grad at gen step {g_step}")
            opt_g.step()
            sync(device)
            dt_g = time.perf_counter() - tg
            t_gen_total += dt_g
            gen_updates += 1

            mean = lambda i: sum(r[i] for r in step_rows) / len(step_rows)
            gen_rows.append((g_step, mean(2), mean(3), mean(4), g_loss.item(), dt_c, dt_g))
            if g_step == 1 or g_step % (20 if periodic else 8) == 0:
                print(f"gen step {g_step}: L_D={mean(2):.4f} GP={mean(3):.4f} W={mean(4):.4f} L_G={g_loss.item():.4f} "
                      f"[{elapsed_prev + time.perf_counter() - t_start:.0f}s]")
            if periodic and g_step % args.sample_every == 0:
                take_snapshot(g_step)
                write_csvs(args.out_dir, gen_rows, crit_rows)
            if periodic and args.checkpoint_every and g_step % args.checkpoint_every == 0 and g_step != args.gen_steps:
                save_ckpt(args.out_dir / f"checkpoint_step_{g_step:04d}.pth")
    except Exception as exc:
        print(f"TRAINING FAILED at gen step {gen_updates + 1}: {type(exc).__name__}: {exc}", file=sys.stderr)
        write_csvs(args.out_dir, gen_rows, crit_rows)
        (args.out_dir / "metrics.json").write_text(json.dumps({
            "status": "failed", "error": f"{type(exc).__name__}: {exc}", "device": str(device),
            "critic_updates_completed": critic_updates, "generator_updates_completed": gen_updates,
            "all_losses_and_grads_finite": finite}, indent=2))
        return 1
    runtime = elapsed_prev + time.perf_counter() - t_start
    write_csvs(args.out_dir, gen_rows, crit_rows)

    # ---- final samples / checkpoint ----
    names = [IDX_TO_CLASS[c] for c in range(NUM_CLASSES)]
    after = class_grid(G, fixed_z, device)
    stats_after = conditioning_stats(after)
    if periodic:
        if not snapshots or snapshots[-1]["step"] != gen_updates:
            take_snapshot(gen_updates)
        save_progress(args.out_dir, [sn["step"] for sn in snapshots])
        ckpt_path = args.out_dir / f"checkpoint_step_{gen_updates:04d}.pth"
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
        for ax, imgs, title in zip(axes, (before_imgs, after),
                                   ("before training", f"after {gen_updates} generator updates")):
            ax.imshow(to_grid(imgs.flatten(0, 1), nrow=imgs.size(1)), cmap="gray")
            ax.set_title(f"{title}\nrows = {', '.join(f'{i}:{n}' for i, n in enumerate(names))}; columns = same 8 latents")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out_dir / "samples_by_class.png", dpi=120)
        plt.close(fig)
        from dcgan_train import Generator as BaseGenerator
        rows, row_names = [], []
        for name, path in (("DCGAN (unconditional)", "outputs/dcgan_smoke/checkpoint.pth"),
                           ("WGAN-GP (unconditional)", "outputs/wgangp_smoke/checkpoint.pth")):
            bg = BaseGenerator()
            bg.load_state_dict(torch.load(PROJECT_ROOT / path, map_location="cpu", weights_only=True)["G"])
            with torch.no_grad():
                rows.append(bg.eval()(fixed_z))
            row_names.append(name)
        for c in range(NUM_CLASSES):
            rows.append(after[c])
            row_names.append(f"cWGAN-GP label={names[c]}")
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(to_grid(torch.cat(rows), nrow=8), cmap="gray")
        ax.set_title("Same 8 fixed latents; rows: " + " | ".join(row_names), fontsize=7)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out_dir / "fixed_noise_comparison.png", dpi=120)
        plt.close(fig)
        ckpt_path = args.out_dir / "checkpoint.pth"
    (args.out_dir / "samples_metadata.json").write_text(json.dumps({
        "row_labels": {str(i): n for i, n in enumerate(names)},
        "columns": f"latents torch.randn(64,100, Generator(seed={args.seed}))[:8], identical for every row and every snapshot",
        "layout": "4 rows x 8 columns", "snapshot_steps": [sn["step"] for sn in snapshots]}, indent=2))
    save_ckpt(ckpt_path)

    # ---- reload checks ----
    back = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    G2 = CondGenerator().to(device)
    G2.load_state_dict(back["G"])
    ProjectionCritic().load_state_dict(back["D"])
    same_device = bool(torch.allclose(after, class_grid(G2, fixed_z, device), atol=1e-6))
    G3 = CondGenerator()
    G3.load_state_dict(back["G"])
    cpu_max_diff = (class_grid(G3, fixed_z, torch.device("cpu")) - after).abs().max().item()

    gps = [r[3] for r in crit_rows]
    gen_losses = [r[4] for r in gen_rows]
    wd = [r[3] for r in gen_rows]
    metrics = {
        "status": "ok", "device": str(device), "seed": args.seed, "image_size": 64, "batch_size": args.batch_size,
        "images_used": len(chosen), "subset_file_list_sha256": subset_hash,
        "subset_hash_matches_dcgan_wgangp_smoke": hash_matches, "dataset_class_counts": subset_counts,
        "n_critic": args.n_critic, "lambda_gp": args.lambda_gp, "resumed_from": str(args.resume) if args.resume else None,
        "critic_updates_completed": critic_updates, "generator_updates_completed": gen_updates,
        "real_image_batches_consumed": critic_updates, "real_images_seen": critic_updates * args.batch_size,
        "approx_epochs_of_real_data": round(critic_updates * args.batch_size / len(chosen), 3),
        "runtime_seconds": round(runtime, 3),
        "seconds_per_generator_update_incl_5_critic_updates": round(runtime / gen_updates, 4),
        "avg_seconds_per_critic_update": round(t_critic_total / critic_updates, 4),
        "avg_seconds_per_generator_phase": round(t_gen_total / gen_updates, 4),
        "initial_critic_loss": crit_rows[0][2], "final_critic_loss": crit_rows[-1][2],
        "initial_gen_loss": gen_rows[0][4], "final_gen_loss": gen_rows[-1][4],
        "gen_loss_min": min(gen_losses), "gen_loss_max": max(gen_losses),
        "initial_w_dist_estimate": crit_rows[0][4], "final_w_dist_estimate": crit_rows[-1][4],
        "w_dist_estimate_per_gen_step_max": max(wd), "w_dist_estimate_last_gen_step": wd[-1],
        "grad_penalty": {"min": min(gps), "mean": sum(gps) / len(gps), "max": max(gps), "last": gps[-1]},
        "all_losses_and_grads_finite": finite,
        "generator_params_changed": params_changed(g0, G), "critic_params_changed": params_changed(d0, D),
        "checkpoint": str(ckpt_path), "checkpoint_loads_with_weights_only": True,
        "reloaded_checkpoint_reproduces_samples_same_device_atol_1e-6": same_device,
        "reloaded_on_cpu_max_abs_diff_vs_mps_samples": cpu_max_diff,
        "conditioning_before_training": stats_before, "conditioning_after_training": stats_after,
        "conditioning_snapshots": snapshots,
        "trained_on_testing_dir": False, "mps_fallback_env_set": False, "classifier_guidance": False,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("conditioning_snapshots",)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
