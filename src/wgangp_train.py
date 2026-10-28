"""WGAN-GP baseline (64x64 grayscale MRI). Trains on archive/Training only.

    python src/wgangp_train.py --subset-size 512 --gen-steps 32 --out-dir outputs/wgangp_smoke

Loss sign convention (both are MINIMISED):
  critic    L_D = E[D(G(z))] - E[D(x)] + lambda * E[(||grad_xhat D(xhat)||_2 - 1)^2]
            (so -(E[D(x)] - E[D(G(z))]) is the negative Wasserstein-distance estimate; w_dist = E[D(x)] - E[D(G(z))])
  generator L_G = -E[D(G(z))]
n_critic=5 critic updates per generator update, lambda=10, Adam(lr=1e-4, betas=(0.5, 0.9)), z_dim=100.
Critic has no BatchNorm. Generator and weight init reuse the DCGAN modules. No CPU fallback is enabled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data_pipeline import DEFAULT_TRAIN_DIR, IDX_TO_CLASS, ManifestDataset, MRIDataset, rel_path
from dcgan_train import (Generator, Z_DIM, all_finite_grads, init_weights, params_changed, sample_grid,
                         save_atomic, save_before_after, save_grid, set_seed, snapshot)
from smoke_test_wgangp_mps import Critic, gradient_penalty


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def infinite(loader: DataLoader):
    while True:
        yield from loader


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gen-steps", type=int, default=32, help="number of generator updates")
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--lambda-gp", type=float, default=10.0)
    ap.add_argument("--subset-size", type=int, default=512, help="0 = full training set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sample-every", type=int, default=0, help="save samples_step_NNNN.png (8x8 fixed-latent grid) every N generator updates (0 = off)")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="save checkpoint_step_NNNN.pth every N generator updates (0 = off)")
    ap.add_argument("--manifest", type=Path, default=None, help="train on the representatives of partition 'train' of a dedup split_manifest.csv instead of archive/Training")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/wgangp_smoke"))
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

    # identical subset selection to dcgan_train.py (same seed, same randperm)
    full = ManifestDataset(args.manifest, 'train', 64, True) if args.manifest else MRIDataset(DEFAULT_TRAIN_DIR, image_size=64, validate=True)
    assert args.manifest or all("Training" in Path(p).parts and "Testing" not in Path(p).parts for p, _ in full.samples)
    if args.subset_size and args.subset_size < len(full):
        idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(args.seed))[: args.subset_size].sort().values.tolist()
        ds, chosen = Subset(full, idx), [full.samples[i] for i in idx]
    else:
        ds, chosen = full, full.samples
    subset_hash = hashlib.sha256("\n".join(rel_path(p) for p, _ in chosen).encode()).hexdigest()
    subset_counts = {IDX_TO_CLASS[k]: v for k, v in sorted(Counter(l for _, l in chosen).items())}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "image_size": 64, "z_dim": Z_DIM, "optimizer": "Adam", "betas": [0.5, 0.9],
              "loss": "WGAN-GP: L_D = E[D(fake)] - E[D(real)] + lambda*GP ; L_G = -E[D(fake)]",
              "critic_batchnorm": False, "weight_init": "N(0,0.02)", "train_dir": str(DEFAULT_TRAIN_DIR),
              "images_used": len(chosen), "subset_class_counts": subset_counts,
              "subset_file_list_sha256": subset_hash, "torch_version": str(torch.__version__),
              "device": str(device), "conditional": False}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"device={device} images={len(chosen)} {subset_counts} subset_sha={subset_hash[:12]}")

    G, D = Generator().to(device), Critic().to(device)
    G.apply(init_weights)
    D.apply(init_weights)
    opt_g = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.9))
    g0, d0 = snapshot(G), snapshot(D)
    fixed_z = torch.randn(64, Z_DIM, generator=torch.Generator().manual_seed(args.seed)).to(device)  # same as DCGAN
    grid_before = sample_grid(G, fixed_z)
    if args.sample_every:
        save_grid(grid_before, args.out_dir / "samples_step_0000.png", "WGAN-GP, step 0 (fixed latents)")

    data = infinite(loader)
    crit_rows, gen_rows = [], []
    finite, critic_updates, gen_updates = True, 0, 0
    t_critic_total = t_gen_total = 0.0
    t_start = time.perf_counter()
    try:
        for g_step in range(1, args.gen_steps + 1):
            # ---- critic phase ----
            for p in D.parameters():
                p.requires_grad_(True)
            tc = time.perf_counter()
            step_rows = []
            for _ in range(args.n_critic):
                x, _ = next(data)
                x = x.to(device)
                b = x.size(0)
                with torch.no_grad():
                    fake = G(torch.randn(b, Z_DIM, device=device))
                d_real, d_fake = D(x), D(fake)
                gp = gradient_penalty(D, x, fake)
                d_loss = d_fake.mean() - d_real.mean() + args.lambda_gp * gp
                opt_d.zero_grad()
                d_loss.backward()
                if not (torch.isfinite(d_loss).item() and all_finite_grads(D)):
                    finite = False
                    raise FloatingPointError(f"non-finite critic loss/grad at gen step {g_step}")
                opt_d.step()
                critic_updates += 1
                step_rows.append((critic_updates, g_step, d_loss.item(), gp.item(),
                                  (d_real.mean() - d_fake.mean()).item()))
            sync(device)
            dt_c = time.perf_counter() - tc
            t_critic_total += dt_c
            crit_rows += step_rows

            # ---- generator phase ----
            for p in D.parameters():
                p.requires_grad_(False)
            tg = time.perf_counter()
            g_loss = -D(G(torch.randn(args.batch_size, Z_DIM, device=device))).mean()
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
            if g_step == 1 or g_step % (100 if args.sample_every else 8) == 0:
                print(f"gen step {g_step}: L_D={mean(2):.4f} GP={mean(3):.4f} W={mean(4):.4f} "
                      f"L_G={g_loss.item():.4f} [{time.perf_counter() - t_start:.0f}s]")
            if args.sample_every and g_step % args.sample_every == 0:
                save_grid(sample_grid(G, fixed_z), args.out_dir / f"samples_step_{g_step:04d}.png", f"WGAN-GP, step {g_step} (fixed latents)")
            if args.checkpoint_every and g_step % args.checkpoint_every == 0:
                save_atomic({"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(),
                             "opt_d": opt_d.state_dict(), "gen_updates": g_step, "critic_updates": critic_updates,
                             "config": config, "elapsed_seconds": time.perf_counter() - t_start},
                            args.out_dir / f"checkpoint_step_{g_step:04d}.pth")
    except Exception as exc:
        print(f"TRAINING FAILED at gen step {gen_updates + 1}: {type(exc).__name__}: {exc}", file=sys.stderr)
        (args.out_dir / "metrics.json").write_text(json.dumps({
            "status": "failed", "error": f"{type(exc).__name__}: {exc}", "device": str(device),
            "critic_updates_completed": critic_updates, "generator_updates_completed": gen_updates,
            "all_losses_and_grads_finite": finite}, indent=2))
        return 1
    runtime = time.perf_counter() - t_start

    with open(args.out_dir / "losses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen_step", "critic_loss_mean_of_5", "grad_penalty_mean_of_5", "w_dist_estimate_mean_of_5",
                    "gen_loss", "critic_phase_seconds", "gen_phase_seconds"])
        w.writerows(gen_rows)
    with open(args.out_dir / "critic_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["critic_update", "gen_step", "critic_loss", "grad_penalty", "w_dist_estimate"])
        w.writerows(crit_rows)

    save_before_after(grid_before, sample_grid(G, fixed_z), args.out_dir / "samples.png",
                      f"after {gen_updates} generator / {critic_updates} critic updates")
    save_atomic({"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(),
                 "opt_d": opt_d.state_dict(), "gen_updates": gen_updates, "critic_updates": critic_updates,
                 "config": config}, args.out_dir / "checkpoint.pth")

    back = torch.load(args.out_dir / "checkpoint.pth", map_location="cpu", weights_only=True)
    G2 = Generator()
    G2.load_state_dict(back["G"])
    Critic().load_state_dict(back["D"])
    G.eval()
    G2.eval()
    with torch.no_grad():
        roundtrip = bool(torch.allclose(G2(fixed_z.cpu()), G(fixed_z).cpu(), atol=1e-4))

    gps = [r[3] for r in crit_rows]
    metrics = {
        "status": "ok", "device": str(device), "seed": args.seed, "image_size": 64, "batch_size": args.batch_size,
        "images_used": len(chosen), "subset_file_list_sha256": subset_hash, "subset_class_counts": subset_counts,
        "n_critic": args.n_critic, "lambda_gp": args.lambda_gp,
        "critic_updates_completed": critic_updates, "generator_updates_completed": gen_updates,
        "runtime_seconds": round(runtime, 3),
        "seconds_per_generator_update_incl_5_critic_updates": round(runtime / gen_updates, 4),
        "avg_seconds_per_critic_update": round(t_critic_total / critic_updates, 4),
        "avg_seconds_per_generator_phase": round(t_gen_total / gen_updates, 4),
        "initial_critic_loss": crit_rows[0][2], "final_critic_loss": crit_rows[-1][2],
        "initial_gen_loss": gen_rows[0][4], "final_gen_loss": gen_rows[-1][4],
        "initial_w_dist_estimate": crit_rows[0][4], "final_w_dist_estimate": crit_rows[-1][4],
        "grad_penalty": {"min": min(gps), "mean": sum(gps) / len(gps), "max": max(gps), "last": gps[-1]},
        "all_losses_and_grads_finite": finite,
        "generator_params_changed": params_changed(g0, G), "critic_params_changed": params_changed(d0, D),
        "checkpoint_loads_with_weights_only": True, "checkpoint_reload_matches_samples": roundtrip,
        "trained_on_testing_dir": False, "mps_fallback_env_set": False,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
