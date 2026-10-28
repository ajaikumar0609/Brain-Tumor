#!/usr/bin/env python3
"""FID/KID of the three GANs retrained on the de-duplicated TRAIN partition, against the cluster-disjoint TEST partition.

    python src/dedup_eval_fid_kid.py [--smoke --device cpu --out-dir <scratch>]

Same implementation as src/eval_gan_fid_kid.py (torch-fidelity 0.4.0, 'inception-v3-compat' pool3 2048-d features, identical preprocessing for real and
generated images: grayscale -> 64x64 uint8 -> 3 channels). Reference = test-partition representatives (n=492; class counts 196/122/46/128).
Generated: unconditional models n=492 per seed; cWGAN-GP class counts equal to the test class counts. 3 generation seeds (1001-1003).
Floor: class-matched random draws of TRAIN representatives vs test (3 draws). KID uses 100 subsets (size 300 overall, <=100 per class).
The test partition is used for nothing else here (no checkpoint selection or tuning)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import eval_gan_fid_kid as ev
import dedup_downstream as dd
from cwgangp_train import CondGenerator
from data_pipeline import ManifestDataset, rel_path
from dcgan_train import Generator

ROOT = Path(__file__).resolve().parent.parent
DEXP = ROOT / "outputs/deduplicated_experiment"
CLASSES = ev.CLASSES
SEEDS = [1001, 1002, 1003]
FLOOR_SEEDS = [2001, 2002, 2003]


@torch.no_grad()
def gen(model: str, ck: dict, seed: int, counts: list[int], device) -> tuple[torch.Tensor, torch.Tensor | None]:
    g = torch.Generator().manual_seed(seed)
    if model == "cWGAN-GP":
        G = CondGenerator()
        G.load_state_dict(ck["G"])
        G = G.to(device).eval()
        y = torch.cat([torch.full((n,), c, dtype=torch.long) for c, n in enumerate(counts)])
        z = torch.randn(len(y), ev.Z_DIM, generator=g)
        out = torch.cat([G(z[i:i + 200].to(device), y[i:i + 200].to(device)).cpu() for i in range(0, len(y), 200)])
        return ev.to_uint8(out), y
    G = Generator()
    G.load_state_dict(ck["G"])
    G = G.to(device).eval()
    n = int(sum(counts))
    z = torch.randn(n, ev.Z_DIM, generator=g)
    return ev.to_uint8(torch.cat([G(z[i:i + 200].to(device)).cpu() for i in range(0, n, 200)])), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEXP / "split_manifest.csv")
    ap.add_argument("--gan-dir", type=Path, default=DEXP / "gan")
    ap.add_argument("--out-dir", type=Path, default=DEXP / "gan_comparison")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = ap.parse_args()
    if (args.out_dir / "metrics.json").exists():
        print("ERROR: metrics.json exists; refusing to overwrite", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    names = {"DCGAN": "dcgan", "WGAN-GP": "wgangp", "cWGAN-GP": "cwgangp"}
    cks, info = {}, {}
    recomputed = hashlib.sha256("\n".join(rel_path(p) for p, _ in ManifestDataset(args.manifest, "train", 64, False).samples).encode()).hexdigest()
    for m, d in names.items():
        p = args.gan_dir / d / "checkpoint_step_3000.pth"
        ck = torch.load(p, map_location="cpu", weights_only=True)
        gu = ck.get("gen_updates", ck.get("batches"))
        cu = ck.get("critic_updates", ck.get("batches"))
        h = ck["config"].get("subset_file_list_sha256")
        if not args.smoke:
            assert gu == 3000, f"{m}: {gu} generator updates"
            assert h == recomputed, f"{m}: trained on a different file list than the manifest's train partition"
        cks[m], info[m] = ck, {"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "generator_updates": gu, "discriminator_or_critic_updates": cu, "train_file_list_sha256": h,
                               "images_used": ck["config"].get("images_used"), "matches_manifest_train_partition": h == recomputed}
    # reference + floor
    xte, yte, fte = dd.load_partition(args.manifest, "test", 6 if args.smoke else None)
    counts = np.bincount(yte.numpy(), minlength=4).tolist()
    xtr_all, ytr_all, ftr_all = dd.load_partition(args.manifest, "train", 12 if args.smoke else None)
    assert not (set(fte) & set(ftr_all))
    cpu = torch.device("cpu")
    device = torch.device("mps" if args.device in ("auto", "mps") and torch.backends.mps.is_available() else "cpu")
    ex_cpu = ev.FeatureExtractorInceptionV3("inception-v3-compat", ["2048"], feature_extractor_weights_path=str(ev.WEIGHTS))
    extractor, dev_note = ex_cpu, "cpu"
    if device.type == "mps":
        ex_mps = ev.FeatureExtractorInceptionV3("inception-v3-compat", ["2048"], feature_extractor_weights_path=str(ev.WEIGHTS)).to(device)
        a, b = ev.features(ex_cpu, xte[:32], cpu), ev.features(ex_mps, xte[:32], device)
        rel = ((a - b).abs().max() / a.abs().max()).item()
        print(f"MPS vs CPU feature max relative difference {rel:.2e}")
        if rel < 1e-3:
            extractor, dev_note = ex_mps, "mps"
    dev = torch.device(dev_note)
    f_real = ev.features(extractor, xte, dev)
    n_ref = len(f_real)
    ks = min(300, n_ref - 1) if not args.smoke else max(2, n_ref // 2)
    print(f"reference n={n_ref} class counts {counts}; KID subset {ks}; feature device {dev_note}", flush=True)

    rows, feats = [], {}
    for s in FLOOR_SEEDS:                                    # class-matched real-train draws vs test
        rng = np.random.RandomState(s)
        pick = np.concatenate([rng.choice(np.where(ytr_all.numpy() == c)[0], min(counts[c], int((ytr_all == c).sum())), replace=False) for c in range(4)])
        f = ev.features(extractor, xtr_all[pick], dev)
        k, kstd = ev.kid(f, f_real, min(ks, len(f) - 1), seed=s)
        rows.append({"model": "Real train reps (class-matched, reference)", "seed": s, "n_generated": len(f), "n_real": n_ref, "fid": ev.fid(f, f_real), "kid_mean": k, "kid_std_subsets": kstd})
        print(rows[-1], flush=True)
    for m in names:
        feats[m] = []
        for s in SEEDS:
            u8, y = gen(m, cks[m], s, counts, device)
            f = ev.features(extractor, u8, dev)
            feats[m].append((f, y))
            k, kstd = ev.kid(f, f_real, ks, seed=s)
            rows.append({"model": m, "seed": s, "n_generated": len(f), "n_real": n_ref, "fid": ev.fid(f, f_real), "kid_mean": k, "kid_std_subsets": kstd})
            print(rows[-1], flush=True)
    agg = {}
    for m in ["Real train reps (class-matched, reference)"] + list(names):
        r = [x for x in rows if x["model"] == m]
        fv, kv = np.array([x["fid"] for x in r]), np.array([x["kid_mean"] for x in r])
        agg[m] = {"fid_mean": float(fv.mean()), "fid_std_across_seeds": float(fv.std(ddof=1)), "kid_mean": float(kv.mean()), "kid_std_across_seeds": float(kv.std(ddof=1)),
                  "kid_mean_within_seed_subset_std": float(np.mean([x["kid_std_subsets"] for x in r])), "n_seeds": len(r), "n_generated_per_seed": r[0]["n_generated"], "n_real": n_ref}
    with open(args.out_dir / "fid_kid_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # per-class + confusion for the conditional model
    conf = np.zeros((4, 4))
    pc = {c: [] for c in range(4)}
    for f, y in feats["cWGAN-GP"]:
        for c in range(4):
            fg = f[y == c]
            for j in range(4):
                fr = f_real[yte == j]
                sub = max(2, min(100, int(0.8 * min(len(fg), len(fr)))))
                conf[c, j] += ev.kid(fg, fr, sub, seed=c * 10 + j)[0] / len(SEEDS)
            fr = f_real[yte == c]
            sub = max(2, min(100, int(0.8 * min(len(fg), len(fr)))))
            k, kstd = ev.kid(fg, fr, sub, seed=c)
            pc[c].append({"fid": ev.fid(fg, fr), "kid": k, "sub": sub})
    prow = [{"class": CLASSES[c], "n_generated_per_seed": counts[c], "n_real": counts[c], "fid_mean": float(np.mean([x["fid"] for x in pc[c]])), "fid_std": float(np.std([x["fid"] for x in pc[c]], ddof=1)),
             "kid_mean": float(np.mean([x["kid"] for x in pc[c]])), "kid_std": float(np.std([x["kid"] for x in pc[c]], ddof=1)), "kid_subset_size": pc[c][0]["sub"],
             "reliability_note": "n<100: FID/KID indicative only" if counts[c] < 100 else "n>=100 (still small for 2048-d FID)",
             "own_class_is_lowest_kid_row_wise": bool(conf[c].argmin() == c), "own_label_is_lowest_kid_column_wise": bool(conf[:, c].argmin() == c)} for c in range(4)]
    with open(args.out_dir / "per_class_fid_kid_cwgangp.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    with open(args.out_dir / "class_confusion_kid_cwgangp.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generated_label \\ real_test_class"] + CLASSES)
        for c in range(4):
            w.writerow([CLASSES[c]] + [f"{v:.6f}" for v in conf[c]])
    cfg = {"implementation": "torch-fidelity 0.4.0 inception-v3-compat pool3", "inception_weights_sha256": hashlib.sha256(ev.WEIGHTS.read_bytes()).hexdigest(), "feature_device": dev_note,
           "reference": "test-partition representatives of the cluster-disjoint split", "n_ref": n_ref, "test_class_counts": counts, "generation_seeds": SEEDS, "floor_seeds": FLOOR_SEEDS,
           "kid": {"subsets": 100, "subset_size_overall": ks, "kernel": "poly deg 3", "per_class_subset": "min(100, 0.8*min(n_gen, n_real))"}, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
           "manifest_train_hash": recomputed, "checkpoints": info, "uncertainty": "std across 3 generation seeds only; one training run per GAN; single reference set",
           "generator_mode": "eval()", "smoke": args.smoke, "eval_seconds": round(time.perf_counter() - t0, 1)}
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (args.out_dir / "metrics.json").write_text(json.dumps({"aggregate": agg, "per_class_cwgangp": prow, "class_confusion_kid_cwgangp": {"rows": CLASSES, "cols": CLASSES, "kid": conf.tolist()}}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    order = list(agg)
    short = ["Real train\n(reference)", "DCGAN", "WGAN-GP", "cWGAN-GP"]
    colors = ["#9aa0a6", "#4c78a8", "#f58518", "#54a24b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, sc, title in ((axes[0], "fid", 1, "FID (lower better)"), (axes[1], "kid", 1000, "KID x1000 (lower better)")):
        mean = [agg[k][f"{key}_mean"] * sc for k in order]
        std = [agg[k][f"{key}_std_across_seeds"] * sc for k in order]
        bars = ax.bar(short, mean, yerr=std, capsize=4, color=colors)
        for b, v in zip(bars, mean):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"De-duplicated split: vs test partition (n={n_ref}); error bars = std over 3 generation seeds; 1 training run per GAN", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fid_kid_bar_chart.png", dpi=130)
    plt.close(fig)
    panels = {"Real test images (rows: glioma, meningioma, notumor, pituitary)": torch.cat([xte[(yte == c).nonzero().squeeze(1)[:8]] for c in range(4)])}
    for m in names:
        u8, y = gen(m, cks[m], 1001, counts, device)
        panels[m + (" (rows: glioma, meningioma, notumor, pituitary)" if y is not None else " (unconditional)")] = (torch.cat([u8[(y == c).nonzero().squeeze(1)[:8]] for c in range(4)]) if y is not None else u8[:32])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4))
    for ax, (t, u) in zip(axes.flat, panels.items()):
        ax.imshow(make_grid(u.float() / 255, nrow=8, padding=2).permute(1, 2, 0).numpy(), cmap="gray")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "sample_grids.png", dpi=120)
    print(json.dumps(agg, indent=1))
    print(f"done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
