"""FID / KID comparison of DCGAN, WGAN-GP and cWGAN-GP (3,000 generator updates each).

    python src/eval_gan_fid_kid.py [--device auto|mps|cpu]

Implementation: torch-fidelity 0.4.0, feature extractor 'inception-v3-compat' (the standard FID Inception-v3 weights
from the 2015-12-05 TensorFlow release, pool3 2048-d features, TF-style bilinear resize to 299x299, float32).
Features are extracted once and fed to torch-fidelity's own functions:
    FID = fid_features_to_statistics + fid_statistics_to_metric
    KID = kid_features_to_metric (polynomial kernel, degree 3, gamma=1/d, coef0=1, unbiased MMD^2, repeated subsets).
Reference set: archive/Testing (read ONLY here, never during GAN training). Real and generated images go through the
same pipeline: 64x64 grayscale uint8 -> replicated to 3 channels -> Inception. No checkpoint selection or tuning uses it.
"""
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
from PIL import Image
from torchvision import transforms

import torch_fidelity
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
from torch_fidelity.metric_kid import kid_features_to_metric

from cwgangp_train import CondGenerator
from dcgan_train import Generator

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
WEIGHTS = ROOT / "external_models/inception_fid_weights/weights-inception-2015-12-05-6726825d.pth"
CKPTS = {"DCGAN": ROOT / "outputs/dcgan_3000/checkpoint_step_3000.pth",
         "WGAN-GP": ROOT / "outputs/wgangp_3000/checkpoint_step_3000.pth",
         "cWGAN-GP": ROOT / "outputs/cwgangp_3000/checkpoint_step_3000.pth"}
GEN_SEEDS = [1001, 1002, 1003]          # distinct from the training seed (42)
TRAIN_REF_SEEDS = [2001, 2002, 2003]
Z_DIM = 100
TF = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((64, 64)), transforms.ToTensor()])


def load_split(split_dir: Path, per_class: int | None = None, seed: int | None = None):
    """Returns uint8 images (N,1,64,64) and labels. per_class=None -> all images."""
    imgs, labels = [], []
    for c, name in enumerate(CLASSES):
        files = sorted(p for p in (split_dir / name).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if per_class is not None:
            rng = np.random.RandomState(seed)
            files = [files[i] for i in sorted(rng.choice(len(files), per_class, replace=False))]
        for p in files:
            with Image.open(p) as im:
                imgs.append((TF(im) * 255).round().to(torch.uint8))
            labels.append(c)
    return torch.stack(imgs), torch.tensor(labels)


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1,1] -> uint8 in the same way real images are stored (round to nearest level)."""
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)


@torch.no_grad()
def features(extractor, imgs_u8: torch.Tensor, device, bs: int = 100) -> torch.Tensor:
    out = []
    for i in range(0, len(imgs_u8), bs):
        x = imgs_u8[i:i + bs].repeat(1, 3, 1, 1).to(device)
        out.append(extractor(x)[0].float().cpu())
    return torch.cat(out)


def fid(f1, f2) -> float:
    return fid_statistics_to_metric(fid_features_to_statistics(f1), fid_features_to_statistics(f2), False)["frechet_inception_distance"]


def kid(f1, f2, subset_size: int, subsets: int = 100, seed: int = 0):
    r = kid_features_to_metric(f1, f2, kid_subsets=subsets, kid_subset_size=subset_size, verbose=False, rng_seed=seed,
                               kid_kernel="poly", kid_kernel_poly_degree=3, kid_kernel_poly_gamma=None, kid_kernel_poly_coef0=1)
    return r["kernel_inception_distance_mean"], r["kernel_inception_distance_std"]


@torch.no_grad()
def generate(model: str, ckpt_path: Path, seed: int, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    g_cpu = torch.Generator().manual_seed(seed)
    if model == "cWGAN-GP":
        G = CondGenerator()
        G.load_state_dict(ck["G"])
        G = G.to(device).eval()
        y = torch.arange(4).repeat_interleave(400)                      # 400 per class (test-set distribution)
        z = torch.randn(len(y), Z_DIM, generator=g_cpu)
        out = torch.cat([G(z[i:i + 200].to(device), y[i:i + 200].to(device)).cpu() for i in range(0, len(y), 200)])
        return to_uint8(out), y
    G = Generator()
    G.load_state_dict(ck["G"])
    G = G.to(device).eval()
    z = torch.randn(1600, Z_DIM, generator=g_cpu)
    out = torch.cat([G(z[i:i + 200].to(device)).cpu() for i in range(0, 1600, 200)])
    return to_uint8(out), None


def check_counters() -> dict:
    info = {}
    for m, p in CKPTS.items():
        if not p.exists():
            raise SystemExit(f"missing checkpoint: {p}")
        ck = torch.load(p, map_location="cpu", weights_only=True)
        gen = ck["batches"] if m == "DCGAN" else ck["gen_updates"]
        crit = ck["batches"] if m == "DCGAN" else ck["critic_updates"]
        if gen != 3000:
            raise SystemExit(f"{m}: checkpoint has {gen} generator updates, expected 3000")
        info[m] = {"path": str(p.relative_to(ROOT)), "generator_updates": gen, "discriminator_or_critic_updates": crit,
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/gan_comparison")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    ck_info = check_counters()
    w_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    assert w_sha.startswith("6726825d"), "Inception weights hash mismatch"
    torch.load(WEIGHTS, map_location="cpu", weights_only=True)            # plain tensors only
    cpu = torch.device("cpu")
    device = torch.device("mps" if args.device in ("auto", "mps") and torch.backends.mps.is_available() else "cpu")
    ex_cpu = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"], feature_extractor_weights_path=str(WEIGHTS))
    extractor, dev_note = ex_cpu, "cpu"
    real_te, y_te = load_split(ROOT / "archive/Testing")
    assert len(real_te) == 1600 and torch.bincount(y_te).tolist() == [400] * 4
    if device.type == "mps":                                              # sanity-check MPS vs CPU features
        ex_mps = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"], feature_extractor_weights_path=str(WEIGHTS)).to(device)
        a, b = features(ex_cpu, real_te[:32], cpu), features(ex_mps, real_te[:32], device)
        rel = ((a - b).abs().max() / a.abs().max()).item()
        print(f"MPS vs CPU feature max relative difference: {rel:.2e}")
        if rel < 1e-3:
            extractor, dev_note = ex_mps, "mps"
        else:
            print("MPS features differ too much from CPU; using CPU")
    dev = torch.device(dev_note)
    print(f"feature extractor device: {dev_note}")

    f_real = features(extractor, real_te, dev)
    print(f"real test features {tuple(f_real.shape)}")

    rows, per_class_rows, aggregates, gen_feats = [], [], {}, {}
    # real-training reference (class-balanced 400/class) vs test: the real-vs-real floor
    for s in TRAIN_REF_SEEDS:
        imgs, _ = load_split(ROOT / "archive/Training", per_class=400, seed=s)
        f = features(extractor, imgs, dev)
        k, ks = kid(f, f_real, 1000, seed=s)
        rows.append({"model": "Real train (class-balanced, reference)", "seed": s, "n_generated": 1600, "n_real": 1600,
                     "fid": fid(f, f_real), "kid_mean": k, "kid_std_subsets": ks})
        print(rows[-1])
    for m in CKPTS:
        gen_feats[m] = []
        for s in GEN_SEEDS:
            imgs, y = generate(m, CKPTS[m], s, device)
            f = features(extractor, imgs, dev)
            gen_feats[m].append((f, y))
            k, ks = kid(f, f_real, 1000, seed=s)
            rows.append({"model": m, "seed": s, "n_generated": len(f), "n_real": len(f_real),
                         "fid": fid(f, f_real), "kid_mean": k, "kid_std_subsets": ks})
            print(rows[-1])

    for m in ["Real train (class-balanced, reference)"] + list(CKPTS):
        r = [x for x in rows if x["model"] == m]
        fv, kv = np.array([x["fid"] for x in r]), np.array([x["kid_mean"] for x in r])
        aggregates[m] = {"fid_mean": float(fv.mean()), "fid_std_across_seeds": float(fv.std(ddof=1)),
                         "kid_mean": float(kv.mean()), "kid_std_across_seeds": float(kv.std(ddof=1)),
                         "kid_mean_within_seed_subset_std": float(np.mean([x["kid_std_subsets"] for x in r])),
                         "n_seeds": len(r), "n_generated_per_seed": r[0]["n_generated"], "n_real": r[0]["n_real"]}
    with open(args.out_dir / "fid_kid_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- per-class and class-confusion (conditional model only) ----
    conf = np.zeros((4, 4))
    per_class_seed = {c: [] for c in range(4)}
    for f, y in gen_feats["cWGAN-GP"]:
        for c in range(4):
            fg = f[y == c]
            for j in range(4):
                fr = f_real[y_te == j]
                k, _ = kid(fg, fr, 200, seed=c * 10 + j)
                conf[c, j] += k / len(GEN_SEEDS)
            fr = f_real[y_te == c]
            k, ks = kid(fg, fr, 200, seed=c)
            per_class_seed[c].append({"fid": fid(fg, fr), "kid_mean": k, "kid_std_subsets": ks})
    for c in range(4):
        v = per_class_seed[c]
        per_class_rows.append({"class": CLASSES[c], "n_generated_per_seed": 400, "n_real": 400, "n_seeds": len(v),
                               "fid_mean": float(np.mean([x["fid"] for x in v])),
                               "fid_std_across_seeds": float(np.std([x["fid"] for x in v], ddof=1)),
                               "kid_mean": float(np.mean([x["kid_mean"] for x in v])),
                               "kid_std_across_seeds": float(np.std([x["kid_mean"] for x in v], ddof=1)),
                               "kid_subset_size": 200,
                               "kid_to_own_real_class_is_lowest_of_4": bool(conf[c].argmin() == c)})
    with open(args.out_dir / "per_class_fid_kid_cwgangp.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class_rows[0]))
        w.writeheader()
        w.writerows(per_class_rows)
    with open(args.out_dir / "class_confusion_kid_cwgangp.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generated_label \\ real_test_class"] + CLASSES)
        for c in range(4):
            w.writerow([CLASSES[c]] + [f"{v:.6f}" for v in conf[c]])

    # ---- config / metrics ----
    config = {"implementation": f"torch-fidelity {torch_fidelity.__version__}", "feature_extractor": "inception-v3-compat",
              "feature_layer": "2048 (pool3)", "inception_weights": str(WEIGHTS.relative_to(ROOT)), "inception_weights_sha256": w_sha,
              "feature_dtype": "float32", "feature_device": dev_note, "resize_to_inception": "TF-compatible bilinear 64->299",
              "image_preprocessing": "grayscale -> Resize(64,64) -> uint8 -> repeat to 3 channels (identical for real and generated)",
              "generated_to_uint8": "clamp [-1,1] -> (x+1)/2*255 -> round", "generator_mode": "eval() (BatchNorm running statistics)",
              "real_reference": "archive/Testing, 1600 images (400/class), all used, read only by this script",
              "generated_samples": {"DCGAN": "1600 unconditional per seed", "WGAN-GP": "1600 unconditional per seed",
                                    "cWGAN-GP": "1600 per seed = 400 per label (glioma, meningioma, notumor, pituitary)"},
              "generation_seeds": GEN_SEEDS, "training_seed": 42,
              "fid": "torch-fidelity fid_features_to_statistics + fid_statistics_to_metric, full 1600 vs 1600",
              "kid": {"subsets": 100, "subset_size_overall": 1000, "subset_size_per_class": 200, "kernel": "poly(degree=3, gamma=1/d, coef0=1)", "estimator": "unbiased MMD^2"},
              "uncertainty": "std across the 3 generation seeds (ddof=1) and mean within-seed KID subset std; no uncertainty for the single reference set",
              "real_train_reference": {"description": "class-balanced random 400/class draws from archive/Training vs the test set", "seeds": TRAIN_REF_SEEDS},
              "checkpoints": ck_info, "torch_version": str(torch.__version__), "total_eval_seconds": round(time.perf_counter() - t0, 1)}
    (args.out_dir / "comparison_config.json").write_text(json.dumps(config, indent=2))
    (args.out_dir / "metrics.json").write_text(json.dumps({"aggregate": aggregates, "per_class_cwgangp": per_class_rows,
                                                            "class_confusion_kid_cwgangp": {"rows": CLASSES, "cols": CLASSES, "kid": conf.tolist()}}, indent=2))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    names = list(aggregates)
    short = ["Real train\n(reference)", "DCGAN", "WGAN-GP", "cWGAN-GP"]
    colors = ["#9aa0a6", "#4c78a8", "#f58518", "#54a24b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in ((axes[0], "fid", "FID (lower is better)"), (axes[1], "kid", "KID x1000 (lower is better)")):
        mean = [aggregates[n][f"{key}_mean"] * (1000 if key == "kid" else 1) for n in names]
        std = [aggregates[n][f"{key}_std_across_seeds"] * (1000 if key == "kid" else 1) for n in names]
        bars = ax.bar(short, mean, yerr=std, capsize=4, color=colors)
        for b, v in zip(bars, mean):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}" if key == "fid" else f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("vs archive/Testing (n=1600); error bars = std over 3 generation seeds; 3,000 generator updates each", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fid_kid_bar_chart.png", dpi=130)
    plt.close(fig)

    def panel(u8, n_rows_per_class=None):
        return make_grid(u8.float() / 255, nrow=8, padding=2).permute(1, 2, 0).numpy()
    g = torch.Generator().manual_seed(7)
    real_pick = torch.cat([real_te[(y_te == c).nonzero().squeeze(1)[torch.randperm(400, generator=g)[:8]]] for c in range(4)])
    imgs_by_model = {"Real test images (rows: glioma, meningioma, notumor, pituitary)": real_pick}
    d_imgs, _ = generate("DCGAN", CKPTS["DCGAN"], 1001, device)
    w_imgs, _ = generate("WGAN-GP", CKPTS["WGAN-GP"], 1001, device)
    c_imgs, c_y = generate("cWGAN-GP", CKPTS["cWGAN-GP"], 1001, device)
    imgs_by_model["DCGAN (unconditional)"] = d_imgs[:32]
    imgs_by_model["WGAN-GP (unconditional)"] = w_imgs[:32]
    imgs_by_model["cWGAN-GP (rows: glioma, meningioma, notumor, pituitary)"] = torch.cat([c_imgs[c * 400:c * 400 + 8] for c in range(4)])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4))
    for ax, (t, u) in zip(axes.flat, imgs_by_model.items()):
        ax.imshow(panel(u), cmap="gray")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "sample_grids_real_vs_generated.png", dpi=120)
    plt.close(fig)

    print(json.dumps(aggregates, indent=2))
    print("per-class:", json.dumps(per_class_rows, indent=2))
    print("class confusion KID (rows=generated label, cols=real class):\n", np.round(conf, 4))
    print(f"done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
