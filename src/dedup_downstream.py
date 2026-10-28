#!/usr/bin/env python3
"""Downstream classification on the CLUSTER-DISJOINT split: A real-only, B real + cWGAN-GP synthetic, C matched-composition real-only control.

    python src/dedup_downstream.py                       # after the three GANs are verified
    python src/dedup_downstream.py --smoke --device cpu --out-dir <scratch>

* Data: representatives of partitions train / val / test from outputs/deduplicated_experiment/split_manifest.csv (no copies, no cluster in two partitions).
* A: all real training representatives.  B: A + synthetic images from the cWGAN-GP trained on the SAME training partition; synthetic_per_class = 1.25 * n_train / 4
  (same ratio as the earlier experiment), labels = the conditioning label each image was generated for (correctness unverified).
  C: real training representatives resampled WITH replacement to B's per-class composition (same total, same class mix, no synthetic images).
* Identical for all conditions: SmallCNN, Adam 1e-3, batch 64, 15 epochs, no augmentation, seeds, real-only validation partition for best-epoch selection.
* The test partition is read only after training to evaluate the validation-selected weights. Predictions are saved for every run.
* Provenance check: the GAN checkpoint's recorded training-file hash must equal the hash of the manifest's training representatives.
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

import downstream_classification as dc
from data_pipeline import ManifestDataset, rel_path

ROOT = Path(__file__).resolve().parent.parent
DEXP = ROOT / "outputs/deduplicated_experiment"
CLASSES = dc.CLASSES


def load_partition(manifest: Path, partition: str, cap_per_class: int | None = None):
    imgs, labels, files = [], [], []
    seen = {c: 0 for c in CLASSES}
    for r in csv.DictReader(open(manifest)):
        if r["partition"] != partition or r["is_representative"] != "True":
            continue
        if cap_per_class and seen[r["class_name"]] >= cap_per_class:
            continue
        seen[r["class_name"]] += 1
        with Image.open(ROOT / "archive" / r["source_path"]) as im:
            imgs.append((dc.TF(im) * 255).round().to(torch.uint8))
        labels.append(int(r["class_idx"]))
        files.append(r["source_path"])
    return torch.stack(imgs), torch.tensor(labels), files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEXP / "split_manifest.csv")
    ap.add_argument("--gan-ckpt", type=Path, default=DEXP / "gan/cwgangp/checkpoint_step_3000.pth")
    ap.add_argument("--out-dir", type=Path, default=DEXP / "downstream")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--resample-seed", type=int, default=5001)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = ap.parse_args()
    if (args.out_dir / "summary.json").exists():
        print(f"ERROR: {args.out_dir}/summary.json exists; refusing to overwrite", file=sys.stderr)
        return 2
    device = torch.device("mps" if args.device in ("auto", "mps") and torch.backends.mps.is_available() else "cpu")
    if args.smoke:
        args.seeds, args.epochs = args.seeds[:1], 1
    (args.out_dir / "runs").mkdir(parents=True, exist_ok=True)

    cap = (12, 6, 6) if args.smoke else (None, None, None)
    xtr, ytr, ftr = load_partition(args.manifest, "train", cap[0])
    xva, yva, fva = load_partition(args.manifest, "val", cap[1])
    assert not (set(ftr) & set(fva)), "train/val file overlap"
    ck = torch.load(args.gan_ckpt, map_location="cpu", weights_only=True)
    audit = {"gan_checkpoint": str(args.gan_ckpt), "gan_gen_updates": ck["gen_updates"], "gan_critic_updates": ck["critic_updates"]}
    if not args.smoke:
        cfg = ck["config"]
        recorded = cfg.get("subset_file_list_sha256")
        recomputed = hashlib.sha256("\n".join(rel_path(p) for p, _ in ManifestDataset(args.manifest, "train", 64, False).samples).encode()).hexdigest()
        audit.update({"gan_manifest_arg": cfg.get("manifest"), "gan_recorded_train_hash": recorded, "manifest_train_hash": recomputed, "gan_trained_on_exactly_the_train_partition": recorded == recomputed,
                      "gan_images_used": cfg.get("images_used")})
        assert recorded == recomputed and cfg.get("images_used") == len(ftr), "GAN training data differs from the manifest's train partition"
        assert ck["gen_updates"] == 3000 and ck["critic_updates"] == 15000
    n_train = len(ftr)
    syn_pc = max(2, round(1.25 * n_train / 4))
    xsyn, ysyn = dc.make_synthetic(syn_pc, device, ck) if not args.smoke else dc.make_synthetic(syn_pc, device, {"G": ck["G"], "gen_updates": 3000})
    base = np.bincount(ytr.numpy(), minlength=4)
    target = base + syn_pc                                             # B's per-class composition
    rng = np.random.RandomState(args.resample_seed)
    cidx = np.concatenate([rng.choice(np.where(ytr.numpy() == c)[0], int(target[c]), replace=True) for c in range(4)])
    mult = np.bincount(cidx, minlength=n_train)
    conds = {"A_real_only": (xtr, ytr), "B_real_plus_cwgangp_synthetic": (torch.cat([xtr, xsyn]), torch.cat([ytr, ysyn])), "C_real_only_matched_composition": (xtr[cidx], ytr[cidx])}
    for k, (x, y) in conds.items():
        print(f"{k}: n={len(x)} per-class={np.bincount(y.numpy(), minlength=4).tolist()}", flush=True)
    assert len(conds["B_real_plus_cwgangp_synthetic"][0]) == len(conds["C_real_only_matched_composition"][0])
    assert np.array_equal(np.bincount(conds["B_real_plus_cwgangp_synthetic"][1].numpy(), minlength=4), np.bincount(conds["C_real_only_matched_composition"][1].numpy(), minlength=4))
    (args.out_dir / "resampling_manifest_C.json").write_text(json.dumps({"seed": args.resample_seed, "examples": int(len(cidx)), "per_class": target.tolist(),
                                                                       "unique_files_used": int((mult > 0).sum()), "unique_available": n_train, "max_multiplicity": int(mult.max()),
                                                                       "file_multiplicity": {ftr[i]: int(mult[i]) for i in range(n_train) if mult[i] > 0}}))
    config = {"manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "seeds": args.seeds, "epochs": args.epochs,
              "n_train": n_train, "n_val": len(fva), "synthetic_per_class": syn_pc, "synthetic_seed": dc.SYN_SEED, "synthetic_labels": "requested conditioning labels; correctness unverified",
              "composition": {k: np.bincount(y.numpy(), minlength=4).tolist() for k, (_, y) in conds.items()}, "resample_seed": args.resample_seed, "leakage_audit": audit,
              "model": "SmallCNN (4 conv blocks 32-64-128-256, GAP, dropout 0.3), Adam 1e-3, batch 64, no augmentation", "device": str(device), "smoke": args.smoke,
              "test_use": "read only after training, for final evaluation of validation-selected weights", "torch_version": str(torch.__version__)}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))

    test_cache, captured = {}, {}
    orig_predict = dc.predict

    def capturing(model, x_u8, dev, bs=256):
        out = orig_predict(model, x_u8, dev, bs)
        captured["pred"] = out.tolist()
        return out
    dc.predict = capturing

    def load_test():
        if "d" not in test_cache:
            x, y, f = load_partition(args.manifest, "test", cap[2])
            assert not (set(f) & set(ftr)) and not (set(f) & set(fva)), "test files overlap train/val"
            test_cache["d"], captured["files"] = (x, y), f
        return test_cache["d"]

    runs = []
    t_all = time.perf_counter()
    for cond, (x, y) in conds.items():
        for seed in args.seeds:
            rp = args.out_dir / "runs" / f"{cond}_seed{seed}.json"
            if rp.exists():
                runs.append(json.loads(rp.read_text()))
                print(f"skip existing {rp.name}")
                continue
            r = dc.train_one(cond, seed, x, y, xva, yva, load_test, args.epochs, device)
            cm = np.array(r["test"]["confusion_matrix"], float)
            r["test"]["balanced_accuracy"] = float(np.mean(np.diag(cm) / np.maximum(cm.sum(1), 1)))
            rp.write_text(json.dumps(r, indent=2))
            (args.out_dir / "runs" / f"{cond}_seed{seed}_test_predictions.json").write_text(json.dumps({"test_files": captured["files"], "y_true": test_cache["d"][1].tolist(), "y_pred": captured["pred"]}))
            runs.append(r)
            print(f"{cond} seed {seed}: n_train={r['n_train']} best_ep={r['best_epoch']} val_acc={r['best_val_acc']:.4f} test_acc={r['test']['accuracy']:.4f} "
                  f"macroF1={r['test']['macro_f1']:.4f} balAcc={r['test']['balanced_accuracy']:.4f} ({r['train_seconds']}s)", flush=True)

    def agg(rs):
        a = {}
        for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
            v = np.array([r["test"][k] for r in rs])
            a[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "values": v.tolist()}
        va_ = np.array([r["best_val_acc"] for r in rs])
        a["val_accuracy"] = {"mean": float(va_.mean()), "values": va_.tolist()}
        for key, name in (("f1-score", "per_class_f1_mean"), ("recall", "per_class_recall_mean"), ("precision", "per_class_precision_mean")):
            a[name] = {c: float(np.mean([r["test"]["per_class"][c][key] for r in rs])) for c in CLASSES}
        a["mean_confusion_matrix"] = np.mean([r["test"]["confusion_matrix"] for r in rs], axis=0).tolist()
        a["n_train"], a["n_runs"] = rs[0]["n_train"], len(rs)
        return a
    summary = {"config": config, "n_test": runs[0]["n_test"], "conditions": {c: agg([r for r in runs if r["condition"] == c]) for c in conds}}
    cs = summary["conditions"]
    pd_ = {}
    for key in ("accuracy", "macro_f1"):
        a_, b_, c_ = (np.array(cs[k][key]["values"]) for k in conds)
        pd_[key] = {"B_minus_A": (b_ - a_).tolist(), "C_minus_A": (c_ - a_).tolist(), "B_minus_C": (b_ - c_).tolist(), "mean_B_minus_A": float((b_ - a_).mean()),
                    "mean_C_minus_A": float((c_ - a_).mean()), "mean_B_minus_C": float((b_ - c_).mean())}
    summary["paired_differences"] = pd_
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.out_dir / "comparison_A_B_C.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_train", "n_runs", "test_acc_mean", "test_acc_std", "macro_f1_mean", "macro_f1_std", "balanced_acc_mean", "val_acc_mean"] + [f"f1_{c}" for c in CLASSES])
        for k, s in cs.items():
            w.writerow([k, s["n_train"], s["n_runs"], s["accuracy"]["mean"], s["accuracy"]["std"], s["macro_f1"]["mean"], s["macro_f1"]["std"], s["balanced_accuracy"]["mean"], s["val_accuracy"]["mean"]] +
                       [s["per_class_f1_mean"][c] for c in CLASSES])
    with open(args.out_dir / "classification_report.txt", "w") as f:
        for k, s in cs.items():
            f.write(f"== {k} (mean over {s['n_runs']} seeds; n_train={s['n_train']}; test n={summary['n_test']}) ==\naccuracy {s['accuracy']['mean']:.4f}  macro-F1 {s['macro_f1']['mean']:.4f}  "
                    f"weighted-F1 {s['weighted_f1']['mean']:.4f}  balanced-acc {s['balanced_accuracy']['mean']:.4f}\n")
            for c in CLASSES:
                f.write(f"  {c:11s} P {s['per_class_precision_mean'][c]:.4f}  R {s['per_class_recall_mean'][c]:.4f}  F1 {s['per_class_f1_mean'][c]:.4f}\n")
            f.write("  mean confusion (rows true, cols pred):\n" + "".join("   " + " ".join(f"{v:7.1f}" for v in row) + "\n" for row in s["mean_confusion_matrix"]) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#4c78a8", "#54a24b", "#b279a2"]
    labs = ["A real only", "B real + synthetic", "C real, matched"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (key, title) in zip(axes, (("accuracy", "TEST accuracy"), ("macro_f1", "TEST macro-F1"), ("val_accuracy", "VALIDATION accuracy (epoch selection)"))):
        for i, (k, c) in enumerate(zip(conds, cols)):
            v = cs[k][key]["values"]
            ax.bar(i, np.mean(v), yerr=(np.std(v, ddof=1) if len(v) > 1 else 0), capsize=4, color=c, alpha=0.85)
            ax.scatter([i] * len(v), v, color="black", s=14, zorder=3)
        lo = min(min(cs[k][key]["values"]) for k in conds)
        ax.set_ylim(max(0, lo - 0.1), 1.0)
        for i, k in enumerate(conds):
            ax.text(i, ax.get_ylim()[0] + 0.005, f"{cs[k][key]['mean']:.3f}", ha="center", color="white", fontweight="bold", fontsize=8)
        ax.set_xticks(range(3), labs, fontsize=8)
        ax.set_title(title + f" (n={summary['n_test'] if key != 'val_accuracy' else len(fva)})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("De-duplicated split: A vs B vs C (3 seeds; bars mean +/- sd, dots = seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "accuracy_A_B_C.png", dpi=130)
    print(json.dumps({k: {"acc": round(v["accuracy"]["mean"], 4), "macro_f1": round(v["macro_f1"]["mean"], 4), "bal_acc": round(v["balanced_accuracy"]["mean"], 4)} for k, v in cs.items()}, indent=1))
    print(f"total {time.perf_counter() - t_all:.0f}s on {device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
