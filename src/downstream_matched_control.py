"""Condition C: real-only, sample-count-matched control for the downstream comparison (ChatGPT handoff #1, validated).

    python src/downstream_matched_control.py                  # after the clean downstream experiment has completed
    python src/downstream_matched_control.py --smoke --out-dir <scratch>

Condition C trains on ONLY the 4,480 CNN-training real images, resampled WITH replacement (class-balanced, fixed seed) to the
same size as Condition B (real + synthetic = 10,080; 2,520 per class). Repeated images are duplicates, not new data.
Everything else is identical to the clean experiment (same CNN, preprocessing, Adam 1e-3, batch 64, 15 epochs, split seed,
run seeds, real-only validation set for epoch selection). archive/Testing is read only for the final evaluation of the
validation-selected weights. The GAN and synthetic images are not used.
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

import downstream_classification as dc

ROOT = Path(__file__).resolve().parent.parent
CLASSES = dc.CLASSES
COND = "real_only_matched_size"


def fail(msg: str) -> int:
    print(f"PRECONDITION FAILED: {msg}", file=sys.stderr)
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-dir", type=Path, default=ROOT / "outputs/downstream_classification_clean")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/downstream_classification_matched")
    ap.add_argument("--resample-seed", type=int, default=5001)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = ap.parse_args()
    if (args.out_dir / "summary.json").exists():
        return fail(f"{args.out_dir}/summary.json exists; refusing to overwrite")
    device = torch.device("mps" if args.device in ("auto", "mps") and torch.backends.mps.is_available() else "cpu")

    # ---- preconditions: the verified clean experiment ----
    clean = None
    if args.smoke:
        seeds, epochs, cap, n_syn_pc = [42], 1, 40, 50
    else:
        sp = args.clean_dir / "summary.json"
        if not sp.exists():
            return fail("clean downstream summary.json missing (clean experiment not complete)")
        clean = json.loads(sp.read_text())
        cfg = clean["config"]
        if cfg.get("smoke") or cfg.get("leakage_free") is not True:
            return fail("clean experiment is a smoke run or not leakage-free")
        if cfg["leakage_audit"]["gan_overlap_with_cnn_validation"] != 0 or cfg["leakage_audit"]["gan_saw_archive_testing"]:
            return fail("clean experiment's leakage audit is not clean")
        seeds, epochs, cap, n_syn_pc = cfg["run_seeds"], cfg["epochs"], None, cfg["synthetic_per_class"]
        if cfg["split_seed"] != dc.SPLIT_SEED:
            return fail("split seed differs from the downstream module")
        runs = list((args.clean_dir / "runs").glob("*.json"))
        if len(runs) != 2 * len(seeds):
            return fail(f"expected {2 * len(seeds)} clean runs, found {len(runs)}")

    x_all, y_all, paths = dc.load_images(ROOT / "archive/Training", cap)
    tr, va = dc.stratified_split(y_all, 0.2, dc.SPLIT_SEED)
    n_real = len(tr)
    n_b = n_real + 4 * n_syn_pc
    if clean is not None and (clean["conditions"]["real_only"]["n_train"] != n_real or clean["conditions"]["real_plus_cwgangp_synthetic"]["n_train"] != n_b):
        return fail("training-set sizes differ from the clean experiment's recorded sizes")
    per_class = n_b // 4
    assert per_class * 4 == n_b

    # ---- Condition C training set: resample the permitted real training images ----
    rng = np.random.RandomState(args.resample_seed)
    tr_arr, y_np = np.array(tr), y_all.numpy()
    drawn = []
    for c in range(4):
        pool = tr_arr[y_np[tr_arr] == c]
        drawn += rng.choice(pool, per_class, replace=True).tolist()
    drawn = np.array(drawn)
    assert set(drawn.tolist()) <= set(tr) and not (set(drawn.tolist()) & set(va)), "resampling left the CNN-training partition"
    assert len(drawn) == n_b and all((y_np[drawn] == c).sum() == per_class for c in range(4))
    mult = np.bincount(drawn, minlength=len(paths))
    manifest = {paths[i]: int(mult[i]) for i in tr if mult[i] > 0}
    uniq = {CLASSES[c]: int(len({i for i in drawn if y_np[i] == c})) for c in range(4)}
    manifest_stats = {"n_examples": int(len(drawn)), "per_class": per_class, "unique_files_used": len(manifest), "unique_files_available": n_real,
                      "unique_used_per_class": uniq, "max_multiplicity": int(mult.max()), "unused_files": n_real - len(manifest)}
    xtr, ytr = x_all[drawn], y_all[drawn]
    xva, yva = x_all[va], y_all[va]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "runs").mkdir(exist_ok=True)
    (args.out_dir / "resampling_manifest.json").write_text(json.dumps({"stats": manifest_stats, "file_multiplicity": manifest}, indent=0))
    config = {"condition": COND, "description": "real-only, resampled with replacement to the size of the real+synthetic condition", "resample_seed": args.resample_seed,
              "split_seed": dc.SPLIT_SEED, "run_seeds": seeds, "epochs": epochs, "device": str(device), "smoke": args.smoke,
              "manifest_stats": manifest_stats, "manifest_sha256": hashlib.sha256((args.out_dir / "resampling_manifest.json").read_bytes()).hexdigest(),
              "clean_experiment_dir": str(args.clean_dir), "same_as_clean": "CNN, preprocessing, Adam 1e-3, batch 64, no augmentation, real-only validation",
              "test_set_role": "final evaluation of validation-selected weights only (also used earlier in the project as the FID/KID reference and for a pretrained-classifier "
                               "sanity check, but never to train any GAN or to select anything in this experiment)", "torch_version": str(torch.__version__)}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"device={device} condition C: {len(xtr)} examples ({per_class}/class) from {len(manifest)}/{n_real} unique real files; val={len(xva)}", flush=True)

    # ---- capture test predictions without altering the training/eval code ----
    captured = {}
    orig_predict = dc.predict

    def capturing(model, x_u8, dev, bs=256):
        out = orig_predict(model, x_u8, dev, bs)
        captured["pred"] = out.tolist()
        return out
    dc.predict = capturing
    test_cache = {}

    def load_test():
        if "d" not in test_cache:
            x, y, tp = dc.load_images(ROOT / "archive/Testing", 100 if args.smoke else None)
            test_cache["d"] = (x, y)
            captured["test_files"] = tp
        return test_cache["d"]

    t0, runs = time.perf_counter(), []
    for seed in seeds:
        r = dc.train_one(COND, seed, xtr, ytr, xva, yva, load_test, epochs, device)
        (args.out_dir / "runs" / f"{COND}_seed{seed}.json").write_text(json.dumps(r, indent=2))
        (args.out_dir / "runs" / f"{COND}_seed{seed}_test_predictions.json").write_text(
            json.dumps({"test_files": captured["test_files"], "y_true": test_cache["d"][1].tolist(), "y_pred": captured["pred"]}))
        runs.append(r)
        print(f"{COND} seed {seed}: n_train={r['n_train']} best_epoch={r['best_epoch']} val_acc={r['best_val_acc']:.4f} test_acc={r['test']['accuracy']:.4f} "
              f"macroF1={r['test']['macro_f1']:.4f} ({r['train_seconds']}s)", flush=True)

    # ---- aggregate + comparison with the recorded A and B ----
    def agg(rs):
        a = {}
        for k in ("accuracy", "macro_f1", "weighted_f1"):
            v = np.array([r["test"][k] for r in rs])
            a[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "values": v.tolist()}
        a["per_class_f1_mean"] = {c: float(np.mean([r["test"]["per_class"][c]["f1-score"] for r in rs])) for c in CLASSES}
        a["per_class_precision_mean"] = {c: float(np.mean([r["test"]["per_class"][c]["precision"] for r in rs])) for c in CLASSES}
        a["per_class_recall_mean"] = {c: float(np.mean([r["test"]["per_class"][c]["recall"] for r in rs])) for c in CLASSES}
        a["mean_confusion_matrix"] = np.mean([r["test"]["confusion_matrix"] for r in rs], axis=0).tolist()
        a["n_train"], a["n_runs"] = rs[0]["n_train"], len(rs)
        return a
    c_agg = agg(runs)
    conds = {"A_real_only": clean["conditions"]["real_only"] if clean else None, "B_real_plus_synthetic": clean["conditions"]["real_plus_cwgangp_synthetic"] if clean else None,
             "C_real_only_matched_size": c_agg}
    summary = {"config": config, "condition_C": c_agg, "recorded_A_and_B_from": str(args.clean_dir / "summary.json")}
    if clean:
        a_v, b_v, c_v = (np.array(conds[k]["accuracy"]["values"]) for k in conds)
        summary["paired_accuracy_differences"] = {"B_minus_A": (b_v - a_v).tolist(), "C_minus_A": (c_v - a_v).tolist(), "B_minus_C": (b_v - c_v).tolist(),
                                                  "mean_B_minus_A": float((b_v - a_v).mean()), "mean_C_minus_A": float((c_v - a_v).mean()), "mean_B_minus_C": float((b_v - c_v).mean())}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.out_dir / "comparison_A_B_C.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_train", "unique_real_images", "synthetic_images", "n_runs", "test_acc_mean", "test_acc_std", "macro_f1_mean", "macro_f1_std", "weighted_f1_mean"] + [f"f1_{c}" for c in CLASSES])
        meta = {"A_real_only": (n_real, 0), "B_real_plus_synthetic": (n_real, 4 * n_syn_pc), "C_real_only_matched_size": (len(manifest), 0)}
        for name, s in conds.items():
            if s is None:
                continue
            u, sy = meta[name]
            w.writerow([name, s["n_train"], u, sy, s["n_runs"], s["accuracy"]["mean"], s["accuracy"]["std"], s["macro_f1"]["mean"], s["macro_f1"]["std"], s["weighted_f1"]["mean"]] +
                       [s["per_class_f1_mean"][c] for c in CLASSES])
    with open(args.out_dir / "classification_report.txt", "w") as f:
        for name, s in conds.items():
            if s is None:
                continue
            f.write(f"== {name} (mean over {s['n_runs']} seeds; n_train={s['n_train']}) ==\naccuracy {s['accuracy']['mean']:.4f}  macro-F1 {s['macro_f1']['mean']:.4f}  weighted-F1 {s['weighted_f1']['mean']:.4f}\n")
            for c in CLASSES:
                f.write(f"  {c:11s} P {s['per_class_precision_mean'][c]:.4f}  R {s['per_class_recall_mean'][c]:.4f}  F1 {s['per_class_f1_mean'][c]:.4f}\n")
            f.write("  mean confusion matrix (rows true, cols pred):\n" + "".join("   " + " ".join(f"{v:7.1f}" for v in row) + "\n" for row in s["mean_confusion_matrix"]) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = [(n, s) for n, s in conds.items() if s is not None]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, shown):
        cm = np.array(s["mean_confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(4), CLASSES)
        ax.set_title(f"{name}\nacc {s['accuracy']['mean']:.3f}", fontsize=8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrices_A_B_C.png", dpi=120)

    # ---- interpretation (numbers computed; wording rule-based and hedged) ----
    lines = ["# Matched-sample-count control: interpretation", "",
             "Condition A trains on 4,480 real images; B on the same real images plus 5,600 cWGAN-GP images (10,080 examples); C on the same 4,480 real "
             "images resampled with replacement to 10,080 examples (2,520 per class) with no synthetic data, so B and C have identical size and optimizer-step "
             "counts.", "", "| Condition | n_train | unique real images | test accuracy (mean +/- sd, 3 seeds) | macro-F1 |", "|---|---|---|---|---|"]
    for name, s in conds.items():
        if s:
            u, _ = meta[name]
            sd = f"{s['accuracy']['std']:.4f}" if s["accuracy"]["std"] is not None else "n/a"
            lines.append(f"| {name} | {s['n_train']} | {u} | {s['accuracy']['mean']:.4f} +/- {sd} | {s['macro_f1']['mean']:.4f} |")
    lines.append("")
    if clean:
        d = summary["paired_accuracy_differences"]
        sdB, sdC = conds["B_real_plus_synthetic"]["accuracy"]["std"] or 0, c_agg["accuracy"]["std"] or 0
        lines += [f"Per-seed paired accuracy differences: B-A {np.round(d['B_minus_A'], 4).tolist()}, C-A {np.round(d['C_minus_A'], 4).tolist()}, B-C {np.round(d['B_minus_C'], 4).tolist()} "
                  f"(means {d['mean_B_minus_A']:+.4f}, {d['mean_C_minus_A']:+.4f}, {d['mean_B_minus_C']:+.4f}).", ""]
        gap = d["mean_B_minus_C"]
        if abs(gap) <= max(sdB, sdC):
            verdict = ("B and C differ by less than the seed-to-seed standard deviation, so with 3 seeds this experiment cannot distinguish the effect of "
                       "synthetic images from the effect of simply training on more (duplicated) examples.")
        elif gap > 0:
            verdict = ("B exceeds C by more than the seed standard deviation, which is consistent with the synthetic images adding something beyond sample count "
                       "(diversity, regularisation); it does not show the images are anatomically correct or class-faithful.")
        else:
            verdict = "C matches or exceeds B: there is no evidence here that the synthetic images add benefit beyond training on more copies of the real images."
        lines += ["**Reading (hedged, no significance test run):** " + verdict, ""]
    lines += ["## What this control can and cannot show", "- It controls for training-set size and optimizer steps; it does not control for diversity: duplicated real images are exact copies, "
              "whereas generated images may add variation or label noise.",
              "- Synthetic labels are the cWGAN-GP conditioning labels; the GAN evaluation gave no strong evidence that images match tumour classes.",
              f"- Condition C uses {manifest_stats['unique_files_used']} of {n_real} unique real images ({manifest_stats['unused_files']} were not drawn); do not describe it as 10,080 unique images.",
              "- Three run seeds, one resampling seed, one GAN training run: variability from resampling and GAN training is unmeasured.",
              "- archive/Testing role: final evaluation of validation-selected weights for A, B and C only; it was also used earlier in the project as the FID/KID reference and for the "
              "pretrained-classifier sanity check, so it is not untouched across the whole project (no GAN or selection step used it)."]
    (args.out_dir / "interpretation.md").write_text("\n".join(lines) + "\n")
    print(f"total runtime {time.perf_counter() - t0:.1f}s on {device}")
    print(json.dumps({"C": {"acc_mean": c_agg["accuracy"]["mean"], "macro_f1_mean": c_agg["macro_f1"]["mean"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
