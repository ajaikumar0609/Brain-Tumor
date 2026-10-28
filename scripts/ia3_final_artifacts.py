#!/usr/bin/env python3
"""Build outputs/gan_comparison/final_comparison.csv/.png from the REAL metrics files (no manual values).

    python scripts/ia3_final_artifacts.py --root <project root>
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
ap.add_argument("--downstream-dir", default="outputs/downstream_classification")
ap.add_argument("--preliminary-dir", default=None, help="leaky/preliminary downstream results, listed separately and labelled")
ap.add_argument("--matched-dir", default=None, help="matched-sample-count control (Condition C) results")
_a = ap.parse_args()
root = _a.root
gd = root / "outputs" / "gan_comparison"
m = json.loads((gd / "metrics.json").read_text())["aggregate"]
ds = json.loads((root / _a.downstream_dir / "summary.json").read_text())
prelim = json.loads((root / _a.preliminary_dir / "summary.json").read_text()) if _a.preliminary_dir and (root / _a.preliminary_dir / "summary.json").exists() else None
rows = []
for name, v in m.items():
    rows += [["gan_quality", name, "FID", v["fid_mean"], v["fid_std_across_seeds"], v["n_seeds"], f"n_generated={v['n_generated_per_seed']} vs n_real={v['n_real']}"],
             ["gan_quality", name, "KID_x1000", v["kid_mean"] * 1000, v["kid_std_across_seeds"] * 1000, v["n_seeds"], "poly-kernel unbiased MMD^2"]]
for cond, c in ds["conditions"].items():
    for k, label in (("accuracy", "test_accuracy"), ("macro_f1", "test_macro_f1")):
        rows.append(["downstream_classification", cond, label, c[k]["mean"], c[k]["std"], c["n_runs"], f"n_train={c['n_train']}; test=archive/Testing 1600"])
import time as _t
for _name in ("final_comparison.csv", "final_comparison.png"):          # never overwrite earlier versions
    _p = gd / _name
    if _p.exists():
        _p.rename(gd / f"{_p.stem}.prev-{_t.strftime('%Y%m%d-%H%M%S')}{_p.suffix}")
matched = json.loads((root / _a.matched_dir / "summary.json").read_text()) if _a.matched_dir and (root / _a.matched_dir / "summary.json").exists() else None
if matched:
    c = matched["condition_C"]
    for k, label in (("accuracy", "test_accuracy"), ("macro_f1", "test_macro_f1")):
        rows.append(["downstream_classification", "C_real_only_matched_size", label, c[k]["mean"], c[k]["std"], c["n_runs"],
                     f"n_train={c['n_train']} (resampled real, duplicates); control for training-set size"])
if prelim:
    for cond, c in prelim["conditions"].items():
        for k, label in (("accuracy", "test_accuracy"), ("macro_f1", "test_macro_f1")):
            rows.append(["downstream_PRELIMINARY_leaky", cond, label, c[k]["mean"], c[k]["std"], c["n_runs"],
                         "PRELIMINARY: GAN saw the CNN validation images; do not use for conclusions"])
with open(gd / "final_comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["table", "name", "metric", "mean", "std_across_seeds", "n_seeds", "notes"])
    w.writerows(rows)

models = list(m)
labels = ["Real train\n(reference)" if k.startswith("Real") else k for k in models]
colors = ["#9aa0a6", "#4c78a8", "#f58518", "#54a24b"][: len(models)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, key, scale, title in ((axes[0], "fid", 1, "FID (lower better)"), (axes[1], "kid", 1000, "KID x1000 (lower better)")):
    mean = [m[k][f"{key}_mean"] * scale for k in models]
    std = [m[k][f"{key}_std_across_seeds"] * scale for k in models]
    bars = ax.bar(labels, mean, yerr=std, capsize=4, color=colors)
    for b, v in zip(bars, mean):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
import glob as _g
_dd = root / _a.downstream_dir
def _runs(pattern):
    return [json.loads(open(f).read()) for f in sorted(_g.glob(str(pattern)))]
_A = _runs(_dd / "runs" / "real_only_seed4?.json")
_B = _runs(_dd / "runs" / "real_plus_cwgangp_synthetic_seed4?.json")
_C = _runs(root / _a.matched_dir / "runs" / "real_only_matched_size_seed4?.json") if matched else []
groups = [("real only\n(A, n=4,480)", _A, "#4c78a8"), ("real + synthetic\n(B, n=10,080)", _B, "#54a24b")]
if _C:
    groups.append(("real, matched size\n(C, n=10,080)", _C, "#b279a2"))
def _bars(ax, getter, title):
    _lo = min(getter(r) for _, rs, _ in groups for r in rs)
    ax.set_ylim(max(0, _lo - 0.08), 1.0)
    for i, (lab, rs, col) in enumerate(groups):
        v = [getter(r) for r in rs]
        ax.bar(i, sum(v) / len(v), yerr=(__import__("statistics").stdev(v) if len(v) > 1 else 0), capsize=4, color=col, alpha=0.85)
        ax.scatter([i] * len(v), v, color="black", s=14, zorder=3)
        ax.text(i, ax.get_ylim()[0] + 0.004, f"mean {sum(v) / len(v):.3f}", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    lo = min(getter(r) for _, rs, _ in groups for r in rs)
    ax.set_ylim(max(0, lo - 0.08), 1.0)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
_bars(axes[2], lambda r: r["test"]["accuracy"], "Downstream TEST accuracy (archive/Testing, n=1600)\nbars = mean +/- sd, dots = seeds")
fig.suptitle("Error bars: std across seeds (GAN: generation seeds; classifier: run seeds). One GAN training run per model.", fontsize=9)
fig.tight_layout()
fig.savefig(gd / "final_comparison.png", dpi=130)
print(f"wrote {gd / 'final_comparison.csv'} ({len(rows)} rows) and final_comparison.png")

if _C:
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.2))
    _bars(ax2[0], lambda r: r["test"]["accuracy"], "TEST accuracy (archive/Testing, n=1600)")
    _bars(ax2[1], lambda r: r["best_val_acc"], "VALIDATION accuracy at selected epoch (n=1120)\nepoch selection only; 44 val images duplicate training")
    fig2.suptitle("Matched-sample-count control: A real only vs B real+synthetic vs C real resampled to the same size (3 seeds)", fontsize=9)
    fig2.tight_layout()
    _out = root / _a.matched_dir / "accuracy_A_B_C.png"
    if _out.exists():
        _out.rename(_out.with_name(f"accuracy_A_B_C.prev-{_t.strftime('%Y%m%d-%H%M%S')}.png"))
    fig2.savefig(_out, dpi=130)
    print(f"wrote {_out}")
