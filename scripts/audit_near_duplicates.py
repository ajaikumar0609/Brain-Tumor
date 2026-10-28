#!/usr/bin/env python3
"""Read-only near-duplicate audit (numpy + PIL): are Testing / CNN-validation images near-copies (incl. flips/rotations) of training images?

    python scripts/audit_near_duplicates.py          -> outputs/final_audit/near_duplicate_audit.json (+ example pair images)
Images -> 32x32 grayscale, mean-removed, L2-normalised; similarity = cosine. A training image is compared in 8 orientations
(identity, h/v flip, rot90/180/270, transposes). Exact byte duplicates give similarity 1.0.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "final_audit"
OUT.mkdir(parents=True, exist_ok=True)
CL = ["glioma", "meningioma", "notumor", "pituitary"]


def load(split):
    names, vecs = [], []
    for c in CL:
        for p in sorted((ROOT / "archive" / split / c).iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                im = Image.open(p).convert("L").resize((32, 32), Image.BILINEAR)
                names.append(f"{c}/{p.name}")
                vecs.append(np.asarray(im, np.float32))
    return names, np.stack(vecs)


def norm(a):
    v = a.reshape(len(a), -1)
    v = v - v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def variants(a):
    return [a, a[:, :, ::-1], a[:, ::-1, :], np.rot90(a, 1, (1, 2)), np.rot90(a, 2, (1, 2)), np.rot90(a, 3, (1, 2)), a.transpose(0, 2, 1), a.transpose(0, 2, 1)[:, ::-1, ::-1]]


def best_match(q, ref):
    """for each query: max cosine over all reference images and 8 orientations; returns sim, ref index, orientation."""
    qn = norm(q)
    best = np.full(len(q), -1.0)
    arg = np.zeros(len(q), int)
    ori = np.zeros(len(q), int)
    for k, rv in enumerate(variants(ref)):
        s = qn @ norm(np.ascontiguousarray(rv)).T
        m, i = s.max(1), s.argmax(1)
        upd = m > best
        best[upd], arg[upd], ori[upd] = m[upd], i[upd], k
    return best, arg, ori


def self_best(a):
    """training images vs other training images (exclude self in identity orientation)"""
    qn = norm(a)
    best = np.full(len(a), -1.0)
    for k, rv in enumerate(variants(a)):
        s = qn @ norm(np.ascontiguousarray(rv)).T
        if k == 0:
            np.fill_diagonal(s, -1)
        best = np.maximum(best, s.max(1))
    return best


def rate(sims, thr):
    return {f">={t}": int((sims >= t).sum()) for t in thr}


trn, tra = load("Training")
ten, tea = load("Testing")
THR = [0.999, 0.99, 0.98, 0.95, 0.90]
res = {"method": __doc__.strip().splitlines()[0], "n_train": len(trn), "n_test": len(ten)}
sim_te, idx_te, ori_te = best_match(tea, tra)
aug = np.array(["aug" in n for n in ten])
res["testing_vs_training"] = {"all": rate(sim_te, THR), "n": len(ten),
                              "te_aug_images": {"n": int(aug.sum()), **rate(sim_te[aug], THR), "median_best_sim": float(np.median(sim_te[aug]))},
                              "non_aug_images": {"n": int((~aug).sum()), **rate(sim_te[~aug], THR), "median_best_sim": float(np.median(sim_te[~aug]))},
                              "orientation_of_best_match_for_hits_ge_0.98": {str(int(o)): int(((ori_te == o) & (sim_te >= 0.98)).sum()) for o in range(8)}}
sim_tt = self_best(tra)
taug = np.array(["aug" in n for n in trn])
res["training_vs_other_training"] = {"all": rate(sim_tt, THR), "tr_aug_images": {"n": int(taug.sum()), **rate(sim_tt[taug], THR)}, "non_aug": rate(sim_tt[~taug], THR),
                                     "percentiles_best_sim_p50_p90_p99": [float(np.percentile(sim_tt, p)) for p in (50, 90, 99)]}
# CNN partitions (the split used by the downstream experiments)
tr_files = set(json.load(open(ROOT / "outputs/cwgangp_clean_split/cnn_train_files.json")))
va_files = set(json.load(open(ROOT / "outputs/cwgangp_clean_split/cnn_val_files.json")))
itr = np.array([i for i, n in enumerate(trn) if n in tr_files])
iva = np.array([i for i, n in enumerate(trn) if n in va_files])
sim_v, idx_v, ori_v = best_match(tra[iva], tra[itr])
res["cnn_validation_vs_cnn_train"] = {"n_val": int(len(iva)), **rate(sim_v, THR), "val_aug_images": {"n": int(taug[iva].sum()), **rate(sim_v[taug[iva]], THR)}}
# baseline: exact-duplicate threshold sanity (hash check gave 44 exact val/train pairs)
res["note"] = "similarity 1.0 = identical up to resize; flips/rotations are included so augmented copies are detectable; this is a heuristic, not proof."
json.dump(res, open(OUT / "near_duplicate_audit.json", "w"), indent=2)

# example pairs: top-closest Testing->Training (non-identity orientation preferred to show augmentation) for visual confirmation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = np.argsort(-sim_te)[:6].tolist() + [i for i in np.argsort(-sim_te) if aug[i]][:6]
fig, axes = plt.subplots(2, len(order), figsize=(2 * len(order), 4.4))
for j, i in enumerate(order):
    for r, (nm, path) in enumerate(((ten[i], ROOT / "archive/Testing" / ten[i]), (trn[idx_te[i]], ROOT / "archive/Training" / trn[idx_te[i]]))):
        axes[r, j].imshow(Image.open(path).convert("L"), cmap="gray")
        axes[r, j].axis("off")
        axes[r, j].set_title(("TEST " if r == 0 else "TRAIN ") + nm.split("/")[1][:14] + (f"\nsim {sim_te[i]:.3f} ori {ori_te[i]}" if r == 0 else ""), fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "closest_testing_training_pairs.png", dpi=110)
print(json.dumps({k: res[k] for k in ("testing_vs_training", "cnn_validation_vs_cnn_train")}, indent=1))
print("training self-sim percentiles p50/p90/p99:", res["training_vs_other_training"]["percentiles_best_sim_p50_p90_p99"], "| train aug images:", res["training_vs_other_training"]["tr_aug_images"])
