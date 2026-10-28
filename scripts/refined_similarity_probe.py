#!/usr/bin/env python3
"""Calibration probe: does a higher-resolution, shift-tolerant similarity separate copies from merely-similar images?
Samples val-rep -> best train-rep pairs (32x32 similarity bins), then rescoring at 96x96 with +/-4px shifts in the matched orientation."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "src")
ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "archive"
rows = list(csv.DictReader(open(ROOT / "outputs/deduplicated_experiment/split_manifest.csv")))
tr = [r["source_path"] for r in rows if r["partition"] == "train" and r["is_representative"] == "True"]
va = [r["source_path"] for r in rows if r["partition"] == "val" and r["is_representative"] == "True"]

def im(p, s):
    return np.asarray(Image.open(A / p).convert("L").resize((s, s), Image.BILINEAR), np.float32)
def variants(a):
    return [a, a[:, ::-1], a[::-1, :], np.rot90(a, 1), np.rot90(a, 2), np.rot90(a, 3), a.T, a.T[::-1, ::-1]]
def norm(a):
    v = a.reshape(len(a), -1); v = v - v.mean(1, keepdims=True); return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
def refined(pa, pb, k, s=96, sh=4):
    a = im(pa, s); b = variants(im(pb, s))[k]
    best = -1.0
    for dy in range(-sh, sh + 1):
        for dx in range(-sh, sh + 1):
            x = a[max(0, dy):s + min(0, dy), max(0, dx):s + min(0, dx)]
            y = b[max(0, -dy):s + min(0, -dy), max(0, -dx):s + min(0, -dx)]
            x = x - x.mean(); y = y - y.mean()
            best = max(best, float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)))
    return best
T = np.stack([im(p, 32) for p in tr]); V = np.stack([im(p, 32) for p in va])
bs = np.full(len(V), -1.0); bj = np.zeros(len(V), int); bk = np.zeros(len(V), int)
for k, v in enumerate(variants_ := [T, T[:, :, ::-1], T[:, ::-1, :], np.rot90(T, 1, (1, 2)), np.rot90(T, 2, (1, 2)), np.rot90(T, 3, (1, 2)), T.transpose(0, 2, 1), T.transpose(0, 2, 1)[:, ::-1, ::-1]]):
    s = norm(V) @ norm(np.ascontiguousarray(v)).T
    m, a_ = s.max(1), s.argmax(1); u = m > bs; bs[u], bj[u], bk[u] = m[u], a_[u], k
rng = np.random.RandomState(0)
out = {}
for lo, hi in [(0.97, 0.98), (0.95, 0.97), (0.93, 0.95), (0.90, 0.93), (0.80, 0.90)]:
    idx = np.where((bs >= lo) & (bs < hi))[0]
    sel = rng.choice(idx, min(40, len(idx)), replace=False) if len(idx) else []
    r = [refined(va[i], tr[bj[i]], int(bk[i])) for i in sel]
    out[f"[{lo},{hi})"] = {"n_in_bin": int(len(idx)), "n_sampled": len(r), "refined_pcts_p10_p50_p90": [round(float(np.percentile(r, p)), 4) for p in (10, 50, 90)] if r else None}
    print(f"32x32 sim in [{lo},{hi}): {len(idx):3d} val reps; refined(96px, +/-4px) p10/p50/p90 =", out[f"[{lo},{hi})"]["refined_pcts_p10_p50_p90"], flush=True)
json.dump(out, open(ROOT / "outputs/deduplicated_experiment/refined_similarity_calibration.json", "w"), indent=1)
