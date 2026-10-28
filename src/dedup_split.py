#!/usr/bin/env python3
"""Cluster-aware, class-stratified, cluster-disjoint train/val/test split of the COMBINED pool archive/Training + archive/Testing.

    python src/dedup_split.py [--threshold 0.99] [--out-dir outputs/deduplicated_experiment]

Near-duplicate detection (same method as scripts/audit_near_duplicates.py): 32x32 grayscale, mean-removed, L2-normalised,
cosine similarity; image j is compared with image i in 8 orientations (identity, h/v flips, rot90/180/270, two transposes) so mirrored
copies are found. Pairs with similarity >= threshold (and byte-identical files) are edges; connected components (transitive matches)
are clusters. Clusters with conflicting class labels, and files whose NAME marks them as augmented ('aug'), are quarantined (the latter because
augmented images can be transformed copies this method cannot link). One deterministic representative per cluster is used as the real image;
representatives are split 70/15/15 per class by cluster. This is a heuristic: it detects copies/mirrors/re-encodes, NOT same-patient
neighbouring slices (no patient ids exist), so residual leakage of that kind is possible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "archive"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
METHOD = "32x32 mean-removed cosine, 8 orientations, edge if sim>=threshold or byte-identical; connected components"


def load_pool():
    paths, vecs, shas = [], [], []
    for split in ("Training", "Testing"):
        for c in CLASSES:
            for p in sorted((ARCH / split / c).iterdir()):
                if p.suffix.lower() in EXTS:
                    paths.append(f"{split}/{c}/{p.name}")
                    shas.append(hashlib.sha256(p.read_bytes()).hexdigest())
                    vecs.append(np.asarray(Image.open(p).convert("L").resize((32, 32), Image.BILINEAR), np.float32))
    return paths, np.stack(vecs), shas


def norm(a):
    v = a.reshape(len(a), -1)
    v = v - v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def variants(a):
    return [a, a[:, :, ::-1], a[:, ::-1, :], np.rot90(a, 1, (1, 2)), np.rot90(a, 2, (1, 2)), np.rot90(a, 3, (1, 2)),
            a.transpose(0, 2, 1), a.transpose(0, 2, 1)[:, ::-1, ::-1]]


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def components(n, edges, thr, shas):
    uf = UF(n)
    for (i, j), (s, k) in edges.items():
        if s >= thr:
            uf.union(i, j)
    by = defaultdict(list)
    for i, h in enumerate(shas):
        by[h].append(i)
    for idxs in by.values():
        for i in idxs[1:]:
            uf.union(idxs[0], i)
    return [uf.find(i) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/deduplicated_experiment")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "split_manifest.csv").exists():
        print("split_manifest.csv exists; refusing to overwrite (use another --out-dir)")
        return 2
    paths, vecs, shas = load_pool()
    n = len(paths)
    cls = np.array([CLASSES.index(p.split("/")[1]) for p in paths])
    orig_split = [p.split("/")[0] for p in paths]
    print(f"pool: {n} images; exact-hash duplicate groups: {sum(1 for v in Counter(shas).values() if v > 1)}")

    qn = norm(vecs)
    edges = {}                                  # (i,j) i<j -> (best sim, orientation) for sim >= 0.98
    best = np.full(n, -1.0)
    best_j = np.zeros(n, int)
    best_k = np.zeros(n, int)
    for k, v in enumerate(variants(vecs)):
        s = qn @ norm(np.ascontiguousarray(v)).T
        np.fill_diagonal(s, -1.0)
        m, a = s.max(1), s.argmax(1)
        upd = m > best
        best[upd], best_j[upd], best_k[upd] = m[upd], a[upd], k
        ii, jj = np.where(s >= 0.98)
        for i, j in zip(ii.tolist(), jj.tolist()):
            key = (min(i, j), max(i, j))
            sv = float(s[i, j])
            if key not in edges or sv > edges[key][0]:
                edges[key] = (sv, k)
        print(f"  orientation {k}: edges so far {len(edges)}", flush=True)

    # sensitivity of cluster structure to the threshold
    sens = {}
    for thr in (0.999, 0.995, 0.99, 0.98):
        roots = components(n, edges, thr, shas)
        sizes = Counter(roots)
        sens[str(thr)] = {"n_clusters": len(sizes), "max_cluster_size": max(sizes.values()), "n_images_in_multi_clusters": int(sum(v for v in sizes.values() if v > 1))}
    print("sensitivity:", json.dumps(sens))
    roots = components(n, edges, args.threshold, shas)
    uniq_roots = sorted(set(roots))
    cid = {r: f"C{idx:05d}" for idx, r in enumerate(uniq_roots)}
    members = defaultdict(list)
    for i, r in enumerate(roots):
        members[r].append(i)

    aug = np.array(["aug" in Path(p).name.lower() for p in paths])
    rows = []
    cluster_info = {}
    for r, idxs in members.items():
        labels = {int(cls[i]) for i in idxs}
        has_aug = any(aug[i] for i in idxs)
        cross = len({orig_split[i] for i in idxs}) > 1
        reason = None
        if len(labels) > 1:
            reason = "conflicting class labels"
        elif all(aug[i] for i in idxs):
            reason = "all members are files marked augmented"
        # representative: prefer non-augmented, then Training before Testing, then path order
        cand = sorted(idxs, key=lambda i: (bool(aug[i]), 0 if orig_split[i] == "Training" else 1, paths[i]))
        cluster_info[r] = {"id": cid[r], "size": len(idxs), "labels": sorted(labels), "cross_split": cross, "reason": reason, "rep": cand[0], "has_aug": has_aug}

    # cluster-level, class-stratified split 70/15/15
    rng = np.random.RandomState(args.seed)
    assign = {}
    eligible_by_class = defaultdict(list)
    for r, ci in cluster_info.items():
        if ci["reason"] is None:
            eligible_by_class[ci["labels"][0]].append(r)
    for c in sorted(eligible_by_class):
        rs = sorted(eligible_by_class[c], key=lambda r: cluster_info[r]["id"])
        rng.shuffle(rs)
        nv, nt = int(round(0.15 * len(rs))), int(round(0.15 * len(rs)))
        for t, r in enumerate(rs):
            assign[r] = "val" if t < nv else ("test" if t < nv + nt else "train")

    for i in range(n):
        r = roots[i]
        ci = cluster_info[r]
        part = "quarantine" if ci["reason"] else assign[r]
        ev = f"nearest other image {paths[best_j[i]]} sim {best[i]:.4f} orientation {int(best_k[i])}"
        rows.append({"source_path": paths[i], "original_split": orig_split[i], "class_name": CLASSES[cls[i]], "class_idx": int(cls[i]), "partition": part,
                     "cluster_id": ci["id"], "cluster_size": ci["size"], "cluster_spans_original_train_and_test": ci["cross_split"],
                     "is_representative": bool(ci["rep"] == i and not ci["reason"]), "quarantined": bool(ci["reason"]), "quarantine_reason": ci["reason"] or "",
                     "is_augmented_filename": bool(aug[i]), "similarity_method": METHOD, "threshold": args.threshold,
                     "max_similarity_to_other_image": round(float(best[i]), 5), "matching_evidence": ev})
    with open(out / "split_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # verification + audit summary
    reps = [r for r in rows if r["is_representative"]]
    part_counts = {p: {c: sum(1 for r in reps if r["partition"] == p and r["class_name"] == c) for c in CLASSES} for p in ("train", "val", "test")}
    parts_of_cluster = defaultdict(set)
    for r in rows:
        if r["partition"] != "quarantine":
            parts_of_cluster[r["cluster_id"]].add(r["partition"])
    cross_part_clusters = [c for c, s in parts_of_cluster.items() if len(s) > 1]
    sizes = Counter(ci["size"] for ci in cluster_info.values())
    # cross-partition similarity between representatives (should be < threshold by construction)
    ridx = {p: np.array([paths.index(r["source_path"]) for r in reps if r["partition"] == p]) for p in ("train", "val", "test")}
    cross = {}
    top_pairs = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        bs = np.full(len(ridx[b]), -1.0)
        bj = np.zeros(len(ridx[b]), int)
        bk = np.zeros(len(ridx[b]), int)
        for k, v in enumerate(variants(vecs[ridx[a]])):
            s = norm(vecs[ridx[b]]) @ norm(np.ascontiguousarray(v)).T
            m, am = s.max(1), s.argmax(1)
            u = m > bs
            bs[u], bj[u], bk[u] = m[u], am[u], k
        cross[f"{a}_vs_{b}"] = {"n_query": int(len(bs)), "max_similarity": float(bs.max()), "n_ge_0.98": int((bs >= 0.98).sum()), "n_ge_0.95": int((bs >= 0.95).sum()),
                                "n_ge_0.90": int((bs >= 0.90).sum()), "median": float(np.median(bs))}
        for q in np.argsort(-bs)[:8]:
            top_pairs.append((float(bs[q]), paths[ridx[b][q]], paths[ridx[a][bj[q]]], int(bk[q]), f"{a}-{b}"))
    top_pairs.sort(reverse=True)
    summary = {
        "method": METHOD, "threshold": args.threshold, "pool_images": n, "exact_hash_duplicate_groups": int(sum(1 for v in Counter(shas).values() if v > 1)),
        "n_clusters": len(cluster_info), "cluster_size_histogram_size_to_count": {str(k): v for k, v in sorted(sizes.items())}, "max_cluster_size": max(ci["size"] for ci in cluster_info.values()),
        "clusters_spanning_original_train_and_test": int(sum(ci["cross_split"] for ci in cluster_info.values())),
        "quarantined_clusters": {"conflicting_labels": int(sum(ci["reason"] == "conflicting class labels" for ci in cluster_info.values())),
                                 "all_augmented_filenames": int(sum(ci["reason"] == "all members are files marked augmented" for ci in cluster_info.values()))},
        "quarantined_images": int(sum(1 for r in rows if r["quarantined"])),
        "augmented_filename_images": int(aug.sum()), "threshold_sensitivity": sens,
        "representatives": len(reps), "partition_class_counts_representatives": part_counts,
        "partition_totals": {p: int(sum(v.values())) for p, v in part_counts.items()},
        "clusters_in_more_than_one_partition": len(cross_part_clusters),
        "cross_partition_representative_similarity": cross,
        "caveats": ["heuristic similarity, not ground truth; no patient ids so same-patient neighbouring slices are not detected",
                    "augmented-looking files are quarantined by filename only; augmented files with unmarked names would not be recognised"]}
    json.dump(summary, open(out / "dedup_audit_summary.json", "w"), indent=2)
    cfg = {"seed": args.seed, "ratios": {"train": 0.70, "val": 0.15, "test": 0.15}, "threshold": args.threshold, "representative_rule": "prefer non-augmented filename, then original Training over Testing, then path order",
           "partition_file_list_sha256": {p: hashlib.sha256("\n".join(sorted(r["source_path"] for r in reps if r["partition"] == p)).encode()).hexdigest() for p in ("train", "val", "test")},
           "manifest_sha256": hashlib.sha256((out / "split_manifest.csv").read_bytes()).hexdigest()}
    json.dump(cfg, open(out / "split_config.json", "w"), indent=2)
    json.dump([{"sim": s, "query": q, "match": m, "orientation": o, "partitions": pp} for s, q, m, o, pp in top_pairs[:24]], open(out / "cross_partition_top_pairs.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = top_pairs[:12]
    fig, axes = plt.subplots(2, len(sel), figsize=(1.9 * len(sel), 4.2))
    for j, (s, q, m, o, pp) in enumerate(sel):
        for r_, pth in enumerate((q, m)):
            axes[r_, j].imshow(Image.open(ARCH / pth).convert("L"), cmap="gray")
            axes[r_, j].axis("off")
            axes[r_, j].set_title((f"{pp}\nsim {s:.3f}" if r_ == 0 else "") + f"\n{Path(pth).name[:13]}", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "cross_partition_top_pairs.png", dpi=110)
    print(json.dumps({k: summary[k] for k in ("n_clusters", "max_cluster_size", "quarantined_clusters", "quarantined_images", "representatives", "partition_totals", "partition_class_counts_representatives", "clusters_in_more_than_one_partition", "cross_partition_representative_similarity")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
