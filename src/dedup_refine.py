#!/usr/bin/env python3
"""Stage 2 of the de-duplicated split: quarantine val/test representatives that still have a near-copy in another partition.

Re-scores each held-out representative's best 32x32 match (>=0.90) in a higher-priority partition at 96x96 with +/-4px shifts in the matched
orientation; pairs with refined similarity >= --refined-threshold (default 0.97; true copies score ~1.0, merely similar scans <= ~0.95)
send the lower-priority image (val/test, never train; for val-vs-test the val image) to quarantine. Writes the final manifest; the stage-1
manifest is kept as split_manifest.stage1.csv."""
import argparse, csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "archive"
CL = ["glioma", "meningioma", "notumor", "pituitary"]


def im(p, s):
    return np.asarray(Image.open(A / p).convert("L").resize((s, s), Image.BILINEAR), np.float32)


def variants(a):
    return [a, a[:, ::-1], a[::-1, :], np.rot90(a, 1), np.rot90(a, 2), np.rot90(a, 3), a.T, a.T[::-1, ::-1]]


def norm(a):
    v = a.reshape(len(a), -1)
    v = v - v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def refined(pa, pb, k, s=96, sh=4):
    a, b = im(pa, s), variants(im(pb, s))[k]
    best = -1.0
    for dy in range(-sh, sh + 1):
        for dx in range(-sh, sh + 1):
            x = a[max(0, dy):s + min(0, dy), max(0, dx):s + min(0, dx)]
            y = b[max(0, -dy):s + min(0, -dy), max(0, -dx):s + min(0, -dx)]
            x, y = x - x.mean(), y - y.mean()
            best = max(best, float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)))
    return best


def best_match(q, ref):
    Q, R = np.stack([im(p, 32) for p in q]), np.stack([im(p, 32) for p in ref])
    bs, bj, bk = np.full(len(Q), -1.0), np.zeros(len(Q), int), np.zeros(len(Q), int)
    for k, v in enumerate([R, R[:, :, ::-1], R[:, ::-1, :], np.rot90(R, 1, (1, 2)), np.rot90(R, 2, (1, 2)), np.rot90(R, 3, (1, 2)), R.transpose(0, 2, 1), R.transpose(0, 2, 1)[:, ::-1, ::-1]]):
        s = norm(Q) @ norm(np.ascontiguousarray(v)).T
        m, a = s.max(1), s.argmax(1)
        u = m > bs
        bs[u], bj[u], bk[u] = m[u], a[u], k
    return bs, bj, bk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=ROOT / "outputs/deduplicated_experiment")
    ap.add_argument("--refined-threshold", type=float, default=0.97)
    args = ap.parse_args()
    d = args.dir
    if (d / "split_manifest.stage1.csv").exists():
        print("already refined; refusing to run twice")
        return 2
    rows = list(csv.DictReader(open(d / "split_manifest.csv")))
    shutil.copy2(d / "split_manifest.csv", d / "split_manifest.stage1.csv")
    reps = {p: [r["source_path"] for r in rows if r["partition"] == p and r["is_representative"] == "True"] for p in ("train", "val", "test")}
    moves, scored = {}, {}
    for name, q, ref in (("val", reps["val"], reps["train"]), ("test", reps["test"], reps["train"]), ("val_vs_test", reps["val"], reps["test"])):
        bs, bj, bk = best_match(q, ref)
        cand = np.where(bs >= 0.90)[0]
        r = np.array([refined(q[i], ref[bj[i]], int(bk[i])) for i in cand])
        scored[name] = {"n_queries": len(q), "n_scored": int(len(cand)), "refined_ge_0.99": int((r >= 0.99).sum()), "refined_ge_0.98": int((r >= 0.98).sum()),
                        "refined_ge_0.97": int((r >= 0.97).sum()), "refined_ge_0.95": int((r >= 0.95).sum()), "max_refined": float(r.max())}
        for t in np.where(r >= args.refined_threshold)[0]:
            i = cand[t]
            moves[q[i]] = f"refined near-copy of {'train' if name != 'val_vs_test' else 'test'} representative {ref[bj[i]]} (refined sim {r[t]:.4f}, coarse {bs[i]:.4f}, orientation {int(bk[i])})"
        print(name, scored[name], flush=True)
    for r in rows:
        if r["source_path"] in moves:
            r["partition"], r["quarantined"], r["is_representative"], r["quarantine_reason"] = "quarantine", "True", "False", moves[r["source_path"]]
    with open(d / "split_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    new = {p: sorted(r["source_path"] for r in rows if r["partition"] == p and r["is_representative"] == "True") for p in ("train", "val", "test")}
    counts = {p: {c: sum(1 for x in v if x.split("/")[1] == c) for c in CL} for p, v in new.items()}
    s = json.load(open(d / "dedup_audit_summary.json"))
    s["stage2_refinement"] = {"refined_threshold": args.refined_threshold, "method": "96x96 shift-tolerant (+/-4px) correlation in the coarse-matched orientation for best 32x32 matches >=0.90", "scored": scored,
                              "quarantined_images": moves, "n_quarantined_by_refinement": len(moves)}
    s["partition_class_counts_representatives"] = counts
    s["partition_totals"] = {p: len(v) for p, v in new.items()}
    s["quarantined_images"] = int(sum(1 for r in rows if r["quarantined"] == "True"))
    s["representatives"] = int(sum(1 for r in rows if r["is_representative"] == "True"))
    json.dump(s, open(d / "dedup_audit_summary.json", "w"), indent=2)
    c = json.load(open(d / "split_config.json"))
    c["partition_file_list_sha256"] = {p: hashlib.sha256("\n".join(v).encode()).hexdigest() for p, v in new.items()}
    c["manifest_sha256"] = hashlib.sha256((d / "split_manifest.csv").read_bytes()).hexdigest()
    c["stage1_manifest_sha256"] = hashlib.sha256((d / "split_manifest.stage1.csv").read_bytes()).hexdigest()
    c["refined_threshold"] = args.refined_threshold
    json.dump(c, open(d / "split_config.json", "w"), indent=2)
    print("final partition totals:", s["partition_totals"], "| per-class:", json.dumps(counts))
    print("quarantined by refinement:", json.dumps(moves, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
