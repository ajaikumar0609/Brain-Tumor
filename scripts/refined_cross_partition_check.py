#!/usr/bin/env python3
"""Re-score EVERY val/test-rep -> best train-rep pair (32x32 sim >= 0.90) at 96x96 with +/-4px shifts; count near-copies that survived the split."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("probe", "scripts/refined_similarity_probe.py")
src = open("scripts/refined_similarity_probe.py").read().split("T = np.stack")[0]
ns = {"__file__": str(Path("scripts/refined_similarity_probe.py").resolve()), "__name__": "probe"}
exec(compile(src, "probe", "exec"), ns)
im, refined, norm, tr, va = ns["im"], ns["refined"], ns["norm"], ns["tr"], ns["va"]
ROOT = Path(".").resolve()
rows = list(csv.DictReader(open(ROOT / "outputs/deduplicated_experiment/split_manifest.csv")))
te = [r["source_path"] for r in rows if r["partition"] == "test" and r["is_representative"] == "True"]
T = np.stack([im(p, 32) for p in tr])
Tv = [T, T[:, :, ::-1], T[:, ::-1, :], np.rot90(T, 1, (1, 2)), np.rot90(T, 2, (1, 2)), np.rot90(T, 3, (1, 2)), T.transpose(0, 2, 1), T.transpose(0, 2, 1)[:, ::-1, ::-1]]
res = {}
for name, q in (("val", va), ("test", te)):
    Q = np.stack([im(p, 32) for p in q]); bs = np.full(len(Q), -1.0); bj = np.zeros(len(Q), int); bk = np.zeros(len(Q), int)
    for k, v in enumerate(Tv):
        s = norm(Q) @ norm(np.ascontiguousarray(v)).T
        m, a = s.max(1), s.argmax(1); u = m > bs; bs[u], bj[u], bk[u] = m[u], a[u], k
    cand = np.where(bs >= 0.90)[0]
    r = np.array([refined(q[i], tr[bj[i]], int(bk[i])) for i in cand])
    top = np.argsort(-r)[:6]
    res[name] = {"n_reps": len(q), "n_scored": int(len(cand)), "refined_ge_0.99": int((r >= 0.99).sum()), "refined_ge_0.98": int((r >= 0.98).sum()), "refined_ge_0.97": int((r >= 0.97).sum()),
                 "refined_ge_0.95": int((r >= 0.95).sum()), "max_refined": float(r.max()), "p50": float(np.median(r)),
                 "top6": [{"query": q[cand[t]], "match": tr[bj[cand[t]]], "coarse": round(float(bs[cand[t]]), 4), "refined": round(float(r[t]), 4)} for t in top]}
    print(name, {k: v for k, v in res[name].items() if k != "top6"}, flush=True)
    for t in res[name]["top6"]: print("   ", t)
json.dump(res, open(ROOT / "outputs/deduplicated_experiment/refined_cross_partition_check.json", "w"), indent=1)
