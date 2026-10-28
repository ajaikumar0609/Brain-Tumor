#!/usr/bin/env python3
"""Final verification of the de-duplicated experiment (read-only). Prints PASS/FAIL per check and writes verification.json.

    python scripts/verify_dedup_experiment.py
"""
import csv, glob, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "outputs/deduplicated_experiment"
res = []


def check(name, ok, detail=""):
    res.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


rows = list(csv.DictReader(open(D / "split_manifest.csv")))
reps = [r for r in rows if r["is_representative"] == "True"]
part = {p: [r["source_path"] for r in reps if r["partition"] == p] for p in ("train", "val", "test")}
# 1 cluster-disjoint split
cl = {}
for r in rows:
    if r["partition"] != "quarantine":
        cl.setdefault(r["cluster_id"], set()).add(r["partition"])
check("1a no cluster appears in more than one partition", all(len(s) == 1 for s in cl.values()), f"{len(cl)} clusters")
check("1b representatives: one per cluster, no file in two partitions", len({r['cluster_id'] for r in reps}) == len(reps) and len(set(part['train']) | set(part['val']) | set(part['test'])) == sum(map(len, part.values())),
      f"train {len(part['train'])} val {len(part['val'])} test {len(part['test'])}")
check("1c quarantined clusters (label conflicts / augmented) excluded from every partition", all(r["partition"] == "quarantine" for r in rows if r["quarantined"] == "True") and not any(r["quarantined"] == "True" and r["is_representative"] == "True" for r in rows))
cfg = json.load(open(D / "split_config.json"))
for p in ("train", "val", "test"):
    check(f"1d {p} file-list hash matches split_config", hashlib.sha256("\n".join(sorted(part[p])).encode()).hexdigest() == cfg["partition_file_list_sha256"][p])
check("1e manifest hash matches split_config", hashlib.sha256((D / "split_manifest.csv").read_bytes()).hexdigest() == cfg["manifest_sha256"])
cls = {c: {p: sum(1 for x in part[p] if x.split('/')[1] == c) for p in part} for c in ("glioma", "meningioma", "notumor", "pituitary")}
print("   per-class counts:", json.dumps(cls))
heldout = set(part["val"]) | set(part["test"]) | {r["source_path"] for r in rows if r["partition"] == "quarantine"}
# 2/3 GANs trained only on the train partition
sys.path.insert(0, str(ROOT / "src"))
from data_pipeline import ManifestDataset, rel_path  # noqa: E402
import torch  # noqa: E402
recomputed = hashlib.sha256("\n".join(rel_path(p) for p, _ in ManifestDataset(D / "split_manifest.csv", "train", 64, False).samples).encode()).hexdigest()
for m, d in (("DCGAN", "dcgan"), ("WGAN-GP", "wgangp"), ("cWGAN-GP", "cwgangp")):
    ck = torch.load(D / "gan" / d / "checkpoint_step_3000.pth", map_location="cpu", weights_only=True)
    gu, cu = ck.get("gen_updates", ck.get("batches")), ck.get("critic_updates", ck.get("batches"))
    mt = json.load(open(D / "gan" / d / "metrics.json"))
    check(f"2/3 {m}: trained exactly on the manifest train partition (hash), never on val/test/quarantine", ck["config"].get("subset_file_list_sha256") == recomputed and mt["images_used"] == len(part["train"]) and mt["trained_on_testing_dir"] is False,
          f"images_used {mt['images_used']}")
    check(f"9 {m}: counters", gu == 3000 and cu == (3000 if d == "dcgan" else 15000) and mt["status"] == "ok" and mt["all_losses_and_grads_finite"] is True, f"gen {gu} critic/disc {cu} device {mt['device']} runtime {mt['runtime_seconds']}s")
    check(f"9 {m}: sample grids at 0,500,...,3000", len(glob.glob(str(D / 'gan' / d / 'samples_step_*.png'))) >= 7)
# 4/5 classifier + test usage
ds = json.load(open(D / "downstream/summary.json"))
dc_cfg = ds["config"]
check("4a downstream used this manifest", dc_cfg["manifest_sha256"] == cfg["manifest_sha256"])
check("4b GAN used for B trained on exactly the train partition", dc_cfg["leakage_audit"]["gan_trained_on_exactly_the_train_partition"] is True)
cm = json.load(open(D / "downstream/resampling_manifest_C.json"))
check("4c Condition C resampling uses only train-partition files", set(cm["file_multiplicity"]) <= set(part["train"]), f"{cm['unique_files_used']}/{cm['unique_available']} files, max mult {cm['max_multiplicity']}")
check("4d composition: B and C have identical per-class counts and size", dc_cfg["composition"]["B_real_plus_cwgangp_synthetic"] == dc_cfg["composition"]["C_real_only_matched_composition"], str(dc_cfg["composition"]))
preds = sorted(glob.glob(str(D / "downstream/runs/*_test_predictions.json")))
ok5 = True
for f in preds:
    p = json.load(open(f))
    ok5 &= set(p["test_files"]) == set(part["test"]) and not (set(p["test_files"]) & (set(part["train"]) | set(part["val"])))
check("5 every run was evaluated on exactly the test partition (disjoint from train/val); predictions saved", ok5 and len(preds) == 3 * len(dc_cfg["seeds"]), f"{len(preds)} prediction files")
ev = json.load(open(D / "gan_comparison/config.json"))
check("5b FID/KID reference = test partition; GAN checkpoints match manifest", ev["n_ref"] == len(part["test"]) and all(v["matches_manifest_train_partition"] for v in ev["checkpoints"].values()) and ev["manifest_sha256"] == cfg["manifest_sha256"])
# 10 earlier results preserved
prelim = ROOT / "outputs/downstream_classification/summary.json"
check("10 earlier results preserved (preliminary summary, previous report versions, old checkpoints)", prelim.exists() and hashlib.sha256(prelim.read_bytes()).hexdigest().startswith("77cd5d5a6b14")
      and (ROOT / "outputs/cwgangp_3000/checkpoint_step_3000.pth").exists() and len(glob.glob(str(ROOT / "outputs/*.prev-*"))) >= 6)
json.dump(res, open(D / "verification.json", "w"), indent=2)
print(f"\n{sum(r['pass'] for r in res)}/{len(res)} checks passed")
sys.exit(0 if all(r["pass"] for r in res) else 1)
