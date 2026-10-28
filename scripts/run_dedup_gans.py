#!/usr/bin/env python3
"""Sequentially retrain DCGAN, WGAN-GP and cWGAN-GP on the de-duplicated TRAIN partition (resumable, verified, stdlib only).

    python scripts/run_dedup_gans.py
Identical recipes/seed to the earlier 3,000-update runs; only the training data (--manifest) differs. Each stage is skipped if its verified
checkpoint already exists; a partial output directory is archived (renamed), never overwritten; cWGAN-GP resumes from its latest
checkpoint at most twice; any other failure stops the driver with the reason recorded. Log: outputs/deduplicated_experiment/gan/driver.log"""
import csv, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "outputs/deduplicated_experiment"
G = D / "gan"
M = D / "split_manifest.csv"
PY = str(ROOT / ".venv/bin/python")
G.mkdir(parents=True, exist_ok=True)
LOG = G / "driver.log"


def log(m):
    with open(LOG, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {m}\n")


def train_counts():
    rows = [r for r in csv.DictReader(open(M)) if r["partition"] == "train" and r["is_representative"] == "True"]
    c = {}
    for r in rows:
        c[r["class_name"]] = c.get(r["class_name"], 0) + 1
    return len(rows), c


N_TRAIN, CLASS_COUNTS = train_counts()
STAGES = {
    "dcgan": ["src/dcgan_train.py", "--manifest", str(M), "--subset-size", "0", "--epochs", "60", "--max-batches", "3000", "--batch-size", "16", "--seed", "42",
              "--sample-every", "500", "--checkpoint-every", "500", "--out-dir", str(G / "dcgan")],
    "wgangp": ["src/wgangp_train.py", "--manifest", str(M), "--subset-size", "0", "--gen-steps", "3000", "--batch-size", "16", "--seed", "42",
               "--sample-every", "500", "--checkpoint-every", "500", "--out-dir", str(G / "wgangp")],
    "cwgangp": ["src/cwgangp_train.py", "--manifest", str(M), "--subset-size", "0", "--gen-steps", "3000", "--batch-size", "16", "--seed", "42",
                "--sample-every", "500", "--checkpoint-every", "500", "--out-dir", str(G / "cwgangp")],
}


def verify(name):
    d = G / name
    mp = d / "metrics.json"
    if not mp.exists():
        return False, ["metrics.json missing"]
    m = json.loads(mp.read_text())
    p = []
    gen = m.get("generator_updates_completed", m.get("batches_completed"))
    if m.get("status") != "ok" or gen != 3000:
        p.append(f"status={m.get('status')} gen_updates={gen}")
    if m.get("all_losses_and_grads_finite") is not True:
        p.append("losses/gradients not confirmed finite")
    if m.get("trained_on_testing_dir") is not False:
        p.append("trained_on_testing_dir not False")
    if m.get("images_used") != N_TRAIN:
        p.append(f"images_used {m.get('images_used')} != {N_TRAIN}")
    cc = m.get("dataset_class_counts", m.get("subset_class_counts"))
    if cc != CLASS_COUNTS:
        p.append(f"class counts {cc} != {CLASS_COUNTS}")
    if not (d / "checkpoint_step_3000.pth").exists():
        p.append("checkpoint_step_3000.pth missing")
    return not p, p


def running(name):
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return any(STAGES[name][0] in l and str(G / name) in l and "python" in l.lower() for l in out.splitlines())


def run(name, resume=None):
    cmd = [PY, "-u", *STAGES[name]] + (["--resume", str(resume)] if resume else [])
    (G / name).mkdir(parents=True, exist_ok=True)
    log(f"START {name} {'resume ' + resume.name if resume else ''}")
    with open(G / f"{name}_train.log", "a") as lf:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode
    log(f"END {name} rc={rc}")
    return rc


def main():
    log(f"driver start pid={os.getpid()} train={N_TRAIN} {CLASS_COUNTS}")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        log("ABORT: PYTORCH_ENABLE_MPS_FALLBACK is set")
        return 2
    for name in ("dcgan", "wgangp", "cwgangp"):
        ok, pr = verify(name)
        if ok:
            log(f"SKIP {name}: already verified")
            continue
        if running(name):
            log(f"ABORT: {name} already running (not starting a duplicate)")
            return 3
        d = G / name
        retries = 0
        existing = sorted(d.glob("checkpoint_step_*.pth")) if d.exists() else []
        resume = existing[-1] if (name == "cwgangp" and existing) else None
        if d.exists() and any(d.iterdir()) and resume is None:
            arch = d.with_name(f"{name}.partial-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            d.rename(arch)
            log(f"archived partial {name} output -> {arch.name}")
        while True:
            rc = run(name, resume)
            ok, pr = verify(name)
            if rc == 0 and ok:
                log(f"VERIFIED {name}")
                break
            log(f"FAILED {name}: rc={rc} problems={pr}")
            ck = sorted(d.glob("checkpoint_step_*.pth"))
            if name == "cwgangp" and retries < 2 and ck:
                retries += 1
                resume = ck[-1]
                log(f"retry {retries}/2 from {resume.name}")
                continue
            log(f"STOP: {name} could not be completed; partial outputs preserved")
            return 1
    log("ALL GANS VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
