#!/usr/bin/env python3
"""After the three de-duplicated GANs are verified: re-verify checkpoints -> downstream A/B/C -> FID/KID -> verify_dedup_experiment.py.
Waits for 'ALL GANS VERIFIED' in gan/driver.log; stops at the first failure (logs kept); skips completed stages; never starts a duplicate job.
Log: outputs/deduplicated_experiment/stage2.log"""
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "outputs/deduplicated_experiment"
PY = str(ROOT / ".venv/bin/python")
LOG = D / "stage2.log"


def log(m):
    with open(LOG, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {m}\n")


def procs(*names):
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "python" in l.lower() and any(n in l for n in names) and "run_dedup_stage2" not in l]


def run(cmd, logfile):
    log("RUN " + " ".join(cmd[1:3]))
    with open(logfile, "a") as lf:
        return subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode


PRECHECK = r'''
import json, sys, torch, hashlib
sys.path.insert(0, "src")
from pathlib import Path
from data_pipeline import ManifestDataset, rel_path
D = Path("outputs/deduplicated_experiment")
h = hashlib.sha256("\n".join(rel_path(p) for p, _ in ManifestDataset(D / "split_manifest.csv", "train", 64, False).samples).encode()).hexdigest()
out, ok = {}, True
for m, d, crit in (("DCGAN", "dcgan", 3000), ("WGAN-GP", "wgangp", 15000), ("cWGAN-GP", "cwgangp", 15000)):
    ck = torch.load(D / "gan" / d / "checkpoint_step_3000.pth", map_location="cpu", weights_only=True)
    met = json.load(open(D / "gan" / d / "metrics.json"))
    gu, cu = ck.get("gen_updates", ck.get("batches")), ck.get("critic_updates", ck.get("batches"))
    fin = all(torch.isfinite(v).all().item() for v in list(ck["G"].values()) + list(ck["D"].values()) if v.is_floating_point())
    r = {"generator_updates": gu, "critic_or_discriminator_updates": cu, "weights_finite": fin, "reload_weights_only": True, "train_hash_matches_manifest": ck["config"].get("subset_file_list_sha256") == h,
         "images_used": met.get("images_used"), "trained_on_testing_dir": met.get("trained_on_testing_dir"), "all_losses_and_grads_finite": met.get("all_losses_and_grads_finite"),
         "status": met.get("status"), "device": met.get("device"), "runtime_seconds": met.get("runtime_seconds"), "resumed_from": met.get("resumed_from")}
    r["pass"] = gu == 3000 and cu == crit and fin and r["train_hash_matches_manifest"] and r["images_used"] == 2311 and r["trained_on_testing_dir"] is False and r["all_losses_and_grads_finite"] is True and r["status"] == "ok"
    ok &= r["pass"]
    out[m] = r
json.dump({"manifest_train_hash": h, "models": out, "all_pass": ok}, open(D / "gan/gan_verification.json", "w"), indent=2)
print(json.dumps(out, indent=1)); sys.exit(0 if ok else 1)
'''


def main():
    log(f"stage2 start pid={os.getpid()}")
    t0 = time.time()
    while True:
        t = (D / "gan/driver.log").read_text() if (D / "gan/driver.log").exists() else ""
        if "ALL GANS VERIFIED" in t.splitlines()[-1] if t else False:
            break
        if any(k in t.splitlines()[-1] for k in ("STOP:", "ABORT")) if t else False:
            log("STOP: GAN driver stopped; not starting stage 2")
            return 1
        if time.time() - t0 > 4 * 3600:
            log("STOP: timed out waiting for the GAN driver")
            return 1
        time.sleep(15)
    log("GANs verified by driver; pre-checking checkpoints")
    if run([PY, "-c", PRECHECK], D / "gan/gan_verification.log") != 0:
        log("STOP: checkpoint verification failed (see gan/gan_verification.log)")
        return 1
    log("PASS gan checkpoints")
    if not (D / "downstream/summary.json").exists():
        if procs("dedup_downstream"):
            log("STOP: a downstream job is already running")
            return 1
        rc = run([PY, "-u", "src/dedup_downstream.py"], D / "downstream_run.log")
        if rc != 0 or not (D / "downstream/summary.json").exists():
            log(f"STOP: downstream failed rc={rc} (see downstream_run.log; partial outputs preserved)")
            return 1
    log("PASS downstream")
    if not (D / "gan_comparison/metrics.json").exists():
        if procs("dedup_eval_fid_kid"):
            log("STOP: an eval job is already running")
            return 1
        rc = run([PY, "-u", "src/dedup_eval_fid_kid.py"], D / "gan_comparison_run.log")
        if rc != 0 or not (D / "gan_comparison/metrics.json").exists():
            log(f"STOP: FID/KID failed rc={rc} (see gan_comparison_run.log; partial outputs preserved)")
            return 1
    log("PASS fid_kid")
    rc = run([PY, "scripts/verify_dedup_experiment.py"], D / "verification.log")
    log(f"verify_dedup_experiment rc={rc} (see verification.log / verification.json)")
    log("STAGE2 DONE" if rc == 0 else "STAGE2 DONE WITH FAILED CHECKS")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
