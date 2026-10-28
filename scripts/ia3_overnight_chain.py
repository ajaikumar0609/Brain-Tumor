#!/usr/bin/env python3
"""Overnight chain (standard library only): clean cWGAN-GP finishes -> verify -> autopilot reruns stages C and D -> validate.

    python scripts/ia3_overnight_chain.py --dry-run     # show verification result + planned actions, change nothing
    python scripts/ia3_overnight_chain.py               # run (intended to be launched detached; see log)

It never trains anything new except to RESUME the clean GAN from its latest checkpoint if its process died (<= 2 times),
never starts a second controller, never deletes results, and stops with a recorded blocker instead of looping.
Log: outputs/autopilot/overnight_chain.log   Result: outputs/autopilot/OVERNIGHT_RESULT.md
"""
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AP = ROOT / "outputs" / "autopilot"
GAN_DIR = ROOT / "outputs" / "cwgangp_clean_3000"
SPLIT = ROOT / "outputs" / "cwgangp_clean_split"
LOG = AP / "overnight_chain.log"
RESULT = AP / "OVERNIGHT_RESULT.md"
PY = str(ROOT / ".venv" / "bin" / "python")
LAUNCHER = str(ROOT / "scripts" / "run_ia3_autopilot.sh")
T_START = time.time()


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    AP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{ts()} {msg}\n")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def gan_processes() -> list[dict]:
    out = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            argv = shlex.split(parts[2])
        except ValueError:
            argv = parts[2].split()
        if argv and os.path.basename(argv[0]).lower().startswith("python") and any(os.path.basename(a) == "cwgangp_train.py" for a in argv[1:]) \
                and any("cwgangp_clean_3000" in a for a in argv):
            found.append({"pid": int(parts[0]), "etime": parts[1]})
    return found


def verify_clean_gan() -> tuple[bool, list[str], dict]:
    problems, ev, warnings = [], {}, []
    mp, cp = GAN_DIR / "metrics.json", GAN_DIR / "config.json"
    if not mp.exists():
        return False, ["metrics.json missing (training not finished)"], ev
    m, cfg = json.loads(mp.read_text()), json.loads(cp.read_text())
    if m.get("status") != "ok":
        problems.append(f"status={m.get('status')}")
    if m.get("generator_updates_completed") != 3000 or m.get("critic_updates_completed") != 15000:
        problems.append(f"updates {m.get('generator_updates_completed')}/{m.get('critic_updates_completed')} != 3000/15000")
    if m.get("all_losses_and_grads_finite") is not True:
        problems.append("losses/gradients not confirmed finite")
    if m.get("trained_on_testing_dir") is not False:
        problems.append("trained_on_testing_dir is not False")
    if m.get("images_used") != 4480 or set(m.get("dataset_class_counts", {}).values()) != {1120}:
        problems.append(f"training set is not 4,480 images / 1,120 per class: {m.get('images_used')} {m.get('dataset_class_counts')}")
    tl = SPLIT / "cnn_train_files.json"
    vl = SPLIT / "cnn_val_files.json"
    tr, va = json.loads(tl.read_text()), json.loads(vl.read_text())
    sha = sha256_bytes(tl.read_bytes())
    if not str(cfg.get("train_list", "")).endswith("cnn_train_files.json") or cfg.get("train_list_sha256") != sha:
        problems.append("GAN config train_list / sha256 does not match outputs/cwgangp_clean_split/cnn_train_files.json")
    if len(tr) != 4480 or len(va) != 1120 or set(tr) & set(va):
        problems.append(f"CNN split lists invalid: train={len(tr)} val={len(va)} overlap={len(set(tr) & set(va))}")
    if sha256_bytes("\n".join(tr).encode()) != cfg.get("subset_file_list_sha256"):
        warnings.append("subset_file_list_sha256 differs from the hash of the sorted train list (ordering quirk); train_list sha256 matched")
    ck = GAN_DIR / "checkpoint_step_3000.pth"
    if not ck.exists():
        problems.append("checkpoint_step_3000.pth missing")
    else:
        code = ("import torch,sys,json;c=torch.load(sys.argv[1],map_location='cpu',weights_only=True);"
                "print(json.dumps([c['gen_updates'],c['critic_updates'],c['config'].get('train_list_sha256')]))")
        r = subprocess.run([PY, "-I", "-c", code, str(ck)], capture_output=True, text=True)
        try:
            g, c_, s_ = json.loads(r.stdout)
            if (g, c_) != (3000, 15000) or s_ != sha:
                problems.append(f"checkpoint embeds counters {g}/{c_} train_list_sha {s_ and s_[:8]}")
        except Exception:
            problems.append(f"could not read checkpoint: {r.stderr[-150:]}")
    rows = list(csv.DictReader(open(GAN_DIR / "losses.csv")))
    if [int(r["gen_step"]) for r in rows] != list(range(1, 3001)):
        problems.append("losses.csv is not a contiguous 1..3000")
    ev = {"runtime_seconds": m.get("runtime_seconds"), "device": m.get("device"), "train_list_sha256": sha[:16], "warnings": warnings,
          "final_critic_loss": m.get("final_critic_loss"), "final_gen_loss": m.get("final_gen_loss")}
    return not problems, problems, ev


def run_ap(*args: str) -> subprocess.CompletedProcess:
    r = subprocess.run([LAUNCHER, *args], capture_output=True, text=True, cwd=ROOT)
    log(f"autopilot {' '.join(args)} -> rc={r.returncode} {r.stdout.strip()[:200]} {r.stderr.strip()[:200]}")
    return r


def controller_state() -> dict:
    return json.loads((AP / "state.json").read_text())


def controller_alive() -> bool:
    r = subprocess.run([LAUNCHER, "--status"], capture_output=True, text=True, cwd=ROOT)
    return "RUNNING" in (r.stdout.splitlines() or [""])[0]


def write_result(status: str, details: str) -> None:
    lines = [f"# Overnight chain result: {status}", f"written {ts()} (chain started {datetime.fromtimestamp(T_START).isoformat(timespec='seconds')})", "", details, ""]
    try:
        st = controller_state()
        lines += ["## Controller state", f"status: {st['status']}", *[f"- {s}: {v['status']} (attempts {v['attempts']}, repairs {v['repairs']}) {v['error'] or ''}" for s, v in st["stages"].items()],
                  f"claude calls {st['counters']['claude_calls']} cost ${st['counters']['claude_cost_usd']}, training seconds {st['counters']['training_seconds']:.0f}"]
    except Exception as e:
        lines.append(f"(could not read controller state: {e})")
    lines += ["", "Inspect: `./scripts/run_ia3_autopilot.sh --status`, `outputs/autopilot/history.jsonl`, `outputs/autopilot/overnight_chain.log`"]
    RESULT.write_text("\n".join(lines) + "\n")


def edit_config() -> None:
    p = AP / "config.json"
    cfg = json.loads(p.read_text())
    backup = AP / f"config.before_clean_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    backup.write_text(json.dumps(cfg, indent=2))
    cfg.update({"downstream_dir": "outputs/downstream_classification_clean", "downstream_preliminary_dir": "outputs/downstream_classification",
                "downstream_gan_ckpt": "outputs/cwgangp_clean_3000/checkpoint_step_3000.pth", "require_leakage_free_downstream": True})
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, p)
    log(f"controller config updated for the clean experiment (backup {backup.name})")


def relaunch_gan(attempt: int) -> None:
    cks = sorted(GAN_DIR.glob("checkpoint_step_*.pth"))
    cmd = [PY, "-u", "src/cwgangp_train.py", "--subset-size", "0", "--train-list", "outputs/cwgangp_clean_split/cnn_train_files.json",
           "--gen-steps", "3000", "--batch-size", "16", "--seed", "42", "--sample-every", "500", "--checkpoint-every", "500",
           "--out-dir", "outputs/cwgangp_clean_3000"]
    if cks:
        cmd += ["--resume", str(cks[-1])]
    log(f"GAN process gone without metrics.json: relaunch attempt {attempt} from {cks[-1].name if cks else 'scratch'}")
    with open(GAN_DIR / "train.log", "a") as lf:
        subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    log(f"chain started pid={os.getpid()} dry_run={a.dry_run}")
    if a.dry_run:
        procs = gan_processes()
        ok, problems, ev = verify_clean_gan()
        print(json.dumps({"gan_processes": procs, "verify_ok": ok, "problems": problems, "evidence": ev}, indent=1))
        print("planned: wait for GAN -> verify -> edit config -> `--rerun C D` -> `--resume` -> wait -> validate outputs")
        return 0

    retries = 0
    while True:                                           # 1. wait for the clean GAN, verify, retry-resume at most twice
        if gan_processes():
            time.sleep(30)
            continue
        ok, problems, ev = verify_clean_gan()
        if ok:
            log(f"clean GAN verified: {json.dumps(ev)}")
            break
        if not (GAN_DIR / "metrics.json").exists() and retries < 2:
            retries += 1
            relaunch_gan(retries)
            continue
        log(f"BLOCKED: clean GAN did not verify: {problems}")
        write_result("BLOCKED before the clean downstream experiment", "The clean cWGAN-GP did not pass verification:\n" + "\n".join(f"- {p}" for p in problems))
        return 2

    if controller_alive():
        write_result("BLOCKED", "An autopilot controller is already running; refusing to start a second one.")
        return 2
    edit_config()                                         # 2. controller -> clean experiment, rerun C and D only
    if run_ap("--rerun", "C", "D").returncode != 0:
        write_result("BLOCKED", "autopilot --rerun C D failed; see the chain log.")
        return 2
    resumes = 0
    deadline = time.time() + 4 * 3600
    while True:
        if run_ap("--resume").returncode != 0:
            write_result("BLOCKED", "autopilot --resume failed; see the chain log.")
            return 2
        time.sleep(30)
        quiet = 0
        while time.time() < deadline:                     # 3. wait for a terminal controller state
            st = controller_state()
            alive = controller_alive()
            if st["status"] in ("finished", "blocked", "paused", "stopped") and not alive:
                break
            if st["status"] == "running" and not alive:
                quiet += 1
                if quiet >= 3:
                    break
            else:
                quiet = 0
            time.sleep(30)
        st = controller_state()
        if st["status"] == "running" and not controller_alive() and resumes < 2:
            resumes += 1
            log(f"controller died while 'running'; resume {resumes}/2")
            continue
        break

    st = controller_state()                               # 4. validate outputs
    problems = []
    if st["status"] != "finished":
        problems.append(f"controller ended in state '{st['status']}'")
    clean = ROOT / "outputs/downstream_classification_clean/summary.json"
    summ = json.loads(clean.read_text()) if clean.exists() else None
    if not summ or summ["config"].get("leakage_free") is not True:
        problems.append("clean downstream summary missing or not leakage-free")
    for f in ("outputs/final_report.md", "outputs/ia3_slides_outline.md", "outputs/ia3_viva_questions.md",
              "outputs/gan_comparison/final_comparison.csv", "outputs/gan_comparison/final_comparison.png"):
        p = ROOT / f
        if not p.exists() or p.stat().st_mtime < T_START:
            problems.append(f"{f} missing or not regenerated after the clean run")
    prelim_ok = (ROOT / "outputs/downstream_classification/summary.json").exists() and list(ROOT.glob("outputs/final_report.prev-*.md"))
    if not prelim_ok:
        problems.append("preliminary results / previous report version not preserved")
    detail = []
    if summ:
        for c, s in summ["conditions"].items():
            detail.append(f"- {c}: test accuracy {s['accuracy']['mean']:.4f} +- {s['accuracy']['std']:.4f} (n_train {s['n_train']}, {s['n_runs']} runs), macro-F1 {s['macro_f1']['mean']:.4f}")
        detail.append(f"- leakage audit: {json.dumps(summ['config'].get('leakage_audit'))}")
    if problems:
        log(f"chain finished with problems: {problems}")
        write_result("FINISHED WITH PROBLEMS", "Problems:\n" + "\n".join(f"- {p}" for p in problems) + "\n\nClean downstream results (if any):\n" + "\n".join(detail))
        return 1
    log("chain finished: all outputs validated")
    write_result("SUCCESS (outputs validated; texts still need a human read)", "Clean (leakage-free) downstream experiment:\n" + "\n".join(detail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
