#!/usr/bin/env python3
"""Controller mechanics test in a SCRATCH project (temp dir). Uses stub scripts and a stub `claude`; touches nothing real.

    python scripts/test_ia3_autopilot.py [--real-claude-decision]

The stubs write FIXTURE artifacts (fake numbers) only inside the temp dir to exercise the controller's logic:
process detection, waiting, stage skipping, retry/repair limits, out-of-scope-edit revert, stop/resume, output versioning.
--real-claude-decision additionally makes ONE real `claude -p` decision call (no tools) to prove the CLI wiring.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTRL = HERE / "ia3_autopilot.py"
REAL = "--real-claude-decision" in sys.argv
results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


STUB_EVAL = r'''
import json, sys, time, pathlib
d = pathlib.Path("outputs/gan_comparison"); d.mkdir(parents=True, exist_ok=True)
sleep = float(pathlib.Path("stub_eval_sleep.txt").read_text()) if pathlib.Path("stub_eval_sleep.txt").exists() else 1.0
if pathlib.Path("stub_eval_fail.flag").exists():
    print("Traceback (most recent call last):\n  stub failure"); sys.exit(1)
time.sleep(sleep)
agg = {k: {"fid_mean": 100.0 + i, "fid_std_across_seeds": 1.0, "kid_mean": 0.1, "kid_std_across_seeds": 0.001,
           "kid_mean_within_seed_subset_std": 0.001, "n_seeds": 3, "n_generated_per_seed": 1600, "n_real": 1600}
       for i, k in enumerate(["Real train (class-balanced, reference)", "DCGAN", "WGAN-GP", "cWGAN-GP"])}
(d / "metrics.json").write_text(json.dumps({"aggregate": agg, "per_class_cwgangp": [], "class_confusion_kid_cwgangp": {}}))
(d / "comparison_config.json").write_text(json.dumps({"real_reference": "archive/Testing FIXTURE", "image_preprocessing": "fixture",
    "generated_samples": {"x": 1}, "generation_seeds": [1], "feature_extractor": "fixture", "implementation": "fixture",
    "checkpoints": {m: {"generator_updates": 3000} for m in ["DCGAN", "WGAN-GP", "cWGAN-GP"]}}))
for f in ["fid_kid_results.csv", "fid_kid_bar_chart.png", "sample_grids_real_vs_generated.png", "per_class_fid_kid_cwgangp.csv", "class_confusion_kid_cwgangp.csv"]:
    (d / f).write_text("fixture")
print("done")
'''

STUB_CLAUDE = r'''#!/usr/bin/env python3
import json, sys, pathlib
prompt = sys.stdin.read()
calls = pathlib.Path("stub_claude_calls.log"); calls.open("a").write(prompt[:20].replace("\n", " ") + "\n")
def out(**kw):
    base = {"is_error": False, "subtype": "success", "num_turns": 1, "total_cost_usd": 0.01, "permission_denials": [], "result": ""}
    base.update(kw); print(json.dumps(base))
if prompt.startswith("DECISION"):
    stop = pathlib.Path("stub_claude_stop.flag").exists()
    out(structured_output={"decision": "stop" if stop else "continue", "rationale": "stub", "concerns": []})
elif prompt.startswith("REPAIR"):
    # repair: fix the stub (allowed file) and ALSO stray into an out-of-scope file (must be reverted by the controller)
    pathlib.Path("stub_eval_fail.flag").unlink(missing_ok=True)
    pathlib.Path("src/eval_gan_fid_kid.py").write_text(pathlib.Path("src/eval_gan_fid_kid.py").read_text() + "\n# repaired by stub\n")
    pathlib.Path("src/other_file.py").write_text("# out of scope edit\n")
    out(result="stub repair")
elif prompt.startswith("ANALYSIS"):
    out(result="## Summary\n" + "x" * 400 + "\n## Quantitative results\nDCGAN is best_fid_model fixture. " + "y" * 400 +
        "\n## What the image grids show\n" + "z" * 300 + "\n## Class conditioning\n" + "c" * 300 + "\n## Limitations\n" + "l" * 300 +
        "\n## What the results support and do not support\n" + "s" * 300)
else:
    out(structured_output={"final_report": "# Title\nabstract results limitations conclusion " + "r" * 900, "slides_outline": "slide 1 " + "s" * 900,
                           "viva_questions": "Q1? " + "q" * 900})
'''


def ctl(root, *args, check_rc=False):
    r = subprocess.run([sys.executable, str(CTRL), "--root", str(root), *args], capture_output=True, text=True)
    return r


def status_json(root):
    return json.loads((root / "outputs/autopilot/state.json").read_text())


def wait_for(cond, timeout=60, step=0.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


def make_root():
    root = Path(tempfile.mkdtemp(prefix="ia3_autopilot_test_"))
    (root / "src").mkdir()
    (root / "scripts").mkdir()
    (root / "src/eval_gan_fid_kid.py").write_text(STUB_EVAL)
    (root / "src/downstream_classification.py").write_text("print('stub downstream')\n")
    (root / "src/other_file.py").write_text("# original\n")
    shutil.copy(HERE / "ia3_final_artifacts.py", root / "scripts/ia3_final_artifacts.py")
    for run in ["dcgan_3000", "wgangp_3000", "cwgangp_3000"]:
        d = root / "outputs" / run
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"status": "ok", "device": "mps", "generator_updates_completed": 3000, "batches_completed": 3000,
                                                    "critic_updates_completed": 15000, "all_losses_and_grads_finite": True,
                                                    "trained_on_testing_dir": False, "runtime_seconds": 1.0, "images_used": 5600}))
        (d / "checkpoint_step_3000.pth").write_text("fixture")
    stub = root / "stub_claude.py"
    stub.write_text(STUB_CLAUDE)
    stub.chmod(0o755)
    (root / "outputs/autopilot").mkdir(parents=True)
    cfg = {"poll_seconds": 1, "poll_max_seconds": 2, "python_bin": sys.executable, "claude_bin": str(stub), "verify_checkpoint_counters": False,
           "max_repairs_per_stage": 2, "stage_timeouts_seconds": {"A": 60, "B": 60, "C": 60, "D": 60}}
    (root / "outputs/autopilot/config.json").write_text(json.dumps(cfg))
    return root


def stub_claude_cwd_wrapper(root):
    """The stub reads/writes relative to cwd; the controller runs claude with cwd=root, so nothing else is needed."""


def fake_valid_downstream(root):
    d = root / "outputs/downstream_classification"
    (d / "runs").mkdir(parents=True, exist_ok=True)
    cond = lambda n: {"n_runs": 3, "n_train": n, "accuracy": {"mean": 0.9, "std": 0.01, "values": [0.9, 0.9, 0.9]}, "macro_f1": {"mean": 0.9, "std": 0.01, "values": [0.9] * 3},
                      "weighted_f1": {"mean": 0.9, "std": 0.01, "values": [0.9] * 3}}
    (d / "summary.json").write_text(json.dumps({"config": {"smoke": False}, "conditions": {"real_only": cond(4480), "real_plus_cwgangp_synthetic": cond(10080)},
                                                 "paired_accuracy_difference_synthetic_minus_real_only": {"per_seed": [0, 0, 0], "mean": 0.0}}))
    for i in range(6):
        (d / "runs" / f"r{i}.json").write_text(json.dumps({"n_test": 1600}))


def main():
    # ---- T1: external running eval process is detected and WAITED for (not duplicated) ----
    root = make_root()
    (root / "stub_eval_sleep.txt").write_text("8")
    ext = subprocess.Popen([sys.executable, "-u", "src/eval_gan_fid_kid.py"], cwd=root, stdout=open(root / "ext.out", "w"), stderr=subprocess.STDOUT)
    (root / "outputs/gan_comparison").mkdir(parents=True, exist_ok=True)
    time.sleep(1)
    fake_valid_downstream(root)
    r = ctl(root, "--status")
    check("T1 status detects the external eval process", f"{ext.pid}" in r.stdout, r.stdout.splitlines()[-1] if r.stdout else r.stderr)
    r = ctl(root, "--start")
    check("T1 start launches background daemon", "started in background" in r.stdout, r.stdout.strip())
    check("T1 second --start refused while running", "already running" in ctl(root, "--start").stdout)
    ok = wait_for(lambda: status_json(root)["stages"]["A"]["evidence"].get("waiting_on"), 20)
    check("T1 controller is WAITING on the external process (no duplicate launched)", ok and len(ctl(root, "--status").stdout) > 0)
    ext.wait()
    ok = wait_for(lambda: status_json(root)["stages"]["A"]["status"] == "done", 30)
    check("T1 stage A validated and completed after the process exited (not re-run)", ok)
    check("T1 controller never launched its own eval child for stage A (no duplicate)", "child_started" not in (root / "outputs/autopilot/history.jsonl").read_text().split("stage_done")[0])
    ok = wait_for(lambda: status_json(root)["status"] in ("finished", "paused", "blocked"), 60)
    st = status_json(root)
    check("T1 B, C(skipped: valid existing), D all completed -> finished", st["status"] == "finished" and all(s["status"] == "done" for s in st["stages"].values()),
          f"status={st['status']} " + str({k: v['status'] for k, v in st['stages'].items()}) + f" err={[v['error'] for v in st['stages'].values() if v['error']]}")
    check("T1 final artifacts written", all((root / "outputs" / f).exists() for f in ("final_report.md", "ia3_slides_outline.md", "ia3_viva_questions.md")) and
          (root / "outputs/gan_comparison/final_comparison.csv").exists() and (root / "outputs/gan_comparison/analysis.md").exists())
    check("T1 claude call count recorded", st["counters"]["claude_calls"] >= 4, str(st["counters"]))
    check("T1 decisions recorded", len(st["decisions"]) == 3, str([d["after"] for d in st["decisions"]]))
    # versioning: re-running a stage never overwrites silently
    (root / "outputs/gan_comparison/analysis.md").write_text("OLD")
    st["stages"]["B"]["status"] = "pending"; st["status"] = "idle"
    for s in "CD":
        st["stages"][s]["status"] = "done"
    (root / "outputs/autopilot/state.json").write_text(json.dumps(st))
    ctl(root, "--start", "--foreground")
    prev = list((root / "outputs/gan_comparison").glob("analysis.prev-*"))
    check("T1 re-created analysis.md keeps the old version (no silent overwrite)", len(prev) == 1 and prev[0].read_text() == "OLD")
    shutil.rmtree(root)

    # ---- T2: failing eval -> repair (stub fixes allowed file; out-of-scope edit reverted) -> succeed; limit respected ----
    root = make_root()
    (root / "stub_eval_fail.flag").write_text("x")
    fake_valid_downstream(root)
    r = ctl(root, "--start", "--foreground")
    st = status_json(root)
    hist = (root / "outputs/autopilot/history.jsonl").read_text()
    check("T2 failed eval triggered a bounded repair", st["stages"]["A"]["repairs"] == 1, f"repairs={st['stages']['A']['repairs']}")
    check("T2 out-of-scope edit was reverted", (root / "src/other_file.py").read_text() == "# original\n" and "out_of_scope" in hist and "src/other_file.py" in hist)
    check("T2 in-scope fix kept and diff recorded", "repaired by stub" in (root / "src/eval_gan_fid_kid.py").read_text() and list((root / "outputs/autopilot/repairs").glob("*.diff")))
    check("T2 workflow then completed", st["status"] == "finished", st["status"])
    shutil.rmtree(root)

    root = make_root()
    (root / "stub_eval_fail.flag").write_text("x")
    # stub 'repair' removes the flag; make it ineffective by making the failure permanent via a second flag the stub cannot clear
    (root / "src/eval_gan_fid_kid.py").write_text("import sys\nprint('Traceback (most recent call last):\\n  permanent failure')\nsys.exit(1)\n")
    fake_valid_downstream(root)
    ctl(root, "--start", "--foreground")
    st = status_json(root)
    check("T3 permanent failure stops at the repair limit (2) and BLOCKS", st["status"] == "blocked" and st["stages"]["A"]["status"] == "blocked" and st["stages"]["A"]["repairs"] == 2,
          f"status={st['status']} repairs={st['stages']['A']['repairs']} calls={st['counters']['claude_calls']}")
    check("T3 later stages untouched", all(st["stages"][s]["status"] == "pending" for s in "BCD"))
    r = ctl(root, "--unblock", "A")
    check("T3 --unblock restores the repair allowance", status_json(root)["stages"]["A"]["repairs"] == 0, r.stdout.strip())
    shutil.rmtree(root)

    # ---- T4: stop mid-run and resume without redoing finished work; claude 'stop' decision pauses ----
    root = make_root()
    (root / "stub_eval_sleep.txt").write_text("25")
    fake_valid_downstream(root)
    ctl(root, "--start")
    ok = wait_for(lambda: (status_json(root)["stages"]["A"].get("child_pid")), 20)
    child = status_json(root)["stages"]["A"]["child_pid"]
    check("T4 controller launched its own eval child", ok and child is not None, f"child pid {child}")
    r = ctl(root, "--stop")
    st = status_json(root)
    check("T4 --stop is graceful: state saved as stopped, stage interrupted", "stopped" in r.stdout and st["status"] == "stopped" and st["stages"]["A"]["status"] == "interrupted", r.stdout.strip() + str(st["status"]))
    try:
        os.kill(child, 0)
        child_alive = True
    except OSError:
        child_alive = False
    check("T4 only the controller-owned child was terminated (and it is gone)", not child_alive)
    (root / "stub_eval_sleep.txt").write_text("1")
    ctl(root, "--resume")
    ok = wait_for(lambda: status_json(root)["status"] in ("finished", "paused", "blocked"), 90)
    st = status_json(root)
    check("T4 resume completes the workflow", st["status"] == "finished", f"{st['status']} " + str({k: v['status'] for k, v in st['stages'].items()}))
    shutil.rmtree(root)

    root = make_root()
    fake_valid_downstream(root)
    (root / "stub_claude_stop.flag").write_text("x")
    (root / "outputs/gan_comparison").mkdir(parents=True)
    (root / "stub_eval_sleep.txt").write_text("0.5")
    ctl(root, "--start", "--foreground")
    st = status_json(root)
    check("T5 Claude 'stop' decision pauses (completed stage stays done)", st["status"] == "paused" and st["stages"]["A"]["status"] == "done" and st["stages"]["B"]["status"] == "pending", st["status"])
    shutil.rmtree(root)

    # ---- T6: claude call budget ----
    root = make_root()
    fake_valid_downstream(root)
    cfg = json.loads((root / "outputs/autopilot/config.json").read_text())
    cfg["max_claude_calls"] = 1
    (root / "outputs/autopilot/config.json").write_text(json.dumps(cfg))
    (root / "stub_eval_sleep.txt").write_text("0.5")
    ctl(root, "--start", "--foreground")
    st = status_json(root)
    check("T6 Claude call budget enforced (blocked, no extra calls)", st["status"] == "blocked" and st["counters"]["claude_calls"] <= 1, f"{st['status']} calls={st['counters']['claude_calls']}")
    shutil.rmtree(root)

    # ---- T8: HOLD file pauses before a stage; removing it and resuming completes ----
    root = make_root()
    fake_valid_downstream(root)
    (root / "stub_eval_sleep.txt").write_text("0.5")
    (root / "outputs/autopilot/HOLD_B").write_text("x")
    ctl(root, "--start", "--foreground")
    st = status_json(root)
    check("T8 HOLD_B pauses before stage B (A done, B untouched, no analysis call)", st["status"] == "paused" and st["stages"]["A"]["status"] == "done"
          and st["stages"]["B"]["status"] == "pending" and not (root / "outputs/gan_comparison/analysis.md").exists(), st["status"])
    (root / "outputs/autopilot/HOLD_B").unlink()
    ctl(root, "--resume", "--foreground")
    st = status_json(root)
    check("T8 after removing HOLD_B, --resume completes without redoing A", st["status"] == "finished" and st["stages"]["A"]["attempts"] == 1, st["status"])
    # ---- T9: --rerun resets only the chosen stages ----
    r = ctl(root, "--rerun", "C", "D")
    st = status_json(root)
    check("T9 --rerun C D resets only C and D", st["stages"]["C"]["status"] == "pending" and st["stages"]["D"]["status"] == "pending"
          and st["stages"]["A"]["status"] == "done" and st["stages"]["B"]["status"] == "done", r.stdout.strip())
    shutil.rmtree(root)

    # ---- T10: leakage-free requirement is enforced by the stage-C validator ----
    root = make_root()
    fake_valid_downstream(root)
    cfg = json.loads((root / "outputs/autopilot/config.json").read_text())
    cfg["require_leakage_free_downstream"] = True
    (root / "outputs/autopilot/config.json").write_text(json.dumps(cfg))
    sys.path.insert(0, str(HERE))
    import ia3_autopilot as ia
    a = ia.Autopilot(root)
    ok, problems, _ = a.validate_downstream()
    check("T10 summary without leakage_free=True is rejected when required", not ok and any("leakage-free" in p for p in problems), str(problems))
    sj = json.loads((root / "outputs/downstream_classification/summary.json").read_text())
    sj["config"]["leakage_free"] = True
    (root / "outputs/downstream_classification/summary.json").write_text(json.dumps(sj))
    ok, problems, _ = a.validate_downstream()
    check("T10 summary with leakage_free=True is accepted", ok, str(problems))
    shutil.rmtree(root)

    # ---- T7: real Claude CLI wiring (one tiny decision call) ----
    if REAL:
        root = make_root()
        cfg = json.loads((root / "outputs/autopilot/config.json").read_text())
        cfg["claude_bin"] = "claude"
        cfg["stage_timeouts_seconds"]["A"] = 60
        (root / "outputs/autopilot/config.json").write_text(json.dumps(cfg))
        sys.path.insert(0, str(HERE))
        import ia3_autopilot as ia
        a = ia.Autopilot(root)
        a.config["max_claude_calls"] = 14
        a.state["current_stage"] = "A"
        nxt = a.decide_next("A", {"evidence": {"fid": {"DCGAN": 133.8}, "note": "fixture"}})
        d = a.state["decisions"][-1]
        check("T7 real `claude -p` decision call works and is recorded", nxt == "B" and "unavailable" not in d["rationale"], str(d)[:200])
        shutil.rmtree(root)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
