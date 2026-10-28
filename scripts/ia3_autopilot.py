#!/usr/bin/env python3
"""Bounded, resumable local autopilot for the Brain Tumor MRI IA3 project (standard library only).

Interface (use scripts/run_ia3_autopilot.sh):
    --status | --plan | --dry-run | --start [--foreground] | --resume [--foreground] | --stop
    --unblock STAGE | --reset

Stages (fixed order): A FID/KID evaluation -> B results analysis -> C downstream classification -> D final artifacts.
It never deletes outputs (old versions are renamed), never uses --dangerously-skip-permissions, never starts GAN
training, and never kills processes it did not start.
"""
from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

STAGES = ["A", "B", "C", "D"]
STAGE_NAMES = {"A": "FID/KID evaluation", "B": "Results analysis", "C": "Downstream classification", "D": "Final research artifacts"}
GAN_RUNS = {"DCGAN": "dcgan_3000", "WGAN-GP": "wgangp_3000", "cWGAN-GP": "cwgangp_3000"}
EVAL_FILES = ["fid_kid_results.csv", "metrics.json", "comparison_config.json", "fid_kid_bar_chart.png",
              "sample_grids_real_vs_generated.png", "per_class_fid_kid_cwgangp.csv", "class_confusion_kid_cwgangp.csv"]
DEFAULT_CONFIG = {
    "poll_seconds": 15, "poll_max_seconds": 60,
    "max_claude_calls": 14, "claude_max_turns": 10, "claude_max_budget_usd_per_call": 1.5, "claude_timeout_seconds": 1200,
    "claude_bin": "claude", "claude_model": None, "decision_with_claude": True,
    "max_repairs_per_stage": 2,
    "stage_timeouts_seconds": {"A": 7200, "B": 1800, "C": 5400, "D": 3600},
    "max_total_training_seconds": 5400,
    "python_bin": None, "min_free_disk_gb": 5, "verify_checkpoint_counters": True,
    "downstream_seeds": [42, 43, 44], "downstream_epochs": 15, "downstream_synthetic_per_class": 1400,
    "downstream_dir": "outputs/downstream_classification", "downstream_preliminary_dir": None,
    "downstream_gan_ckpt": None, "require_leakage_free_downstream": False,
    "downstream_matched_dir": None, "report_notes": [],
}
REFERENCES = [  # supplied so the report does not invent citations; user must verify before submission
    "Mirza & Osindero, Conditional Generative Adversarial Nets, arXiv:1411.1784",
    "Radford, Metz & Chintala, Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN), arXiv:1511.06434",
    "Gulrajani et al., Improved Training of Wasserstein GANs (WGAN-GP), arXiv:1704.00028",
    "Heusel et al., GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID), arXiv:1706.08500",
    "Binkowski et al., Demystifying MMD GANs (KID), arXiv:1801.01401",
    "Miyato & Koyama, cGANs with Projection Discriminator, arXiv:1802.05637",
    "Kaggle 'Brain Tumor MRI Dataset' (masoudnickparvar/brain-tumor-mri-dataset)",
]


class Blocked(Exception):
    """A condition needing human attention (budget, unresolved failure, unsafe proposal)."""


class Stopped(Exception):
    pass


class Paused(Exception):
    """Claude's post-stage decision asked for human review; the completed stage stays 'done'."""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Autopilot:
    def __init__(self, root: Path, state_dir: Path | None = None, config_path: Path | None = None):
        self.root = root.resolve()
        self.dir = (state_dir or self.root / "outputs" / "autopilot").resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path, self.log_path = self.dir / "state.json", self.dir / "autopilot.log"
        self.hist_path, self.lock_path = self.dir / "history.jsonl", self.dir / "autopilot.lock"
        self.pid_path = self.dir / "autopilot.pid"
        self.config_path = config_path or self.dir / "config.json"
        self.config = dict(DEFAULT_CONFIG)
        if self.config_path.exists():
            self.config.update(json.loads(self.config_path.read_text()))
        else:
            self.config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        if not self.config["python_bin"]:
            venv = self.root / ".venv" / "bin" / "python"
            self.config["python_bin"] = str(venv if venv.exists() else sys.executable)
        self.stop_requested = False
        self.child: subprocess.Popen | None = None
        self.state = self.load_state()

    # ------------------------------------------------------------------ state / logging
    def fresh_state(self) -> dict:
        return {"version": 1, "created": now(), "status": "idle", "current_stage": None, "controller_pid": None,
                "stages": {s: {"name": STAGE_NAMES[s], "status": "pending", "attempts": 0, "repairs": 0, "started": None,
                                "ended": None, "evidence": {}, "artifacts": [], "error": None, "child_pid": None} for s in STAGES},
                "counters": {"claude_calls": 0, "claude_cost_usd": 0.0, "training_seconds": 0.0}, "decisions": []}

    def load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return self.fresh_state()

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        os.replace(tmp, self.state_path)

    def log(self, msg: str, level: str = "INFO") -> None:
        line = f"{now()} {level} {msg}"
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        if sys.stdout.isatty():
            print(line, flush=True)

    def event(self, kind: str, **kw) -> None:
        rec = {"time": now(), "event": kind, **kw}
        with open(self.hist_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.log(f"{kind} {json.dumps(kw, default=str)[:300]}")

    def stage(self, s: str) -> dict:
        return self.state["stages"][s]

    # ------------------------------------------------------------------ safe file helpers
    def write_versioned(self, path: Path, text: str) -> None:
        """Write text; an existing file is renamed (never deleted or silently overwritten)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.rename(path.with_name(f"{path.stem}.prev-{stamp()}{path.suffix}"))
        path.write_text(text)

    def archive_dir(self, d: Path) -> None:
        if d.exists():
            d.rename(d.with_name(f"{d.name}.prev-{stamp()}"))

    # ------------------------------------------------------------------ processes
    @staticmethod
    def find_processes(script_basename: str, exclude_pids: set[int] | None = None) -> list[dict]:
        """Python processes whose ARGV contains the script as an argument (not substring of a wrapper command line)."""
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,etime=,command="], capture_output=True, text=True).stdout
        found = []
        for line in out.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            pid, ppid, etime, cmd = int(parts[0]), int(parts[1]), parts[2], parts[3]
            if pid == os.getpid() or pid in (exclude_pids or set()):
                continue
            try:
                argv = shlex.split(cmd)
            except ValueError:
                argv = cmd.split()
            if argv and os.path.basename(argv[0]).lower().startswith("python") and \
                    any(os.path.basename(a) == script_basename for a in argv[1:]):
                found.append({"pid": pid, "ppid": ppid, "etime": etime, "argv": argv[:6]})
        return found

    def sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if self.stop_requested:
                raise Stopped()
            time.sleep(min(1.0, max(0.0, end - time.time())))

    def poll_interval(self, n: int) -> float:
        return min(self.config["poll_seconds"] * (1 + n // 8), self.config["poll_max_seconds"])

    def run_child(self, s: str, cmd: list[str], log_file: Path, timeout: float, count_training: bool = False) -> int:
        """Run a controller-owned subprocess in its own session; returns the exit code. Terminated only if WE started it."""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(log_file, "a") as lf:
            self.child = subprocess.Popen(cmd, cwd=self.root, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            self.stage(s)["child_pid"] = self.child.pid
            self.save()
            self.event("child_started", stage=s, pid=self.child.pid, cmd=" ".join(cmd)[:200], log=str(log_file))
            n = 0
            while self.child.poll() is None:
                try:
                    self.sleep(self.poll_interval(n))
                except Stopped:
                    self.terminate_child()
                    raise
                n += 1
                if time.time() - t0 > timeout:
                    self.terminate_child()
                    raise Blocked(f"stage {s} subprocess exceeded timeout of {timeout:.0f}s")
            rc = self.child.returncode
            lf.write(f"\nexit {rc}\n")
        if count_training:
            self.state["counters"]["training_seconds"] += time.time() - t0
        self.stage(s)["child_pid"] = None
        self.child = None
        self.save()
        self.event("child_finished", stage=s, exit_code=rc, seconds=round(time.time() - t0, 1))
        return rc

    def terminate_child(self) -> None:
        if self.child and self.child.poll() is None:
            self.log(f"terminating controller-owned child pid {self.child.pid}")
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
                self.child.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(self.child.pid, signal.SIGKILL)
                except Exception:
                    pass

    # ------------------------------------------------------------------ Claude Code CLI
    def claude(self, purpose: str, prompt: str, tools: list[str], schema: dict | None = None, allow_edit: bool = False) -> dict:
        c = self.state["counters"]
        if c["claude_calls"] >= self.config["max_claude_calls"]:
            raise Blocked(f"Claude call budget exhausted ({self.config['max_claude_calls']})")
        cmd = [self.config["claude_bin"], "-p", "--output-format", "json", "--no-session-persistence",
               "--max-turns", str(self.config["claude_max_turns"]),
               "--max-budget-usd", str(self.config["claude_max_budget_usd_per_call"]),
               "--tools", ",".join(tools)]
        if allow_edit:
            cmd += ["--allowedTools", "Edit"]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        if self.config["claude_model"]:
            cmd += ["--model", self.config["claude_model"]]
        c["claude_calls"] += 1
        self.save()
        t0 = time.time()
        try:
            r = subprocess.run(cmd, input=prompt, cwd=self.root, capture_output=True, text=True,
                               timeout=self.config["claude_timeout_seconds"])
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"claude call '{purpose}' timed out")
        calls_dir = self.dir / "claude_calls"
        calls_dir.mkdir(exist_ok=True)
        (calls_dir / f"{c['claude_calls']:03d}_{purpose}.json").write_text(
            json.dumps({"cmd": cmd, "prompt_head": prompt[:1500], "returncode": r.returncode, "stdout": r.stdout[-200000:], "stderr": r.stderr[-4000:]}, indent=2))
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude call '{purpose}' returned non-JSON output (exit {r.returncode}): {r.stderr[-300:]}")
        c["claude_cost_usd"] = round(c["claude_cost_usd"] + float(out.get("total_cost_usd") or 0), 4)
        self.save()
        self.event("claude_call", purpose=purpose, seconds=round(time.time() - t0, 1), turns=out.get("num_turns"),
                   cost_usd=out.get("total_cost_usd"), is_error=out.get("is_error"), denials=len(out.get("permission_denials") or []))
        if r.returncode != 0 or out.get("is_error") or out.get("subtype") not in (None, "success"):
            raise RuntimeError(f"claude call '{purpose}' failed: subtype={out.get('subtype')} result={str(out.get('result'))[:300]}")
        return out

    # ------------------------------------------------------------------ Stage A: FID/KID evaluation
    def eval_dir(self) -> Path:
        return self.root / "outputs" / "gan_comparison"

    def training_summary(self) -> dict:
        info = {}
        for model, d in GAN_RUNS.items():
            p = self.root / "outputs" / d / "metrics.json"
            if not p.exists():
                info[model] = {"error": f"{p} missing"}
                continue
            m = json.loads(p.read_text())
            info[model] = {"status": m.get("status"), "device": m.get("device"),
                           "generator_updates": m.get("generator_updates_completed", m.get("batches_completed")),
                           "critic_or_discriminator_updates": m.get("critic_updates_completed", m.get("batches_completed")),
                           "runtime_seconds": m.get("runtime_seconds"), "finite": m.get("all_losses_and_grads_finite"),
                           "trained_on_testing_dir": m.get("trained_on_testing_dir"), "images_used": m.get("images_used")}
        return info

    def validate_training_runs(self) -> list[str]:
        problems = []
        for model, i in self.training_summary().items():
            if "error" in i:
                problems.append(f"{model}: {i['error']}")
                continue
            if i["status"] != "ok" or i["generator_updates"] != 3000:
                problems.append(f"{model}: status={i['status']} generator_updates={i['generator_updates']} (expected ok / 3000)")
            if i["trained_on_testing_dir"] is not False:
                problems.append(f"{model}: trained_on_testing_dir is not False")
            if i["finite"] is not True:
                problems.append(f"{model}: losses/gradients not confirmed finite")
            ck = self.root / "outputs" / GAN_RUNS[model] / "checkpoint_step_3000.pth"
            if not ck.exists() or ck.stat().st_size == 0:
                problems.append(f"{model}: checkpoint_step_3000.pth missing/empty")
            elif self.config["verify_checkpoint_counters"]:
                code = ("import torch,sys;c=torch.load(sys.argv[1],map_location='cpu',weights_only=True);"
                        "print(c.get('gen_updates',c.get('batches')))")
                r = subprocess.run([self.config["python_bin"], "-I", "-c", code, str(ck)], capture_output=True, text=True)
                if r.stdout.strip() != "3000":
                    problems.append(f"{model}: embedded update counter {r.stdout.strip() or r.stderr[-120:]} != 3000")
        return problems

    def validate_eval(self) -> tuple[bool, list[str], dict]:
        d, problems, ev = self.eval_dir(), [], {}
        for f in EVAL_FILES:
            if not (d / f).exists() or (d / f).stat().st_size == 0:
                problems.append(f"missing/empty {f}")
        log = d / "eval.log"
        tail = log.read_text()[-400:] if log.exists() else ""
        if "Traceback" in tail or any(f"exit {n}" in tail[-60:] for n in range(1, 256)):
            problems.append("eval.log shows a failure (traceback / non-zero exit)")
        if problems:
            return False, problems, ev
        try:
            m, cfg = json.loads((d / "metrics.json").read_text()), json.loads((d / "comparison_config.json").read_text())
            agg = m["aggregate"]
            for model in GAN_RUNS:
                a = agg[model]
                if not all(is_finite_number(a[k]) for k in ("fid_mean", "fid_std_across_seeds", "kid_mean", "kid_std_across_seeds")):
                    problems.append(f"{model}: non-finite or missing FID/KID values")
                if a["n_seeds"] < 3 or a["n_real"] != 1600 or a["n_generated_per_seed"] != 1600:
                    problems.append(f"{model}: unexpected sample counts {a['n_seeds']}/{a['n_generated_per_seed']}/{a['n_real']}")
            if "Testing" not in cfg.get("real_reference", "") or not str(cfg.get("image_preprocessing", "")).strip() \
                    or not cfg.get("generated_samples") or not cfg.get("generation_seeds"):
                problems.append("comparison_config.json does not document reference set / preprocessing / sample counts / seeds")
            for model, c in cfg["checkpoints"].items():
                if c["generator_updates"] != 3000:
                    problems.append(f"{model}: evaluated checkpoint has {c['generator_updates']} updates")
            ev = {"fid": {k: round(v["fid_mean"], 2) for k, v in agg.items()}, "kid_x1000": {k: round(v["kid_mean"] * 1000, 2) for k, v in agg.items()},
                  "feature_extractor": cfg.get("feature_extractor"), "implementation": cfg.get("implementation"),
                  "exit_status_line": "exit 0" in tail}
        except Exception as e:
            problems.append(f"could not parse evaluation outputs: {type(e).__name__}: {e}")
        problems += self.validate_training_runs()
        return not problems, problems, ev

    def eval_state(self) -> tuple[str, dict]:
        procs = self.find_processes("eval_gan_fid_kid.py")
        if procs:
            return "running", {"processes": procs}
        ok, problems, ev = self.validate_eval()
        if ok:
            return "valid", ev
        log = self.eval_dir() / "eval.log"
        text = log.read_text() if log.exists() else ""
        if "Traceback" in text or any(f"exit {n}" in text[-200:] for n in range(1, 256)):
            return "failed", {"problems": problems, "log_tail": text[-1500:]}
        if not log.exists():
            return "not_started", {"problems": problems}
        return "ambiguous", {"problems": problems, "log_tail": text[-600:]}

    def stage_A(self) -> dict:
        st, waits = self.stage("A"), 0
        while True:
            state, info = self.eval_state()
            self.event("eval_state", state=state, detail=str(info)[:300])
            if state == "valid":
                return {"ok": True, "evidence": info, "artifacts": [str(self.eval_dir() / f) for f in EVAL_FILES],
                        "note": "evaluation already complete; artifacts validated, not re-run"}
            if state == "running":
                self.log(f"waiting for running eval process(es): {[p['pid'] for p in info['processes']]}")
                st["evidence"] = {"waiting_on": info["processes"]}
                self.save()
                waited = 0.0
                while self.find_processes("eval_gan_fid_kid.py"):
                    self.sleep(self.poll_interval(waits))
                    waits += 1
                    waited += self.poll_interval(waits)
                    if waited > self.config["stage_timeouts_seconds"]["A"]:
                        raise Blocked("timed out waiting for the running evaluation process (not killed: not owned by autopilot)")
                continue
            if state == "not_started":
                problems = self.validate_training_runs()
                if problems:
                    raise Blocked("cannot start evaluation, training prerequisites failed: " + "; ".join(problems))
                self.launch_eval()
                continue
            if state == "ambiguous":
                log_text = (self.eval_dir() / "eval.log").read_text() if (self.eval_dir() / "eval.log").exists() else ""
                if "exit 0" in log_text[-400:]:
                    raise Blocked(f"evaluation exited 0 but its outputs failed validation (human review needed): {info.get('problems')}")
                if st["repairs"] >= self.config["max_repairs_per_stage"]:
                    raise Blocked(f"evaluation has no exit status and no valid outputs; rerun limit reached: {info.get('problems')}")
                st["repairs"] += 1
                self.save()
                self.event("eval_rerun_interrupted", problems=info.get("problems"))
                (self.eval_dir() / "eval.log").rename(self.eval_dir() / f"eval.log.prev-{stamp()}")
                self.launch_eval()
                continue
            # failed
            if st["repairs"] >= self.config["max_repairs_per_stage"]:
                raise Blocked(f"evaluation {state}; repair limit reached. Problems: {info.get('problems')}")
            st["repairs"] += 1
            self.save()
            self.repair("A", ["src/eval_gan_fid_kid.py"], f"The FID/KID evaluation is in state '{state}'.\nProblems: {info.get('problems')}\n"
                        f"Tail of outputs/gan_comparison/eval.log:\n{info.get('log_tail', '')}")
            if self.eval_dir().joinpath("eval.log").exists():
                self.eval_dir().joinpath("eval.log").rename(self.eval_dir() / f"eval.log.prev-{stamp()}")
            self.launch_eval()

    def launch_eval(self) -> None:
        self.eval_dir().mkdir(parents=True, exist_ok=True)
        rc = self.run_child("A", [self.config["python_bin"], "-u", "src/eval_gan_fid_kid.py"], self.eval_dir() / "eval.log",
                            self.config["stage_timeouts_seconds"]["A"])
        self.event("eval_exit", exit_code=rc)

    # ------------------------------------------------------------------ repair (bounded, reverted if it strays)
    def snapshot_tree(self) -> dict[str, str]:
        files = {}
        for d in ("src", "scripts"):
            for p in (self.root / d).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    files[str(p.relative_to(self.root))] = sha256_file(p)
        return files

    def repair(self, s: str, allowed: list[str], problem: str) -> None:
        before, bdir = self.snapshot_tree(), self.dir / "backups" / stamp()
        for rel in before:
            (bdir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / rel, bdir / rel)
        prompt = (f"REPAIR TASK for stage {s} ({STAGE_NAMES[s]}) of the Brain Tumor MRI IA3 project in the current directory.\n"
                  f"{problem}\n\nDiagnose the real cause using Read/Grep, then make the smallest possible fix by editing ONLY these files: "
                  f"{allowed}. Do NOT change GAN architectures, hyperparameters, checkpoints, datasets or metric definitions, and do not "
                  "touch any other file. If the failure is environmental (missing package, no disk, no checkpoint) make NO edit and say so. "
                  "Reply with a one-paragraph description of the cause and of exactly what you changed.")
        out = self.claude(f"repair_{s}", prompt, ["Read", "Glob", "Grep", "Edit"], allow_edit=True)
        after, reverted, changed = self.snapshot_tree(), [], []
        for rel in sorted(set(before) | set(after)):
            if before.get(rel) == after.get(rel):
                continue
            if rel in allowed:
                changed.append(rel)
                old = (bdir / rel).read_text().splitlines(keepends=True) if rel in before else []
                new = (self.root / rel).read_text().splitlines(keepends=True) if rel in after else []
                (self.dir / "repairs").mkdir(exist_ok=True)
                (self.dir / "repairs" / f"{stamp()}_{rel.replace('/', '_')}.diff").write_text("".join(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}")))
            else:
                reverted.append(rel)
                if rel in before:
                    shutil.copy2(bdir / rel, self.root / rel)
                else:
                    (self.root / rel).rename(self.root / f"{rel}.rejected-{stamp()}")
        self.event("repair_done", stage=s, changed=changed, reverted_out_of_scope=reverted, claude_summary=str(out.get("result"))[:400])

    # ------------------------------------------------------------------ Stage B: analysis
    def facts_pack(self) -> dict:
        d = self.eval_dir()
        m = json.loads((d / "metrics.json").read_text())
        cfg = json.loads((d / "comparison_config.json").read_text())
        agg = {k: v for k, v in m["aggregate"].items() if k in GAN_RUNS}
        by_fid = sorted(agg, key=lambda k: agg[k]["fid_mean"])
        by_kid = sorted(agg, key=lambda k: agg[k]["kid_mean"])
        return {"gan_aggregate": m["aggregate"], "per_class_cwgangp": m["per_class_cwgangp"],
                "class_confusion_kid_cwgangp": m["class_confusion_kid_cwgangp"],
                "ranking_by_fid_best_first": by_fid, "ranking_by_kid_best_first": by_kid,
                "best_fid_model": by_fid[0], "best_kid_model": by_kid[0],
                "evaluation_config": {k: cfg.get(k) for k in ("implementation", "feature_extractor", "feature_layer", "image_preprocessing",
                                                              "real_reference", "generated_samples", "generation_seeds", "kid", "uncertainty")},
                "training_runs": self.training_summary(),
                "image_files_to_view": [str(d / "sample_grids_real_vs_generated.png"), str(d / "fid_kid_bar_chart.png")] +
                                       [str(self.root / "outputs" / r / "samples_step_3000.png") for r in GAN_RUNS.values()]}

    def stage_B(self) -> dict:
        st = self.stage("B")
        facts = self.facts_pack()
        prompt = (
            "ANALYSIS TASK. You are analysing completed experiments for the IA3 project 'Brain Tumor MRI Image Generation and Classification "
            "Using a Hybrid Conditional WGAN-GP'. Use ONLY the numbers in the FACTS JSON below (do not compute new metrics, do not invent any). "
            "Also use Read to LOOK at every file listed in image_files_to_view and describe what the images actually show.\n"
            "Write GitHub-flavoured Markdown with exactly these '##' sections: 'Summary', 'Quantitative results', 'What the image grids show', "
            "'Class conditioning', 'Limitations', 'What the results support and do not support'. Requirements: state which model has the best "
            "FID and best KID (best_fid_model / best_kid_model) even if it is not the proposed hybrid; say whether the gaps exceed the "
            "across-seed std (note that only ONE training run per model exists, so training-seed variability is unmeasured); discuss the "
            "real-vs-real floor; interpret class_confusion_kid_cwgangp honestly (does each generated label sit closest to its own real class?); "
            "state limitations: ImageNet-Inception features on 64x64 grayscale MRI upsampled to 299, FID bias at n=1600, per-class n=400 is small, "
            "single training seed, metrics do not prove anatomically correct tumours, the pretrained classifier may overlap its original "
            "training data. Make no clinical claims. Return ONLY the markdown document as your final answer.\n\nFACTS:\n"
            + json.dumps(facts, indent=1))
        out = self.claude("analysis", prompt, ["Read", "Glob", "Grep"])
        text = str(out.get("result") or "").strip()
        need = ["summary", "quantitative", "image grids", "class conditioning", "limitations", "support"]
        low = text.lower()
        missing = [n for n in need if n not in low]
        for k in ("best_fid_model", "best_kid_model"):
            if facts[k].lower() not in low:
                missing.append(k)
        if len(text) < 1500 or missing:
            raise RuntimeError(f"analysis output failed validation (len={len(text)}, missing={missing})")
        path = self.eval_dir() / "analysis.md"
        self.write_versioned(path, text + "\n")
        return {"ok": True, "artifacts": [str(path)], "evidence": {"best_fid": facts["best_fid_model"], "best_kid": facts["best_kid_model"], "chars": len(text)}}

    # ------------------------------------------------------------------ Stage C: downstream classification
    def ds_dir(self) -> Path:
        return self.root / self.config["downstream_dir"]

    def validate_downstream(self) -> tuple[bool, list[str], dict]:
        d, problems, ev = self.ds_dir(), [], {}
        p = d / "summary.json"
        if not p.exists():
            return False, ["summary.json missing"], ev
        try:
            s = json.loads(p.read_text())
            if s["config"].get("smoke"):
                problems.append("summary is from a --smoke run")
            for cond in ("real_only", "real_plus_cwgangp_synthetic"):
                c = s["conditions"][cond]
                if c["n_runs"] < 3:
                    problems.append(f"{cond}: only {c['n_runs']} runs")
                if not all(is_finite_number(c["accuracy"]["mean"]) and is_finite_number(c["macro_f1"]["mean"]) for _ in [0]):
                    problems.append(f"{cond}: non-finite metrics")
            if self.config["require_leakage_free_downstream"] and s["config"].get("leakage_free") is not True:
                problems.append("downstream experiment is not leakage-free (GAN training data overlaps the CNN validation set)")
            runs = list((d / "runs").glob("*.json"))
            if runs and any(json.loads(r.read_text())["n_test"] != 1600 for r in runs):
                problems.append("a run did not evaluate on all 1600 test images")
            if len(runs) < 6:
                problems.append(f"expected >=6 run files, found {len(runs)}")
            ev = {k: {"acc_mean": round(v["accuracy"]["mean"], 4), "acc_std": v["accuracy"]["std"], "macro_f1_mean": round(v["macro_f1"]["mean"], 4), "n_train": v["n_train"]}
                  for k, v in s["conditions"].items()}
            ev["paired_accuracy_difference_synthetic_minus_real_only"] = s["paired_accuracy_difference_synthetic_minus_real_only"]
            ev["leakage_audit"] = s["config"].get("leakage_audit")
        except Exception as e:
            problems.append(f"could not parse summary.json: {type(e).__name__}: {e}")
        return not problems, problems, ev

    def preflight_C(self) -> list[str]:
        problems = []
        free = shutil.disk_usage(self.root).free / 1e9
        if free < self.config["min_free_disk_gb"]:
            problems.append(f"only {free:.1f} GB free (< {self.config['min_free_disk_gb']})")
        r = subprocess.run([self.config["python_bin"], "-I", "-c", "import torch,sklearn,matplotlib,PIL;print(torch.backends.mps.is_available())"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(f"missing Python dependency: {r.stderr.strip()[-200:]}")
        elif r.stdout.strip() != "True":
            self.log("MPS unavailable; the downstream experiment will run on CPU (slower)", "WARN")
        gan_ck = Path(self.config["downstream_gan_ckpt"] or "outputs/cwgangp_3000/checkpoint_step_3000.pth")
        if not (gan_ck if gan_ck.is_absolute() else self.root / gan_ck).exists():
            problems.append(f"cWGAN-GP checkpoint missing: {gan_ck}")
        remaining = self.config["max_total_training_seconds"] - self.state["counters"]["training_seconds"]
        if remaining <= 0:
            problems.append("max_total_training_seconds budget exhausted")
        return problems

    def stage_C(self) -> dict:
        st = self.stage("C")
        while True:
            ok, problems, ev = self.validate_downstream()
            if ok:
                return {"ok": True, "evidence": ev, "artifacts": [str(self.ds_dir() / "summary.json")], "note": "valid downstream experiment already exists; not re-run"}
            pre = self.preflight_C()
            if pre:
                raise Blocked("downstream preflight failed: " + "; ".join(pre))
            if (self.ds_dir() / "summary.json").exists() or (self.ds_dir() / "runs").exists():
                self.archive_dir(self.ds_dir())
            cmd = [self.config["python_bin"], "-u", "src/downstream_classification.py", "--out-dir", str(self.ds_dir()),
                   "--seeds", *map(str, self.config["downstream_seeds"]), "--epochs", str(self.config["downstream_epochs"]),
                   "--synthetic-per-class", str(self.config["downstream_synthetic_per_class"])]
            if self.config["downstream_gan_ckpt"]:
                cmd += ["--gan-ckpt", str(self.config["downstream_gan_ckpt"])]
            if self.config["require_leakage_free_downstream"]:
                cmd += ["--require-leakage-free"]
            run_log = self.root / "outputs" / f"{self.ds_dir().name}_run.log"
            remaining = self.config["max_total_training_seconds"] - self.state["counters"]["training_seconds"]
            rc = self.run_child("C", cmd, run_log,
                                min(self.config["stage_timeouts_seconds"]["C"], remaining), count_training=True)
            ok, problems, ev = self.validate_downstream()
            if rc == 0 and ok:
                return {"ok": True, "evidence": ev, "artifacts": [str(self.ds_dir() / "summary.json"), str(self.ds_dir() / "classification_report.txt")]}
            if st["repairs"] >= self.config["max_repairs_per_stage"]:
                raise Blocked(f"downstream experiment failed (exit {rc}); repair limit reached. {problems}")
            st["repairs"] += 1
            self.save()
            tail = run_log.read_text()[-1500:]
            self.repair("C", ["src/downstream_classification.py"], f"The downstream classification run exited with {rc}. Validation: {problems}\nLog tail:\n{tail}")

    # ------------------------------------------------------------------ Stage D: final artifacts
    def stage_D(self) -> dict:
        d, out = self.eval_dir(), self.root / "outputs"
        fa = [self.config["python_bin"], "scripts/ia3_final_artifacts.py", "--root", str(self.root), "--downstream-dir", str(self.config["downstream_dir"])]
        if self.config["downstream_matched_dir"]:
            fa += ["--matched-dir", str(self.config["downstream_matched_dir"])]
        if self.config["downstream_preliminary_dir"]:
            fa += ["--preliminary-dir", str(self.config["downstream_preliminary_dir"])]
        r = subprocess.run(fa, cwd=self.root, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"final table/figure generation failed: {r.stderr[-500:]}")
        self.event("final_tables_built", out=r.stdout[-300:])
        facts = self.facts_pack()
        facts["downstream"] = json.loads((self.ds_dir() / "summary.json").read_text())
        facts["downstream_config_note"] = "synthetic labels are the generation labels; whether the images depict the class is unverified"
        facts["downstream_primary_leakage_audit"] = facts["downstream"]["config"].get("leakage_audit")
        facts["operational_notes"] = self.config["report_notes"]
        md = self.config["downstream_matched_dir"]
        if md and (self.root / md / "summary.json").exists():
            facts["downstream_matched_control"] = json.loads((self.root / md / "summary.json").read_text())
            facts["downstream_matched_control_interpretation"] = (self.root / md / "interpretation.md").read_text()
            facts["downstream_matched_control_note"] = (
                "Condition C = real-only resampled with replacement to the size of B (10,080; 2,520/class), no synthetic images. It controls for "
                "training-set size and optimizer steps, not for diversity. A = 4,480 real; B = A + 5,600 synthetic from the CLEAN cWGAN-GP.")
        prelim = self.config["downstream_preliminary_dir"]
        if prelim and (self.root / prelim / "summary.json").exists():
            facts["downstream_preliminary_LEAKY"] = json.loads((self.root / prelim / "summary.json").read_text())
            facts["downstream_preliminary_note"] = (
                "PRELIMINARY and NOT leakage-free: its synthetic images came from a cWGAN-GP trained on all 5,600 Training images, which "
                "included the 1,120 CNN validation images that were used for epoch selection. Archive/Testing was never seen by the GAN.")
        facts["analysis_markdown_path"] = str(d / "analysis.md")
        facts["approved_references"] = REFERENCES
        facts["architectures"] = {
            "DCGAN": "conv/transposed-conv generator (z=100 -> 64x64x1, BatchNorm, Tanh) and BatchNorm discriminator, non-saturating BCE, Adam 2e-4 (0.5,0.999)",
            "WGAN-GP": "same generator, BatchNorm-free critic, Wasserstein loss + gradient penalty (lambda=10), 5 critic steps per generator step, Adam 1e-4 (0.5,0.9)",
            "cWGAN-GP (proposed)": "WGAN-GP with a 50-d label embedding concatenated to z in the generator and a projection critic D(x,y)=psi(h(x))+<E(y),h(x)>; no classifier guidance",
            "downstream CNN": "4 conv blocks (32-64-128-256) + GAP + dropout 0.3, Adam 1e-3, 15 epochs, no augmentation"}
        facts["dataset"] = {"train": "archive/Training 5,600 images (1,400/class), used for GAN training and the stratified 80/20 train/val split",
                            "test": "archive/Testing 1,600 images (400/class), used only for FID/KID reference and final classifier evaluation",
                            "pretrained_classifier_note": "an independent pretrained EfficientNet-B3 scored 95.94% on archive/Testing but may overlap its original training data; not an independent benchmark"}
        prompt = (
            "FINAL DELIVERABLES TASK for the IA3 project 'Brain Tumor MRI Image Generation and Classification Using a Hybrid Conditional WGAN-GP'. "
            "Using ONLY the FACTS JSON (and Read of analysis_markdown_path if useful) produce three Markdown documents. Never invent numbers, "
            "results, datasets or citations; cite only approved_references (by author/title/arXiv id exactly as given) and tell the reader to verify "
            "them. State results honestly even when the proposed model is not best or synthetic augmentation does not help. If FACTS contains "
            "downstream_preliminary_LEAKY, the report MUST (a) explain the leakage plainly, (b) label that experiment preliminary and never "
            "base a conclusion on it, and (c) use `downstream` (the clean, leakage-free run; check downstream_primary_leakage_audit) as the "
            "primary downstream result; compare the two only to show how much the leakage mattered. If FACTS contains downstream_matched_control, "
            "include every item of FACTS.operational_notes (in substance, accurately and without strengthening or weakening them) in the Limitations / "
            "reproducibility discussion; keep validation metrics distinct from test metrics. Also report Condition C (matched-size real-only control) next to A and B and follow its saved hedged interpretation: do not credit the "
            "synthetic images with a benefit unless B exceeds C by more than the seed variability, and say that duplicated real images are not new data. No clinical claims; "
            "generated images are not shown to contain anatomically accurate tumours.\n"
            "1. final_report: sections Title, Abstract, Problem statement, Dataset and preprocessing, Methodology (models, objectives, architectures), "
            "Experimental setup, Results (GAN FID/KID table; per-class and class-confusion findings; downstream classification table with real-only vs "
            "real+synthetic), Discussion, Limitations, Conclusion, References.\n"
            "2. slides_outline: a numbered slide-by-slide outline (about 12-15 slides) with bullet points and which figure/table from outputs/ goes on each.\n"
            "3. viva_questions: at least 20 likely viva questions each with a concise, factually correct answer grounded in the facts.\n\nFACTS:\n"
            + json.dumps(facts, indent=1))
        schema = {"type": "object", "required": ["final_report", "slides_outline", "viva_questions"],
                  "properties": {k: {"type": "string"} for k in ("final_report", "slides_outline", "viva_questions")}}
        res = self.claude("final_artifacts", prompt, ["Read", "Glob", "Grep"], schema=schema)
        so = res.get("structured_output") or {}
        checks = {"final_report": ["abstract", "limitations", "conclusion", "results"], "slides_outline": ["slide"], "viva_questions": ["?"]}
        for k, needles in checks.items():
            t = str(so.get(k, ""))
            if len(t) < 800 or any(n not in t.lower() for n in needles):
                raise RuntimeError(f"{k} failed validation (len={len(t)})")
        paths = {"final_report": out / "final_report.md", "slides_outline": out / "ia3_slides_outline.md", "viva_questions": out / "ia3_viva_questions.md"}
        for k, p in paths.items():
            self.write_versioned(p, so[k].strip() + "\n")
        arts = [str(p) for p in paths.values()] + [str(d / "final_comparison.csv"), str(d / "final_comparison.png")]
        missing = [a for a in arts if not Path(a).exists()]
        if missing:
            raise RuntimeError(f"missing final artifacts: {missing}")
        return {"ok": True, "artifacts": arts, "evidence": {"tables": r.stdout[-200:]}}

    # ------------------------------------------------------------------ decision loop
    def decide_next(self, done: str, result: dict) -> str:
        i = STAGES.index(done)
        if i == len(STAGES) - 1:
            return "finish"
        nxt = STAGES[i + 1]
        decision, rationale = "continue", "default plan: next stage is pre-authorised"
        if self.config["decision_with_claude"] and self.state["counters"]["claude_calls"] < self.config["max_claude_calls"] - 4:
            schema = {"type": "object", "required": ["decision", "rationale"],
                      "properties": {"decision": {"type": "string", "enum": ["continue", "stop"]}, "rationale": {"type": "string"},
                                     "concerns": {"type": "array", "items": {"type": "string"}}}}
            prompt = (f"DECISION. Workflow stages in fixed order: {STAGES} ({STAGE_NAMES}). Stage {done} just completed with this evidence:\n"
                      f"{json.dumps(result.get('evidence'), default=str)[:3000]}\nThe only permitted next stage is {nxt} ({STAGE_NAMES[nxt]}); "
                      "no new GAN training, no architecture changes. Answer 'stop' ONLY if the evidence shows a serious data-integrity problem "
                      "(e.g. test-set leakage, non-finite or implausible metrics) that a human must review; otherwise 'continue'.")
            try:
                out = self.claude(f"decide_after_{done}", prompt, [], schema=schema)
                so = out.get("structured_output") or {}
                decision, rationale = so.get("decision", "continue"), f"{so.get('rationale', '')} concerns={so.get('concerns')}"
            except (RuntimeError, Blocked) as e:
                rationale = f"claude decision unavailable ({e}); falling back to the authorised default"
        self.state["decisions"].append({"time": now(), "after": done, "next": nxt, "claude_decision": decision, "rationale": rationale})
        self.event("decision", after=done, next=nxt, decision=decision, rationale=rationale[:300])
        if decision == "stop":
            raise Paused(f"Claude recommended pausing for human review after stage {done}: {rationale}")
        return nxt

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        self.state["controller_pid"] = os.getpid()
        self.state["status"] = "running"
        self.save()
        self.event("controller_started", pid=os.getpid(), root=str(self.root))
        try:
            while True:
                pending = [s for s in STAGES if self.stage(s)["status"] != "done"]
                if not pending:
                    self.state["status"], self.state["current_stage"] = "finished", None
                    self.event("controller_finished")
                    return
                if self.stop_requested:
                    raise Stopped()
                s = pending[0]
                if (self.dir / f"HOLD_{s}").exists() or (self.dir / "HOLD_ALL").exists():
                    raise Paused(f"stage {s} held by user (remove outputs/autopilot/HOLD_{s} or HOLD_ALL, then --resume)")
                st = self.stage(s)
                if st["status"] == "blocked":
                    raise Blocked(st["error"] or f"stage {s} blocked; use --unblock {s}")
                self.state["current_stage"] = s
                st.update(status="running", started=st["started"] or now(), error=None, attempts=st["attempts"] + 1)
                self.save()
                self.event("stage_start", stage=s, attempt=st["attempts"])
                t0 = time.time()
                fn = getattr(self, f"stage_{s}")
                while True:
                    try:
                        res = fn()
                        break
                    except (Blocked, Stopped, Paused):
                        raise
                    except RuntimeError as e:      # recoverable stage failure: bounded retries
                        if st["repairs"] >= self.config["max_repairs_per_stage"]:
                            raise Blocked(f"stage {s} failed after {st['repairs']} repair attempts: {e}")
                        st["repairs"] += 1
                        st["error"] = str(e)
                        self.save()
                        self.event("stage_retry", stage=s, repairs=st["repairs"], error=str(e)[:300])
                        self.sleep(5)
                st.update(status="done", ended=now(), evidence=res.get("evidence", {}), artifacts=res.get("artifacts", []), error=None,
                          duration_seconds=round(time.time() - t0, 1))
                self.save()
                self.event("stage_done", stage=s, seconds=round(time.time() - t0, 1), note=res.get("note"), artifacts=res.get("artifacts"))
                self.decide_next(s, res)
                self.save()
        except Stopped:
            s = self.state["current_stage"]
            if s and self.stage(s)["status"] == "running":
                self.stage(s)["status"] = "interrupted"
            self.state["status"] = "stopped"
            self.event("controller_stopped", stage=s)
        except Paused as e:
            self.state["status"] = "paused"
            self.event("controller_paused", reason=str(e))
        except Blocked as e:
            s = self.state["current_stage"]
            if s:
                self.stage(s).update(status="blocked", error=str(e))
            self.state["status"] = "blocked"
            self.event("controller_blocked", stage=s, reason=str(e))
        except Exception as e:  # unexpected: record and pause, never loop
            s = self.state["current_stage"]
            if s:
                self.stage(s).update(status="blocked", error=f"{type(e).__name__}: {e}")
            self.state["status"] = "blocked"
            self.event("controller_crashed", stage=s, error=f"{type(e).__name__}: {e}")
        finally:
            self.state["controller_pid"] = None
            self.save()

    # ------------------------------------------------------------------ CLI helpers
    def alive(self) -> int | None:
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        except OSError:
            return int(self.pid_path.read_text()) if self.pid_path.exists() else -1
        finally:
            os.close(fd)

    def status_text(self) -> str:
        st = self.state
        pid = self.alive()
        lines = [f"controller: {'RUNNING pid ' + str(pid) if pid else 'not running'} | state: {st['status']} | current stage: {st['current_stage']}",
                 f"root: {self.root}", f"state file: {self.state_path}", ""]
        for s in STAGES:
            x = st["stages"][s]
            lines.append(f"  [{s}] {x['name']:28s} {x['status']:11s} attempts={x['attempts']} repairs={x['repairs']}"
                         + (f" error={str(x['error'])[:90]}" if x["error"] else ""))
        c = st["counters"]
        lines += ["", f"claude calls: {c['claude_calls']}/{self.config['max_claude_calls']} (cost ${c['claude_cost_usd']}) | "
                      f"training seconds used: {c['training_seconds']:.0f}/{self.config['max_total_training_seconds']}"]
        ev = self.find_processes("eval_gan_fid_kid.py")
        lines.append(f"external eval processes: {[p['pid'] for p in ev] or 'none'}")
        if self.hist_path.exists():
            lines += ["", "recent history:"] + ["  " + l[:200] for l in self.hist_path.read_text().splitlines()[-6:]]
        return "\n".join(lines)

    def plan_text(self) -> str:
        return ("PLAN (fixed order; each stage skipped if its outputs already validate):\n"
                "  A FID/KID evaluation  - wait for a running eval process, validate outputs (or run src/eval_gan_fid_kid.py if never started);\n"
                "                          <=2 bounded Claude-assisted repairs if it failed.\n"
                "  B Results analysis    - claude -p (read-only tools) writes outputs/gan_comparison/analysis.md from the real metrics + images.\n"
                "  C Downstream CNN      - real-only vs real+cWGAN-GP synthetic, 3 seeds, test set used only for final evaluation.\n"
                "  D Final artifacts     - final_comparison.csv/png, final_report.md, ia3_slides_outline.md, ia3_viva_questions.md.\n"
                "After each stage: validate -> claude decision (continue/stop) -> next stage. No GAN training. Pauses on any blocker.")


def daemon_main(ap_obj: Autopilot) -> int:
    fd = os.open(ap_obj.lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another autopilot controller is already running", file=sys.stderr)
        return 3
    ap_obj.pid_path.write_text(str(os.getpid()))

    def handler(signum, frame):
        ap_obj.stop_requested = True
        ap_obj.log(f"signal {signum} received: stopping gracefully")
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    ap_obj.state = ap_obj.load_state()
    for s in STAGES:                                   # a stage left 'running' by a dead controller is resumable
        if ap_obj.stage(s)["status"] in ("running", "interrupted"):
            ap_obj.stage(s)["status"] = "pending"
            ap_obj.stage(s)["child_pid"] = None
    ap_obj.run()
    return 0


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=here)
    p.add_argument("--state-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    g = p.add_mutually_exclusive_group(required=True)
    for a in ("status", "plan", "dry-run", "start", "resume", "stop", "reset", "daemon"):
        g.add_argument(f"--{a}", action="store_true")
    g.add_argument("--unblock", metavar="STAGE")
    g.add_argument("--rerun", nargs="+", metavar="STAGE", help="mark the given stages pending again (others stay done); outputs are versioned, not deleted")
    p.add_argument("--foreground", action="store_true", help="run in this terminal instead of a background daemon")
    a = p.parse_args()
    ap = Autopilot(a.root, a.state_dir, a.config)

    if a.plan:
        print(ap.plan_text())
    elif a.status:
        print(ap.status_text())
    elif a.dry_run:
        print(ap.plan_text() + "\n\nCurrent evidence (read-only checks):")
        state, info = ap.eval_state()
        print(f"  A evaluation: {state} {json.dumps(info.get('problems') or info.get('processes') or info, default=str)[:400]}")
        ok, pr, ev = ap.validate_downstream()
        print(f"  C downstream: {'valid' if ok else 'needs run'} {pr if not ok else ev}")
        for f in ("analysis.md",):
            print(f"  B analysis.md: {'exists' if (ap.eval_dir() / f).exists() else 'missing'}")
        for f in ("final_report.md", "ia3_slides_outline.md", "ia3_viva_questions.md"):
            print(f"  D {f}: {'exists' if (ap.root / 'outputs' / f).exists() else 'missing'}")
    elif a.stop:
        pid = ap.alive()
        if not pid:
            print("controller is not running")
            return 0
        if pid > 0:
            cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
            if "ia3_autopilot.py" not in cmd:
                print(f"pid {pid} is not the autopilot controller; refusing to signal it", file=sys.stderr)
                return 1
            os.kill(pid, signal.SIGTERM)
            for _ in range(60):
                time.sleep(1)
                if not ap.alive():
                    print("controller stopped; state saved (use --resume to continue)")
                    return 0
            print("controller still shutting down; check --status", file=sys.stderr)
            return 1
    elif a.unblock:
        if a.unblock not in STAGES:
            print("stage must be one of", STAGES, file=sys.stderr)
            return 2
        x = ap.stage(a.unblock)
        x.update(status="pending", repairs=0, error=None)
        if ap.state["status"] == "blocked":
            ap.state["status"] = "idle"
        ap.save()
        ap.event("unblocked", stage=a.unblock)
        print(f"stage {a.unblock} reset to pending (repair allowance restored)")
    elif a.rerun:
        if ap.alive():
            print("controller is running; stop it first", file=sys.stderr)
            return 1
        bad = [s for s in a.rerun if s not in STAGES]
        if bad:
            print("unknown stage(s):", bad, file=sys.stderr)
            return 2
        for s in a.rerun:
            ap.stage(s).update(status="pending", repairs=0, error=None, child_pid=None)
        ap.state["status"] = "idle"
        ap.save()
        ap.event("rerun_requested", stages=a.rerun)
        print(f"stages {a.rerun} reset to pending; use --resume")
    elif a.reset:
        if ap.alive():
            print("controller is running; stop it first", file=sys.stderr)
            return 1
        if ap.state_path.exists():
            ap.state_path.rename(ap.state_path.with_name(f"state.prev-{stamp()}.json"))
        print("state archived; next --start begins fresh")
    elif a.daemon or a.foreground and (a.start or a.resume):
        return daemon_main(ap)
    elif a.start or a.resume:
        if ap.alive():
            print("controller is already running; see --status")
            return 1
        exists = ap.state_path.exists()
        if a.resume and not exists:
            print("no saved state to resume; use --start", file=sys.stderr)
            return 2
        if a.start and exists and ap.state["status"] == "finished":
            print("workflow already finished; use --reset to archive state and start over")
            return 0
        if a.start and exists and ap.state["status"] in ("running", "stopped", "blocked", "idle") and any(
                ap.stage(s)["status"] != "pending" for s in STAGES):
            print("saved progress exists; continuing it (same as --resume)")
        out = open(ap.log_path, "a")
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(ap.root), "--state-dir", str(ap.dir),
                                 "--config", str(ap.config_path), "--daemon"], stdout=out, stderr=out, start_new_session=True)
        time.sleep(2)
        print(f"autopilot started in background (pid {proc.pid}); inspect with: scripts/run_ia3_autopilot.sh --status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
