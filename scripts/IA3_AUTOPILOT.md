# IA3 autopilot

A bounded, resumable local controller (`scripts/ia3_autopilot.py`, standard library only) that finishes the remaining IA3
workflow: **A** FID/KID evaluation -> **B** results analysis -> **C** downstream classification -> **D** final artifacts.
It runs on this Mac, in the background, and uses the installed Claude Code CLI (`claude -p`, your existing login).

## Commands

```bash
./scripts/run_ia3_autopilot.sh --plan        # print the plan
./scripts/run_ia3_autopilot.sh --dry-run     # read-only: what each stage's validators see right now
./scripts/run_ia3_autopilot.sh --status      # controller alive?, stage table, counters, last events, external eval processes
./scripts/run_ia3_autopilot.sh --start       # start in the background (continues saved progress if any)
./scripts/run_ia3_autopilot.sh --resume      # continue after --stop / reboot / a Paused or Blocked state (after --unblock)
./scripts/run_ia3_autopilot.sh --stop        # graceful stop: saves state; terminates only a subprocess the controller itself started
./scripts/run_ia3_autopilot.sh --unblock C   # give a blocked stage a fresh repair allowance, then --resume
./scripts/run_ia3_autopilot.sh --reset       # archive state.json (renamed, not deleted) to begin from scratch
./scripts/run_ia3_autopilot.sh --start --foreground   # run in this terminal instead of the background
```

Files (all under `outputs/autopilot/`): `state.json` (stage statuses, counters, decisions), `history.jsonl` (every event),
`autopilot.log`, `config.json` (limits; edit then `--resume`), `claude_calls/` (raw CLI JSON of every Claude call),
`repairs/` (diffs), `backups/` (pre-repair copies of `src/` and `scripts/`).

## What it does

| Stage | Action | Skipped when |
|---|---|---|
| A | Detects a running `eval_gan_fid_kid.py` (by parsing real `ps` argv, never `pgrep -f`) and **waits** for it. If none and outputs are missing, validates the three 3,000-update checkpoints and runs the evaluation. Validates: FID/KID for all three GANs (3 seeds, 1,600 vs 1,600), finite values, preprocessing/reference/seeds recorded, checkpoint counters = 3,000, training runs report `trained_on_testing_dir: false`. | outputs already validate |
| B | `claude -p` with read-only tools (Read/Glob/Grep) looks at the real metrics and image grids and writes `outputs/gan_comparison/analysis.md`. The controller validates required sections and that the best-FID/best-KID model names match its own computation. | `analysis.md` already exists and stage marked done |
| C | Runs `src/downstream_classification.py`: same small CNN, stratified 80/20 split of `archive/Training`, **real-only** vs **real + cWGAN-GP synthetic** (labelled with the generation label; correctness unverified), 3 seeds. `archive/Testing` is read only after training for the final evaluation, never for selection or tuning. Unconditional GAN samples are not used. | a non-smoke `summary.json` with >=3 runs/condition and 1,600 test images already exists |
| D | `scripts/ia3_final_artifacts.py` builds `final_comparison.csv/png` from the real metric files; one `claude -p` call (read-only tools) writes `final_report.md`, `ia3_slides_outline.md`, `ia3_viva_questions.md`. References are limited to a supplied list (verify before submitting). | n/a |

After each stage the controller validates the outputs, asks Claude for a `continue`/`stop` decision (no tools; only the next fixed stage is
permitted), records the decision, and continues. If the Claude budget is exhausted it falls back to the pre-authorised default.

## Permissions it needs

* Existing Claude Code login; `claude -p` calls use `--tools` allow-lists: **Read, Glob, Grep** for analysis/report calls, none for decisions, and
  **Read, Glob, Grep, Edit** (via `--allowedTools Edit`) only for a stage-failure repair. No `--dangerously-skip-permissions`, no Bash tool for Claude.
* Python with torch/scikit-learn/matplotlib (the project `.venv`), MPS for stage C, >= 5 GB free disk.
* It writes only under `outputs/` and (for repairs) the single script named for the failing stage.

## Limits (all in `outputs/autopilot/config.json`)

Claude calls <= 14 (cost recorded per call), tool turns/call 10, USD budget/call 1.5, repairs per stage <= 2, stage timeouts
(A 2 h, B 30 min, C 90 min, D 1 h), total training time 90 min, poll interval 15 s (backs off to 60 s).

## What makes it pause (state `blocked` or `paused`; nothing is retried forever)

* a repair limit reached (2 per stage), a subprocess timeout, a missing/invalid prerequisite checkpoint, low disk or missing dependency;
* the evaluation exited 0 but its outputs fail validation (human review, no auto-edit);
* Claude's decision call answers `stop` (data-integrity concern) -> `paused`; the completed stage stays done;
* the Claude call budget is exhausted;
* an interactive permission would be required (the call fails; the controller records it and blocks).

## Safety guarantees (and how they are enforced)

* Never deletes: old outputs are **renamed** `*.prev-<timestamp>`; checkpoints/datasets/previous experiments are never touched.
* Repairs: before a Claude repair, `src/` and `scripts/` are snapshotted; any change outside the single allowed file is **reverted automatically**
  and logged; the allowed change is saved as a diff.
* Only one controller runs (an `flock` on `autopilot.lock`). It never starts GAN training and never changes GAN architectures/hyperparameters.
* `--stop` signals only the verified controller pid, and terminates only the child process it launched itself. An evaluation started by someone
  else (for example your interactive session) is waited for, never killed.
* A stage left `running` by a dead controller (terminal closed, reboot) is reset to `pending` on resume; finished stages are not repeated.

## Known limitations

* The controller is autonomous only while your Mac is awake and the background process is alive; after sleep/reboot run `--resume`.
* The analysis/report texts are written by Claude from the supplied facts; the controller checks structure and the best-model names, **not every
  sentence**. Read them before submitting. Metrics and tables are computed by code, not typed by the model.
* Stopping during a Claude call waits for that call (up to the 20-minute call timeout) before the controller exits.
* Synthetic images in stage C are labelled with the label they were generated for; the GAN evaluation gives no strong evidence that they show
  the labelled class, so a null or negative result there is a legitimate finding.

## Verification

`scripts/test_ia3_autopilot.py` exercises the controller on a scratch project with stub scripts and a stub `claude` (fixture data only):
external-process detection and waiting, stage skipping, claude/repair limits, out-of-scope-edit revert, stop/resume, pause on a Claude `stop`,
call-budget enforcement and output versioning; `--real-claude-decision` additionally makes one real `claude -p` call.

## Additions after the validation-leakage review (2026-10-12)

* **What happened:** the first full run finished (A-D) before a leakage problem was noticed: the downstream experiment's synthetic images came from a cWGAN-GP
  trained on all 5,600 `archive/Training` images, including the 1,120 images the CNN used for validation/epoch selection. `archive/Testing` was never exposed.
  That run is kept and labelled preliminary (`outputs/downstream_classification/PRELIMINARY_LEAKAGE_NOTICE.md`, `outputs/PRELIMINARY_ARTIFACTS_NOTICE.md`).
* **Clean experiment (run outside the controller, because the controller never trains GANs):** `src/downstream_classification.py --write-split-lists DIR`
  exports the exact CNN train/val lists; `src/cwgangp_train.py --train-list cnn_train_files.json ...` trains a separate cWGAN-GP on only the 4,480 CNN-training files;
  `src/downstream_classification.py --gan-ckpt <clean ckpt> --require-leakage-free` records a leakage audit (overlap of GAN-training files with CNN-validation files)
  and refuses to run unless it is 0.
* **Controller changes:** config keys `downstream_dir`, `downstream_preliminary_dir`, `downstream_gan_ckpt`, `require_leakage_free_downstream`; stage D passes the
  preliminary results to the report writer with an instruction to label them preliminary; `HOLD_<STAGE>` / `HOLD_ALL` files in `outputs/autopilot/` pause before a stage;
  a stop request is now also honoured between stages (previously only inside waits, which let the first run continue into stage D);
  `--rerun STAGE...` resets chosen stages to pending (outputs are versioned, not deleted).

## Notes from the final run (2026-10-12)
* A Mac restart killed the controller and the overnight chain; `--resume` recovered (the interrupted stage is re-run, its partial output directory is archived, not deleted).
* Config keys added for the matched-size control and report notes: `downstream_matched_dir`, `report_notes` (injected into the stage-D facts so the report states incidents/limitations accurately).
* The `Monitor`-style event watchers proved unreliable; poll the state file / `--status` instead.
