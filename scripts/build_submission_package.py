#!/usr/bin/env python3
"""Build the remaining submission files: VIVA_QUESTIONS.md, SUBMISSION_CHECKLIST.md, README.md, IA3_SOURCE_CODE.zip. (No training/evaluation.)

    python scripts/build_submission_package.py
Run scripts/build_submission_docs.py first. Numbers are read from saved result files; existing files are renamed *.prev-<ts>."""
import hashlib, json, re, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "outputs/deduplicated_experiment"
OUT = ROOT / "outputs/submission"
TS = time.strftime("%Y%m%d-%H%M%S")
J = lambda p: json.load(open(D / p))
split, ds, dcfg, gm, gcfg, gv = J("dedup_audit_summary.json"), J("downstream/summary.json"), J("downstream/config.json"), J("gan_comparison/metrics.json"), J("gan_comparison/config.json"), J("gan/gan_verification.json")
ver = J("verification.json")
C = ds["conditions"]
A, B, Cc = C["A_real_only"], C["B_real_plus_cwgangp_synthetic"], C["C_real_only_matched_composition"]
agg = gm["aggregate"]
GA = {k: agg[k] for k in ("DCGAN", "WGAN-GP", "cWGAN-GP")}
rank = sorted(GA, key=lambda k: GA[k]["fid_mean"])
floor = agg["Real train reps (class-matched, reference)"]
cnt = split["partition_class_counts_representatives"]
NT = ds["n_test"]
f3 = lambda x: f"{x:.3f}"
pm = lambda m, s, d=3: f"{m:.{d}f} ± {s:.{d}f}"
npass = sum(r["pass"] for r in ver)


def versioned(p):
    if p.exists():
        p.rename(p.with_name(f"{p.stem}.prev-{TS}{p.suffix}"))


# ------------------------------------------------------------------ viva
src = (D / "ia3_viva_questions.md").read_text().strip().split("\n")
n0 = len(src)
extra = [
    ("Where did the dataset come from, and what is its licence?", "It was downloaded in Chrome and no URL was recorded, but it matches the Kaggle page for Brain Tumor MRI Dataset v2 (Masoud Nickparvar, CC BY 4.0; https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on file count (7,200), per-class split (1,400/400) and size (167.85 MB). The original download event remains unverified, and the dataset is not redistributed."),
    (f"Why is the final test set so small and what does that mean?", f"After removing near-duplicates only {split['representatives']:,} of {split['pool_images']:,} images are unique, leaving {NT} test images ({cnt['test']['notumor']} no-tumour). A single accuracy has about ±3 points of sampling error and per-class figures, especially no-tumour, are much less certain."),
    ("What hardware did you use and how long did it take?", f"A MacBook Air (M4, 16 GB) using PyTorch's MPS GPU backend. Retraining took about {gv['models']['DCGAN']['runtime_seconds'] / 60:.0f} min (DCGAN), {gv['models']['WGAN-GP']['runtime_seconds'] / 60:.0f} min (WGAN-GP) and {gv['models']['cWGAN-GP']['runtime_seconds'] / 60:.0f} min (cWGAN-GP, cumulative) while Low Power Mode was on; downstream training and FID/KID each took a few minutes."),
]
viva = ["# Viva questions and answers (final, de-duplicated experiment)", "",
        "All numbers come from `outputs/deduplicated_experiment/` (verified: " + f"{npass}/{len(ver)}" + " checks). Questions 1-20 come from the verified question set; 21-23 were added for the submission package.", ""]
viva += src + [f"{n0 + i}. **Q: {q}** A: {a}" for i, (q, a) in enumerate(extra, 1)]
versioned(OUT / "VIVA_QUESTIONS.md")
(OUT / "VIVA_QUESTIONS.md").write_text("\n".join(viva) + "\n")

# ------------------------------------------------------------------ README (submission)
readme = f"""# IA3 submission: Brain Tumor MRI Image Generation and Classification Using a Hybrid Conditional WGAN-GP

Student: **Ajai Kumar N** (URK23CS5064), B.Tech in Computer Science and Engineering (specialization: Artificial Intelligence), Karunya Institute of Technology and Sciences. Guide / faculty: **[to be filled in]**. Submission date: **[to be filled in]**.

## What is in this package
| File | Content |
|---|---|
| `FINAL_REPORT.pdf` / `FINAL_REPORT.docx` | The final report (same content), with figures and tables from the verified experiment |
| `IA3_Presentation.pptx` | 14-slide presentation |
| `VIVA_QUESTIONS.md` | Viva questions with answers |
| `SUBMISSION_CHECKLIST.md` | Verified results, disclosures, open items and what to submit |
| `IA3_SOURCE_CODE.zip` | Source code, scripts, requirements and small result tables (no data, weights or images) |
| `MANIFEST.md` / `manifest.json` | File list with sizes, SHA-256 hashes and verification status |
| `figures/` | The figures used in the report and slides; `FINAL_REPORT_source.md` is the report's markdown source |

## Headline results (final, de-duplicated experiment; details in the report)
- Data: {split['pool_images']:,} images pooled, {split['n_clusters']:,} near-duplicate clusters, {split['representatives']:,} unique images split by cluster into {split['partition_totals']['train']:,} train / {split['partition_totals']['val']} validation / {split['partition_totals']['test']} test.
- GAN quality (FID / KID x1000 vs {gcfg['n_ref']} test images): {'; '.join(f"{k} {GA[k]['fid_mean']:.1f} / {GA[k]['kid_mean'] * 1000:.1f}" for k in rank)}. **DCGAN has the best FID and KID**; the proposed cWGAN-GP is second by a small margin over WGAN-GP.
- Downstream test accuracy ({NT} test images, 3 seeds): A real only {pm(A['accuracy']['mean'], A['accuracy']['std'])}; B real + synthetic {pm(B['accuracy']['mean'], B['accuracy']['std'])}; **C matched-size real-only {pm(Cc['accuracy']['mean'], Cc['accuracy']['std'])} (highest mean)**. No evidence that the synthetic images help beyond added training volume.

## Limitations (summary; full list in the report)
Small imbalanced data ({NT} test images, {cnt['test']['notumor']} no-tumour); heuristic near-duplicate detection with no patient IDs, so patient-level independence is not guaranteed; 3 seeds and one training run per GAN; unverified synthetic labels; the cWGAN-GP was resumed after an interrupted attempt (non-finite loss at step {gv['models']['cWGAN-GP'].get('interrupted_at', 2861)}, cause unconfirmed); the earlier experiment on the original folders is preliminary and superseded; dataset matched to Kaggle v2 (CC BY 4.0) but original download record unverified; no clinical claims.

## What is NOT included, and how to obtain it
- **The MRI dataset.** It is not redistributed (the package does not need it). It matches Kaggle's *Brain Tumor MRI Dataset* v2 by Masoud Nickparvar, CC BY 4.0, https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset (original download record unverified; its sources are figshare, SARTAJ and Br35H, each with its own terms). Cite: Nickparvar, Brain Tumor MRI Dataset, Kaggle. Place the images you hold as `archive/Training/<class>/*.jpg` and `archive/Testing/<class>/*.jpg` with class folders `glioma`, `meningioma`, `notumor`, `pituitary`. Results will only match if the data are the same files (the split manifest lists the file names used).
- **Inception-v3 weights for FID/KID** (`weights-inception-2015-12-05-6726825d.pth`, 95,628,359 bytes, SHA-256 starting `6726825d`): download from `https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth` into `external_models/inception_fid_weights/`.
- **Trained GAN checkpoints** (about 80 MB each) are not included; they are regenerated by `scripts/run_dedup_gans.py`.
- Optional side checks (`src/eval_pretrained_classifier.py`) need a third-party pretrained classifier, which is not included and is not used for any final result.

## Reproducing the final experiment (macOS, Apple Silicon recommended; about 1-1.5 h of GPU time, keep the charger connected)
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python src/dedup_split.py --threshold 0.98      # cluster-aware split -> outputs/deduplicated_experiment/split_manifest.csv
python src/dedup_refine.py                      # second-stage refinement (quarantines residual near-copies)
python scripts/run_dedup_gans.py                # DCGAN, WGAN-GP, cWGAN-GP, 3,000 generator updates each (resumable)
python src/dedup_downstream.py                  # downstream A / B / C, 3 seeds
python src/dedup_eval_fid_kid.py                # FID / KID against the test partition
python scripts/verify_dedup_experiment.py       # 23 checks (check 10 refers to the original workspace and fails elsewhere)
```
`results_deduplicated_experiment/` inside the source archive holds the small result files of the reported run (split manifest, metrics, per-run results and predictions) so values can be compared without retraining.
"""
versioned(OUT / "README.md")
(OUT / "README.md").write_text(readme)

# ------------------------------------------------------------------ checklist (submission)
chk = f"""# Submission checklist (final package)
Verification of the experiment: `scripts/verify_dedup_experiment.py` passed **{npass}/{len(ver)}** checks. Package files were opened or parsed and cross-checked against the saved results (`MANIFEST.md`).

## Readiness
**Ready to submit once the placeholders are filled in and the open items below are decided.** The results are modest and honestly reported: DCGAN had the best FID/KID; the matched-size real-only control had the highest mean downstream accuracy; no benefit of the hybrid GAN or of synthetic augmentation is shown. The experiment is **not** proven free of patient-level leakage.

## Files to submit
`FINAL_REPORT.pdf` (and `.docx` if an editable copy is wanted), `IA3_Presentation.pptx`, `VIVA_QUESTIONS.md` (for your own preparation), `IA3_SOURCE_CODE.zip` if source code is requested, this checklist and `README.md` if helpful. Keep `MANIFEST.md` for your records.

## Key measured results (test set n={NT}; mean ± sd over 3 seeds)
| Item | Result |
|---|---|
| FID (lower is better) | {'; '.join(f"{k} {GA[k]['fid_mean']:.1f}" for k in rank)}; real-vs-real floor {floor['fid_mean']:.1f} |
| KID x1000 | {'; '.join(f"{k} {GA[k]['kid_mean'] * 1000:.1f}" for k in rank)} |
| Test accuracy A / B / C | {pm(A['accuracy']['mean'], A['accuracy']['std'])} / {pm(B['accuracy']['mean'], B['accuracy']['std'])} / {pm(Cc['accuracy']['mean'], Cc['accuracy']['std'])} |
| Macro-F1 A / B / C | {f3(A['macro_f1']['mean'])} / {f3(B['macro_f1']['mean'])} / {f3(Cc['macro_f1']['mean'])} |
| Validation accuracy (epoch selection only) A / B / C | {f3(A['val_accuracy']['mean'])} / {f3(B['val_accuracy']['mean'])} / {f3(Cc['val_accuracy']['mean'])} |

## Disclosures that must stay in anything you submit
Cluster-aware split method and residual patient-level uncertainty; small test set and no-tumour imbalance; earlier leakage-affected experiment is preliminary and superseded; conditional GAN's limited, not tumour-specific, conditioning; 3 seeds and one training run per GAN; unconfirmed cause of the non-finite loss at cWGAN-GP step 2,861 (resumed from step 2,500); dataset matched to Kaggle v2 (CC BY 4.0) but download record unverified; no clinical claims.

## Final technical closeout (actual results of the last run)
- Package integrity verifier (`scripts/verify_submission_package.py`): 44/44 checks passed. Experiment verifier (`scripts/verify_dedup_experiment.py`): 23/23 passed. No training or evaluation was run.
- All nine deliverable files exist and the manifest hashes match them. The source ZIP holds no images, weights, archives, environments, secrets or machine-specific absolute paths. Previous builds are preserved in `_build_history/`.
- Presentation: LibreOffice is not installed on this Mac and no native render was possible here. A LibreOffice render made elsewhere (before the slide 3/8/12 alignment fix) found no must-fix layout defects; the alignment fix is confirmed in the file's XML and in the approximate preview only. PowerPoint/Keynote check: **still open**.
- Guide / faculty name and submission date: pending administrative fields, not technical blockers.

## Open items for you
1. Fill in the two remaining placeholders, **guide / faculty** and **submission date**, in the report (DOCX, then regenerate or edit the PDF), the presentation title slide and the README. Student name, register number, programme and institution are already filled in.
2. **Dataset provenance: identified, not proven.** The data match Kaggle's Brain Tumor MRI Dataset v2 (Nickparvar, CC BY 4.0) on count, per-class split and size, but no local download record exists. Check your own Kaggle/browser download history if you want to confirm, and cite the dataset in your report.
3. Read the report and slides yourself. The figures and numbers are generated from saved files; the wording is hand-written and hedged.
4. **Open `IA3_Presentation.pptx` in PowerPoint or Keynote and check every slide.** Keynote is installed here but could not be driven from the command line, so there is no genuine render. An approximate shape-based preview (`slide_preview/`, with `PREVIEW_REVIEW.md`) was produced and led to two small fixes (slide 4 line breaks, slide 5 symbol); the preview is not a PowerPoint/Keynote render and is not a visual verification. A second pass enlarged images and added footer clearance on slides 7-11 and 14; remaining possible issues are small text inside chart images and the unbalanced note box on slide 9. A LibreOffice render supplied by the user's ChatGPT session found no must-fix layout defects. Please still check slides 4, 5, 7-11 and 14 in PowerPoint or Keynote.
5. The package contains no dataset, weights or checkpoints. If your institution requires them, check the licences first.

## Not for submission (internal)
Everything under `outputs/` outside `submission/` and `deduplicated_experiment/` (earlier, superseded experiments), all `*.prev-*` versions, `_exploratory_attempt_threshold_0.99/`, `gan/cwgangp.failed-attempt-*`, `_build_history/`, checkpoints and `external_models/`.
"""
versioned(OUT / "SUBMISSION_CHECKLIST.md")
(OUT / "SUBMISSION_CHECKLIST.md").write_text(chk)

# ------------------------------------------------------------------ source zip
SRC_EXCLUDE = {"downstream_matched_control.py"}                       # superseded early-experiment code
SCR_INCLUDE = ["audit_near_duplicates.py", "refined_similarity_probe.py", "refined_cross_partition_check.py", "run_dedup_gans.py", "run_dedup_stage2.py", "verify_dedup_experiment.py",
               "build_dedup_report.py", "build_submission_docs.py", "build_submission_package.py", "verify_submission_package.py", "render_slide_preview.py"]
RES = ["split_manifest.csv", "split_config.json", "dedup_audit_summary.json", "verification.json", "gan/gan_verification.json", "gan_comparison/metrics.json", "gan_comparison/config.json",
       "gan_comparison/fid_kid_results.csv", "gan_comparison/per_class_fid_kid_cwgangp.csv", "gan_comparison/class_confusion_kid_cwgangp.csv", "downstream/summary.json", "downstream/config.json",
       "downstream/comparison_A_B_C.csv", "downstream/classification_report.txt", "downstream/resampling_manifest_C.json"]
for m in ("dcgan", "wgangp", "cwgangp"):
    RES += [f"gan/{m}/metrics.json", f"gan/{m}/config.json", f"gan/{m}/losses.csv"]
RES += [str(p.relative_to(D)) for p in sorted((D / "downstream/runs").glob("*.json"))]
req = """torch==2.14.0
torchvision==0.29.1
torch-fidelity==0.4.0
numpy==2.4.0
Pillow==12.2.0
scipy==1.18.0
scikit-learn==1.9.1
matplotlib==3.11.0
pandas==3.0.3
tqdm==4.68.3
# report/presentation builders only:
python-docx==1.2.0
python-pptx==1.0.2
reportlab==5.0.1
"""
HOME = "/Users/ajaikumarn"


def clean(text: str) -> str:
    return text.replace(str(ROOT) + "/", "").replace(str(ROOT), ".").replace(HOME, "~")


zp = OUT / "IA3_SOURCE_CODE.zip"
versioned(zp)
included = []
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    def add(name, text):
        z.writestr("IA3_SOURCE_CODE/" + name, text)
        included.append(name)
    add("README.md", readme.replace("`FINAL_REPORT.pdf` / `FINAL_REPORT.docx`", "(report files are delivered separately)").split("## What is NOT included")[0].split("## What is in this package")[0] +
        "## About this archive\nSource code, scripts and small result files of the final de-duplicated experiment. The report, presentation and viva questions are delivered separately.\n\n" +
        "## What is NOT included" + readme.split("## What is NOT included")[1])
    add("requirements.txt", req)
    for p in sorted((ROOT / "src").glob("*.py")):
        if p.name not in SRC_EXCLUDE:
            add(f"src/{p.name}", clean(p.read_text()))
    for n in SCR_INCLUDE:
        p = ROOT / "scripts" / n
        if p.exists():
            add(f"scripts/{n}", clean(p.read_text()))
    for r in RES:
        p = D / r
        if p.exists():
            add(f"results_deduplicated_experiment/{r}", clean(p.read_text()))
bad = []
with zipfile.ZipFile(zp) as z:
    for n in z.namelist():
        t = z.read(n).decode("utf-8", "ignore")
        if HOME in t or re.search(r"((api[_-]?key|token|secret|passwd|password)\s*[:=]\s*[\"'][^\"']{6,}[\"'])|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}", t, re.I):
            bad.append(n)
print("zip files:", len(included), "| size", zp.stat().st_size, "| flagged (abs path/secret-like):", bad)
