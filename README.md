# Brain Tumor MRI Image Generation and Classification Using a Hybrid Conditional WGAN-GP

A starter scaffold for the IA3 project. This is **not yet the finished experiment**. It sets up the environment and verifies the dataset before model training.

## Planned experiment

- Dataset: four-class 2D Brain Tumor MRI dataset (glioma, meningioma, pituitary, no tumor)
- Baselines: DCGAN and WGAN-GP
- Proposed model: conditional WGAN-GP (cWGAN-GP)
- Image size for the first pass: 64 x 64 grayscale
- GAN evaluation: FID/KID plus visual sample inspection
- Downstream evaluation: the same CNN trained on real-only and GAN-augmented training data; report accuracy, macro precision, macro recall, macro F1, and confusion matrix

## 1. Create and activate a Python environment (macOS)

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Verify PyTorch and Apple GPU support

```bash
python src/check_environment.py
```

PyTorch can use Apple's Metal/MPS backend where supported. The script reports the selected device; it does not install any system components.

## 3. Download the dataset

Download and extract the dataset from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Place the extracted dataset so this project contains:

```text
data/raw/Training/glioma/
 data/raw/Training/meningioma/
 data/raw/Training/notumor/
 data/raw/Training/pituitary/
data/raw/Testing/...
```

Folder capitalization/class spelling can vary. Do not upload the dataset or patient data to GitHub. Check the dataset's terms and cite the source in the report.

## 4. Validate the folder layout

```bash
python src/inspect_dataset.py --data-dir data/raw
```

The script only counts common image files and reports folder counts. It does not move, rename, or delete files.

## Project rules

1. Do not train the GAN on the final test images.
2. Use the same split and same classifier architecture for every augmentation experiment.
3. Keep outputs/checkpoints out of Git until you decide how to store them; model files and MRI data can be large.
4. Do not claim results until they are measured from your own run.
5. This is an academic prototype, not a clinical diagnostic system.

## Pretrained classifier baseline (inference only)

```bash
python3 -m venv --system-site-packages .venv   # reuses the installed torch; then: .venv/bin/pip install torchvision
.venv/bin/python src/eval_pretrained_classifier.py            # uses MPS if available, else CPU
```

- Checkpoint: `external_models/brain_tumor_classifier/weights/best_model.pth` (EfficientNet-B3, 1-channel input, 4 classes), loaded with `weights_only=True`; no ImageNet download, no training.
- Preprocessing (matches the source repo): grayscale, resize 224x224, normalize mean 0.1856 / std 0.1892.
- Class order (checkpoint): glioma, meningioma, notumor, pituitary; test folder names are mapped explicitly.
- Data: `archive/Testing` only (the script refuses a `Training` path). Outputs: `outputs/pretrained_classifier/` (`metrics.json`, `classification_report.txt`, `confusion_matrix.csv/.png`).
- Limitations: (1) the source repo's training loop saves its best checkpoint by *test-set* accuracy, so these numbers may be optimistic; (2) the checkpoint was trained on a differently packaged Kaggle split (5,712/1,311 images) than our `archive/` (1,400/400 per class), so the images may overlap and test-set independence cannot be verified; (3) this is a pretrained-baseline check, not our own model, and not for clinical use.

## Phase 2: data pipeline and WGAN-GP smoke test

```bash
.venv/bin/python src/data_pipeline.py              # counts, label map, sample batch
.venv/bin/python src/smoke_test_wgangp_mps.py      # 1 critic + 1 generator step (not training)
```

- `src/data_pipeline.py`: reads `archive/Training` only (paths containing `Testing` are rejected), 64x64 grayscale, scaled to [-1, 1]; label map glioma=0, meningioma=1, notumor=2, pituitary=3. Training counts: 1,400 per class (5,600 total), 0 corrupt.
- `src/smoke_test_wgangp_mps.py`: checks the gradient penalty (double backward), critic and generator backward passes, finite losses/gradients and parameter updates on MPS with no CPU fallback. It verifies compatibility only; no model has been trained. Captured output: `outputs/phase2_smoke_test.txt`.
- FID/KID (not set up yet): plan is `torch-fidelity` (standard Inception-v3 FID/KID; adds one small pip package, but downloads ~96 MB Inception weights on first use). Install only when needed; use identical preprocessing and recorded sample counts for real and synthetic sets. The pretrained EfficientNet features must not be reported as standard FID (a feature-space distance from them would be a separately labelled custom metric).
- Leakage rules: GANs train on `archive/Training` only. `archive/Testing` stays untouched for all downstream classifier comparisons. The pretrained classifier may have seen overlapping images, so its 95.94% is a sanity check, not a generalization claim. No augmentation benefit is claimed until measured.

## Phase 3A: DCGAN smoke test

```bash
.venv/bin/python src/dcgan_train.py --subset-size 512 --epochs 1 --out-dir outputs/dcgan_smoke
```

- Standard DCGAN (conv / transposed-conv, BatchNorm, Tanh output), z=100, non-saturating BCE-with-logits loss, Adam lr 2e-4 betas (0.5, 0.999), N(0, 0.02) init, seed 42, batch 16, MPS. Unconditional. Trains on `archive/Training` only (a fixed 512-image seeded subset in the smoke test).
- Smoke result (`outputs/dcgan_smoke/`): 32 batches in 2.2 s (~0.069 s/batch) on MPS, losses and gradients finite, both networks' parameters changed, checkpoint reloads and reproduces the same samples. D loss 1.69 -> 0.92, G loss 3.34 -> 12.28 (peak 24.7), i.e. the discriminator is ahead of the generator.
- The generated samples after 32 batches are textured noise with no brain-like structure. This run only shows the pipeline works and how fast it is; it says nothing about final quality or which GAN is best.

## Phase 3B: WGAN-GP smoke test

```bash
.venv/bin/python src/wgangp_train.py --subset-size 512 --gen-steps 32 --out-dir outputs/wgangp_smoke
.venv/bin/python src/compare_smoke_samples.py      # DCGAN vs WGAN-GP on the same fixed latents -> outputs/compare_smoke/
```

- WGAN-GP: critic loss `E[D(fake)] - E[D(real)] + 10*GP`, generator loss `-E[D(fake)]` (both minimised), 5 critic updates per generator update, Adam lr 1e-4 betas (0.5, 0.9), z=100, batch 16, no BatchNorm in the critic, seed 42. Same 512-image training subset as the DCGAN run (identical file-list hash `02b1dd2c...`) and the same fixed latent vectors. `archive/Testing` is never read.
- Result (`outputs/wgangp_smoke/`): 32 generator / 160 critic updates in 11.4 s on MPS (0.358 s per generator update; 0.066 s per critic update); all losses and gradients finite; GP mean 2.06 (min 0.02, max 3.67), i.e. the 1-Lipschitz constraint is not yet well satisfied; the critic's Wasserstein estimate grew from 0.05 to ~56 (peak ~96), so the critic is far ahead of the generator. Checkpoint loads with `weights_only=True`.
- Visual comparison (outputs/compare_smoke/): neither model produces recognisable brain structure after 32 generator updates. DCGAN: uniform grainy noise. WGAN-GP: noise with a darker border and a lighter central square (a crude "object on dark background" layout, not anatomy). Both are nearly identical across latent vectors; crude diversity (mean per-pixel std across 64 samples) is 0.064 (DCGAN) and 0.058 (WGAN-GP) vs 0.249 for real training images. These numbers are not FID/KID, and the differences between the two GANs are too small and the runs too short to rank them.
- Raw DCGAN and WGAN-GP loss values are not comparable (different objectives). Timing is not like-for-like either: DCGAN did 32 D + 32 G updates; WGAN-GP did 160 critic + 32 G updates.

## Phase 3C: hybrid conditional WGAN-GP (proposed model) — smoke test

```bash
.venv/bin/python src/cwgangp_train.py --subset-size 512 --gen-steps 32 --out-dir outputs/cwgangp_smoke
```

- Generator `G(z, y)`: 50-d learnable label embedding concatenated with z (100-d) -> transposed-conv stack -> 1x64x64 Tanh. Critic `D(x, y)`: conv trunk (no BatchNorm), sum-pooled features `h(x)`, score `w.h + b + <E(y), h>` (projection critic, unbounded scalar).
- Loss (minimised): critic `E[D(G(z,y),y)] - E[D(x,y)] + 10*GP` where real/fake share the real batch's labels and the interpolates keep the label fixed; generator `-E[D(G(z,y),y)]` with y ~ uniform over the 4 classes. 5 critic updates per generator update, Adam lr 1e-4 betas (0.5, 0.9), batch 16. No classifier guidance yet. Same 512-image subset (hash verified identical to the DCGAN/WGAN-GP smoke runs), seed 42, `archive/Testing` never read.
- Result (`outputs/cwgangp_smoke/`): 32 generator / 160 critic updates in 12.4 s on MPS (0.386 s per generator update, 0.071 s per critic update), all losses/gradients finite, both nets' parameters changed, checkpoint loads with `weights_only=True`, reloaded checkpoint reproduces the samples (same device within 1e-6; CPU max abs diff 6e-7). GP mean 2.06 (max 6.33); W-distance estimate 0.03 -> 52.6, i.e. the critic is well ahead of the generator, as in the unconditional WGAN-GP smoke run.
- Conditioning check (same latents, different label): mean |pixel difference| between labels is 0.0016, versus 0.070 between different latents with the same label (before training: 0.00005 vs 0.002). So the label is wired in and measurably changes the output, but its effect is ~40x weaker than the latent's and visually undetectable: all four class rows look the same. Class conditioning is **not demonstrated** at this stage, and the samples are not brain-like (dark border, blotchy repeated texture, almost no variation between latents). Nothing here shows class-specific tumor generation.
- The training set is exactly balanced (1,400 per class), so this model is motivated by class-controlled generation and model comparison, not by class-imbalance correction.

## Phase 3D: periodic-sample cWGAN-GP trial (500 generator updates)

`src/cwgangp_train.py` gained `--sample-every`, `--checkpoint-every` and `--resume` (architecture, loss and hyperparameters unchanged; defaults still reproduce the smoke test). Periodic checkpoints store model/optimizer states, update counts, loss history and torch/MPS/DataLoader RNG states (loaded with `weights_only=True`); resuming restarts the data order at an epoch boundary, so it is not bit-identical. Regression-tested in a scratch directory (smoke mode, periodic mode, resume) before the trial; `outputs/cwgangp_smoke/` was not touched.

```bash
.venv/bin/python src/cwgangp_train.py --subset-size 0 --gen-steps 500 --batch-size 16 --sample-every 100 --checkpoint-every 100 --seed 42 --out-dir outputs/cwgangp_500
```

- Data: all 5,600 images of `archive/Training` (1,400 per class), 0 skipped; `archive/Testing` never read. Settings: 64x64 grayscale, z=100, batch 16, 5 critic updates per generator update, lambda=10, Adam lr 1e-4 betas (0.5, 0.9), MPS, no fallback.
- Completed: 500 generator updates, 2,500 critic updates = 2,500 real-image batches (40,000 real images seen, ~7.1 passes over the data; these are updates, not epochs). Runtime 188.9 s (0.378 s per generator update, 0.069 s per critic update).
- Stability: all losses and gradients finite. Wasserstein estimate (100-step means) 39.4 -> 15.3 -> 11.1 -> 8.9 -> 7.2; gradient penalty mean 1.05 -> 0.09; generator loss 110 -> 30. The critic is still ahead (W estimate > 0), but the gap shrinks steadily. This is not evidence of convergence.
- Outputs (`outputs/cwgangp_500/`): `samples_step_0000/0100/0200/0300/0400/0500.png`, `progress_all_steps.png`, `checkpoint_step_0100..0500.pth`, `losses.csv`, `critic_steps.csv`, `metrics.json`, `config.json`, `samples_metadata.json`, `train.log`. The final checkpoint reloads with `weights_only=True` and reproduces the samples (same device within 1e-6; CPU diff 1.9e-6).
- Image quality (inspected visually): step 0 flat grey; step 100 dark blotchy rectangle; step 200 irregular rounded blob; steps 300-500 a bright, mottled, roughly head-sized mass on a dark background with softer edges than the 32-step smoke runs. The checkerboard tiling seen in the smoke test is no longer obvious. There is no recognizable skull ring, ventricles, tissue contrast or tumor; the texture is blotchy noise. Samples are only moderately varied across latents.
- Conditioning: the label effect rose from 0.00005 (step 0) to 0.025 / 0.108 / 0.113 / 0.158 / 0.149 (steps 100-500), against a latent effect of 0.22 at step 500, so the label is now a first-order influence on the output. Visually the four class rows differ mainly in overall brightness/contrast (no-tumor brightest, glioma darkest). The generated per-class mean pixel value (glioma -0.746, meningioma -0.664, notumor -0.518, pituitary -0.616) closely matches the real training-class means (-0.742, -0.668, -0.506, -0.615), so the model has learned class-level intensity statistics. That is a low-level signal and is **not** evidence of class-specific tumor anatomy.
- Limitations: one seed, 500 updates, no FID/KID, no classifier diagnostic, the same 8 latents only. Nothing here establishes realism, tumor depiction or convergence.

## Phase 3E: cWGAN-GP resumed from step 500 to 3,000 total generator updates

```bash
.venv/bin/python src/cwgangp_train.py --subset-size 0 --gen-steps 3000 --batch-size 16 --sample-every 250 --checkpoint-every 250 --seed 42 \
    --resume outputs/cwgangp_500/checkpoint_step_0500.pth --out-dir outputs/cwgangp_3000
```

`--gen-steps` is the *total*, so this ran generator updates 501-3000 (2,500 additional; 12,500 additional critic updates). The step-500 checkpoint restored both networks, both Adam states (generator step 500, critic step 2,500) and the counters. `outputs/cwgangp_500/` was preserved (checksums unchanged). The sample grids for steps 0-500 were copied into `outputs/cwgangp_3000/` so the progress montage is complete.

- Completed: 3,000 generator / 15,000 critic updates, 240,000 real images seen (~42.9 passes over the 5,600 training images; `archive/Testing` never read). Cumulative training time 1,213 s (20.2 min) on MPS, 0.405 s per generator update; no errors in the log, all losses and gradients finite. The final checkpoint (`checkpoint_step_3000.pth`) carries gen_updates=3000 / critic_updates=15000, loads with `weights_only=True`, and reproduces the samples. `losses.csv` (3,000 rows) and `critic_steps.csv` (15,000 rows) are contiguous with no duplicates. (The monitoring command was interrupted during the run; training itself was unaffected and had completed by the time it was checked.)
- Stability (500-step block means): Wasserstein estimate 16.4 -> 6.1 -> 5.2 -> 4.7 -> 4.4 -> 4.2; gradient penalty 0.36 -> 0.04; generator loss 63 -> ~15. Losses are flattening; the critic still leads slightly. Flat losses are not proof of convergence.
- Images (`outputs/cwgangp_3000/compare_steps_500_to_3000.png`, per-step `samples_step_*.png`, real-image reference `real_reference_grid.png`):
  - 500: blotchy square-ish texture. 1,000: rounded head-shaped masses with a faint outline. 1,500-3,000: recognizable head silhouettes with a bright skull rim, mixed axial/coronal/sagittal-like views, and darker central structures in some (ventricle-like) and skull-base-like structures in coronal-like samples.
  - Still clearly synthetic: soft, ghostly, low detail; no convincing gray/white-matter contrast; **no identifiable tumor** in any sample; fine anatomy is not reliable. Variation across the 8 latents is visible (pose, scale, slice orientation), but only 8 latents were inspected, so mode coverage is not established.
- Conditioning: label effect 0.149 (step 500) -> 0.168 / 0.134 / 0.140 / 0.127 / 0.158 / 0.183 / 0.137 / 0.150 / 0.155 / 0.181 (steps 750-3,000), vs latent effect ~0.21-0.25; it fluctuates and has not kept growing. Visually the no-tumor row is clearly different (brighter, higher-contrast, more T2-like, matching the real no-tumor images), while the glioma, meningioma and pituitary rows are very similar to each other (same pose per latent, slightly different brightness). Generated class mean pixel values at step 3,000 (-0.754, -0.687, -0.535, -0.640) track the real class means (-0.742, -0.668, -0.506, -0.615) a bit darker. So conditioning is meaningful at the level of image contrast/sequence type, especially no-tumor vs tumor classes; it is **not** shown to produce class-specific tumor appearance.
- Not done: no FID/KID, no classifier-consistency diagnostic, no downstream classification experiment, single seed.

## Phase 4: baselines at 3,000 updates and FID/KID comparison

Baselines were trained on all 5,600 `archive/Training` images with the unchanged objectives/hyperparameters (new opt-in `--sample-every/--checkpoint-every` options only):

```bash
.venv/bin/python src/dcgan_train.py  --subset-size 0 --epochs 9 --max-batches 3000 --batch-size 16 --seed 42 --sample-every 500 --checkpoint-every 500 --out-dir outputs/dcgan_3000
.venv/bin/python src/wgangp_train.py --subset-size 0 --gen-steps 3000 --batch-size 16 --seed 42 --sample-every 500 --checkpoint-every 500 --out-dir outputs/wgangp_3000
.venv/bin/python src/eval_gan_fid_kid.py                      # -> outputs/gan_comparison/
```

- DCGAN: 3,000 generator / 3,000 discriminator updates, 204 s on MPS. WGAN-GP: 3,000 generator / 15,000 critic updates, 1,200 s on MPS. All losses/gradients finite; checkpoints at steps 500...3,000; fixed-latent grids at steps 0, 500, ..., 3,000.
- Method (`outputs/gan_comparison/comparison_config.json`): `torch-fidelity` 0.4.0, extractor `inception-v3-compat` (standard FID Inception weights, SHA-256 prefix `6726825d` verified, pool3 2048-d, float32, TF-style resize 64->299). FID and KID computed by torch-fidelity's own functions on features extracted once. Reference = `archive/Testing` (1,600 images, evaluation only). Real and generated images share one pipeline (grayscale 64x64 uint8, replicated to 3 channels). 1,600 generated per model per seed (cWGAN-GP: 400 per label), 3 generation seeds (1001-1003), KID = 100 subsets of 1,000 (per-class: 200). As a floor, class-balanced real training images vs the test set.
- Results (mean +- std over generation seeds; one training run per model):

| Model | FID | KID x1000 |
|---|---|---|
| Real train vs test (floor) | 14.1 +- 0.1 | 0.7 +- 0.2 |
| DCGAN | 133.8 +- 1.1 | 118.8 +- 1.6 |
| cWGAN-GP (proposed) | 147.3 +- 1.3 | 135.3 +- 2.5 |
| WGAN-GP | 160.8 +- 2.0 | 149.0 +- 2.6 |

  DCGAN has the best FID and KID, then the proposed cWGAN-GP, then WGAN-GP; gaps are far larger than the generation-seed std, but training-seed variability was not measured. The proposed model beats WGAN-GP (the unconditional model it extends) but not DCGAN.
- Class evidence (cWGAN-GP, 400 vs 400 per class, small-n): per-class FID 171-198, KID x1000 148-184. Only meningioma's generated images are closest (lowest KID) to their own real class; the other three labels are not, so Inception features show **no evidence of tumour-class-specific generation** (the visible difference is no-tumor images being brighter, T2-like). Per-class FID with n=400 in 2,048-d is biased and noisy; treat KID as primary.
- Images: all three models produce head-shaped MRI-like images with skull rims and some internal structure; DCGAN is more textured but has frayed edges and bright-rim artifacts, the WGAN variants are smoother; none shows an identifiable tumour, unlike the real test images.
- Limitations: ImageNet-Inception features on 64x64 grayscale MRI upsampled to 299; FID bias at n=1,600; single GAN training seed; metrics do not establish anatomical or tumour correctness.

## Local autopilot

`scripts/IA3_AUTOPILOT.md` documents the resumable controller (`./scripts/run_ia3_autopilot.sh --status | --start | --resume | --stop`) for the remaining stages (analysis, downstream classification, final report/slides/viva). It has been tested on a scratch project (`scripts/test_ia3_autopilot.py`, 25/25 checks) but has not been started on this project yet.

## Phase 5: leakage fix, clean downstream experiment and matched-size control

- **Issue:** the cWGAN-GP used for the first downstream experiment trained on all 5,600 `archive/Training` images, including the 1,120 images the CNN used for validation/epoch selection. That run is kept and labelled preliminary (`outputs/downstream_classification/`, `outputs/PRELIMINARY_ARTIFACTS_NOTICE.md`).
- **Clean rerun:** a separate cWGAN-GP (same recipe, seed 42, 3,000 gen / 15,000 critic updates) was trained only on the 4,480 CNN-training files (`src/cwgangp_train.py --train-list`, `outputs/cwgangp_clean_3000/`); `src/downstream_classification.py --gan-ckpt ... --require-leakage-free` audits overlap and refuses to run unless it is zero.
- **Matched-size control (Condition C):** `src/downstream_matched_control.py` trains on the 4,480 real images resampled with replacement to 10,080 examples (2,520/class). Results (test accuracy, archive/Testing n=1,600, seeds 42-44): A real-only 0.811 +/- 0.047; B real + clean synthetic 0.852 +/- 0.017; C matched-size real-only 0.869 +/- 0.027. C >= B in every seed, so there is no evidence that the synthetic images help beyond training-set size.
- **Caveats found by verification:** 44 of the 1,120 validation images have exact copies in the training partition (content-level duplicates in the Kaggle Training folder; none between Training and Testing); an unexplained non-finite loss in a first attempt coincided with an Apple Metal GPU error and a Mac restart (cause unconfirmed; rerun reproduced earlier numbers and finished finite). Details: `outputs/autopilot/OVERNIGHT_RESULT.md`, `outputs/final_report.md`.
- **Reproduce:** `.venv/bin/python src/downstream_classification.py --out-dir outputs/downstream_classification_clean --gan-ckpt outputs/cwgangp_clean_3000/checkpoint_step_3000.pth --require-leakage-free`, then `.venv/bin/python src/downstream_matched_control.py`.

## Phase 6: final audit (read this before submitting)
`outputs/FINAL_SUBMISSION_CHECKLIST.md` lists verified artifacts, results, disclosures and open issues. Key new finding: `archive/Testing` is **not independent** of `archive/Training` (67% of test images have a near-identical, often mirrored, copy in Training; 56% of CNN-validation images have one in CNN-training), so test accuracies and the FID/KID reference are optimistic and the downstream conclusions are provisional (`scripts/audit_near_duplicates.py`, `outputs/final_audit/`). The dataset source/licence is unverified (augmented `aug` files present). A de-duplicated rerun is recommended but was not run.

## Phase 7: cluster-disjoint (de-duplicated) re-evaluation - the final results
Everything is in `outputs/deduplicated_experiment/` (start with `FINAL_REPORT.md`, `SUBMISSION_CHECKLIST.md`). The original train/test folders share many near-copies, so all 7,200 images were pooled, grouped into 3,479 near-duplicate clusters (`src/dedup_split.py`, `src/dedup_refine.py`) and split by cluster into 2,311 / 492 / 492 unique train / validation / test images (3,295 unique of 7,200; 207 quarantined). DCGAN, WGAN-GP and cWGAN-GP were retrained on the new training partition (`scripts/run_dedup_gans.py`; cWGAN-GP resumed from step 2,500 after an interrupted attempt of unconfirmed cause), then `src/dedup_downstream.py` (A real / B real+synthetic / C matched-composition real) and `src/dedup_eval_fid_kid.py` were run (`scripts/run_dedup_stage2.py`); `scripts/verify_dedup_experiment.py` passes 23/23 checks. Results: FID DCGAN 154.7 < cWGAN-GP 177.8 < WGAN-GP 182.1; test accuracy A 0.817, B 0.839, C 0.869 (3 seeds, n=492): no evidence that the synthetic images help beyond training volume. Limitations: small imbalanced data, heuristic de-duplication (no patient IDs), 3 seeds, single GAN runs, unverified dataset source. The earlier documents are superseded (banner added; originals kept as `*.prev-*`).
# Brain-Tumor
