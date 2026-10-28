# NeuroVision AI web app: pre-implementation audit

Audit date: 2026-10-12. Scope: which existing models and result files can honestly back each feature. Nothing was retrained or re-evaluated.

## Verdict

| Capability | Inference-ready? | Backing model / files | Notes |
|---|---|---|---|
| Synthetic MRI generation | **Yes** | `outputs/deduplicated_experiment/gan/cwgangp/checkpoint_step_3000.pth` (key `G`, 3,000 generator updates, 81.8 MB). Architecture `CondGenerator` in `src/cwgangp_train.py` (z 100, label embedding 50, 64x64 grayscale, Tanh). | The state dict loads with `strict=True` into an identical copy of the class. The checkpoint's stored config gives `class_to_idx` = glioma 0, meningioma 1, no tumour 2, pituitary 3 and `classifier_guidance: False`. BatchNorm runs in eval mode. Trained on the 2,311-image cluster-disjoint training partition. |
| MRI classification: **project's own CNNs (A/B/C)** | **No** | None. | `src/dedup_downstream.py` keeps the best-epoch weights in memory and writes only metrics and per-image test predictions (`downstream/runs/*.json`). **No downstream CNN checkpoint exists on disk.** Making one would require retraining, which this task forbids. These models are therefore not exposed for inference, and the app says so. |
| MRI classification: **third-party pretrained model** | **Yes, with caveats** | `external_models/brain_tumor_classifier/weights/best_model.pth` (EfficientNet-B3, 1-channel input, Dropout(0.4)+Linear(1536,4); Apache-2.0; SHA-256 `7840ed87...`, recorded in `external_models/README.md`). | Loads `strict=True` with `weights_only=True`; the architecture is rebuilt with `weights=None`, so no ImageNet download happens. Preprocessing copied from the model's own `app.py`: grayscale, resize 224x224, normalise with mean 0.1856 and std 0.1892. Class order glioma / meningioma / notumor / pituitary. Earlier locally measured accuracy on `archive/Testing` (1,600 images) was 0.959 (`outputs/pretrained_classifier/`), but that figure is **not** shown as evidence of quality: the model was trained on an earlier version of the Kaggle dataset, and this project found near-duplicates across that dataset's train and test folders, so the score may be inflated. |
| Saved model-comparison results | **Yes** | `gan_comparison/metrics.json`, `downstream/summary.json`, `dedup_audit_summary.json`, `split_config.json`, `split_manifest.csv`, `gan/gan_verification.json` | All under `outputs/deduplicated_experiment/`. FID/KID, A/B/C metrics, per-class F1/recall and mean confusion matrices are read at request time. The backend only multiplies KID by 1000 (as the report does). |
| Gallery | **Yes** | Figure sheets in `outputs/submission/figures/` and `gan/*/samples_step_3000.png` | Served from a fixed id-to-file registry. Real vs synthetic is labelled only where the project records provenance. Individual dataset images are not served. |

## Blockers and decisions

1. **No checkpoint for the project's own classifier.** The classification page uses the third-party EfficientNet-B3 and says so in the UI and in every API response (`caveats`). It never claims the A/B/C classifiers produced the prediction. To get the project's own classifier, someone must retrain and save it, which was out of scope.
2. **No out-of-distribution detection.** Any uploaded image receives one of four labels.
3. **Python 3.14 + torch 2.14.0 + MPS** is the environment the research used; the app has its own virtualenv (`app/backend/.venv`) with the same torch/torchvision versions so the research `.venv` is untouched.
4. numpy 2.4.0 is a yanked release (pip warns) but is the version the research code used; it was kept for parity.

## Result files used for the saved metrics

- FID/KID: `outputs/deduplicated_experiment/gan_comparison/metrics.json` (`aggregate`)
- Classifier A/B/C: `outputs/deduplicated_experiment/downstream/summary.json` (`conditions`)
- Dataset/split facts: `split_manifest.csv`, `dedup_audit_summary.json`, `split_config.json`
- Limitations text: mirrors `outputs/submission/FINAL_REPORT.pdf` and is stored in `app/backend/results.py` (`experiment()`); update it there if the report changes.
