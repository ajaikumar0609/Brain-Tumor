"""Paths and limits for the NeuroVision AI backend. Everything is relative to the project root (override with env vars)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("NEUROVISION_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
EXP = PROJECT_ROOT / "outputs" / "deduplicated_experiment"
SUBMISSION_FIGS = PROJECT_ROOT / "outputs" / "submission" / "figures"

CWGAN_CKPT = Path(os.environ.get("NEUROVISION_GENERATOR_CKPT", EXP / "gan" / "cwgangp" / "checkpoint_step_3000.pth"))
CLASSIFIER_CKPT = Path(os.environ.get("NEUROVISION_CLASSIFIER_CKPT", PROJECT_ROOT / "external_models" / "brain_tumor_classifier" / "weights" / "best_model.pth"))
# Recorded in external_models/README.md when the file was downloaded; the classifier is refused if the file differs.
CLASSIFIER_SHA256 = "7840ed87964a5ce06c0866b167e5624bb259a58895139c9687163c3f77d77c4a"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]          # dataset (alphabetical) order used by both models
CLASS_LABELS = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary tumour"}

MAX_UPLOAD_BYTES = int(os.environ.get("NEUROVISION_MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_IMAGE_SIDE = 8192
MIN_IMAGE_SIDE = 32
MAX_GENERATE = 16

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("NEUROVISION_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
