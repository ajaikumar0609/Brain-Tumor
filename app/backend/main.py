"""NeuroVision AI backend (FastAPI). Run from the project root:  python -m uvicorn app.backend.main:app --port 8000"""
import io
import time

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import config, ml, results

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_SIDE * config.MAX_IMAGE_SIDE

app = FastAPI(title="NeuroVision AI", version="1.0.0", description="Academic research prototype around the Brain Tumor MRI IA3 project. Not a medical device.")
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
START = time.time()

DISCLAIMER = "Academic research prototype. Outputs are not a medical diagnosis and must not be used for clinical decisions."
CLASSIFIER_CAVEATS = [
    "This prediction comes from a third-party pretrained EfficientNet-B3 (Apache-2.0), not from this project's own A/B/C classifiers: those did not save model checkpoints and cannot be run.",
    "The third-party model was trained on an earlier version of the Kaggle dataset; this project found near-duplicates across that dataset's train/test folders, so its published accuracy may be inflated and has not been established on this project's cluster-disjoint test set.",
    "No out-of-distribution check is performed: any image will receive one of the four labels. Confidence is a softmax probability, not a calibrated clinical certainty.",
]
GENERATOR_CAVEATS = [
    "Images come from the saved cWGAN-GP generator (64x64, 3,000 updates). The generator did not demonstrate reliable tumour-specific anatomy.",
    "The class is a requested conditioning label, not an expert-verified tumour diagnosis. These images are synthetic and are not real patient scans.",
]


def err(status: int, code: str, message: str, **extra):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra}})


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    return err(422, "invalid_request", f"{'.'.join(str(x) for x in first.get('loc', [])[1:])}: {first.get('msg', 'invalid request')}")


@app.get("/api/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - START, 1), "disclaimer": DISCLAIMER}


@app.get("/api/models/status")
def models_status():
    gen_ok, clf_ok = ml.generator.ensure(), ml.classifier.ensure()
    return {
        "device": str(ml.generator.device),
        "disclaimer": DISCLAIMER,
        "generator": {"name": ml.generator.name, "available": gen_ok, "error": ml.generator.error, **(ml.generator.meta if gen_ok else {})},
        "classifier": {"name": ml.classifier.name, "available": clf_ok, "error": ml.classifier.error, "source": "https://github.com/Phi-007/brain-tumor-classifier",
                       "license": "Apache-2.0", "weights_sha256_verified": clf_ok, "input": "grayscale, resized to 224x224", "classes": config.CLASSES},
        "project_classifiers_abc": {"available": False, "reason": "The project's downstream CNNs (conditions A, B, C) were trained for evaluation only; no model checkpoints were saved, only their test predictions and metrics. They cannot be run on new images, and they were not retrained for this application."},
    }


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    if not ml.classifier.ensure():
        return err(503, "classifier_unavailable", "The classifier could not be loaded.", detail=ml.classifier.error)
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        return err(413, "file_too_large", f"Image is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data:
        return err(400, "empty_file", "The uploaded file is empty.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        if fmt not in ("PNG", "JPEG"):
            return err(415, "unsupported_type", f"Unsupported image format {fmt}. Upload a PNG or JPEG.")
        w, h = probe.size
        if min(w, h) < config.MIN_IMAGE_SIDE or max(w, h) > config.MAX_IMAGE_SIDE:
            return err(422, "bad_dimensions", f"Image is {w}x{h}; sides must be between {config.MIN_IMAGE_SIDE} and {config.MAX_IMAGE_SIDE} pixels.")
        img = Image.open(io.BytesIO(data)); img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        return err(415, "not_an_image", "The file could not be read as a PNG or JPEG image.")
    t0 = time.time()
    probs = ml.classifier.predict(img)
    top = max(probs, key=probs.get)
    return {"predicted_class": top, "predicted_label": config.CLASS_LABELS[top], "confidence": probs[top],
            "probabilities": [{"class": c, "label": config.CLASS_LABELS[c], "probability": probs[c]} for c in config.CLASSES],
            "model": ml.classifier.name, "device": str(ml.classifier.device), "input_size": [w, h], "inference_ms": round((time.time() - t0) * 1000, 1),
            "disclaimer": DISCLAIMER, "caveats": CLASSIFIER_CAVEATS}


class GenerateRequest(BaseModel):
    cls: str = Field(alias="class")
    count: int = Field(default=8, ge=1, le=config.MAX_GENERATE)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    model_config = {"populate_by_name": True}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.cls not in config.CLASSES:
        return err(422, "unknown_class", f"class must be one of {config.CLASSES}")
    if not ml.generator.ensure():
        return err(503, "generator_unavailable", "The generator checkpoint could not be loaded.", detail=ml.generator.error)
    t0 = time.time()
    seed, images = ml.generator.generate(req.cls, req.count, req.seed)
    return {"class": req.cls, "label": config.CLASS_LABELS[req.cls], "count": len(images), "seed": seed, "image_size": 64, "images": images,
            "model": ml.generator.name, "checkpoint": ml.generator.meta.get("checkpoint"), "generator_updates": ml.generator.meta.get("generator_updates"),
            "device": str(ml.generator.device), "generation_ms": round((time.time() - t0) * 1000, 1), "synthetic": True,
            "disclaimer": DISCLAIMER, "caveats": GENERATOR_CAVEATS}


@app.get("/api/metrics")
def get_metrics():
    try:
        return results.metrics()
    except FileNotFoundError as e:
        return err(503, "metrics_unavailable", f"Saved result file missing: {e.filename.replace(str(config.PROJECT_ROOT), '.')}")


@app.get("/api/experiment")
def get_experiment():
    try:
        return results.experiment()
    except FileNotFoundError as e:
        return err(503, "experiment_unavailable", f"Saved result file missing: {e.filename.replace(str(config.PROJECT_ROOT), '.')}")


@app.get("/api/gallery")
def get_gallery():
    return {"items": results.gallery(), "note": "Only images whose provenance is recorded in the project are listed. 'mixed' images contain labelled real and synthetic panels."}


@app.get("/api/gallery/image/{item_id}")
def gallery_image(item_id: str):
    p = results.gallery_path(item_id)
    if p is None:
        return err(404, "not_found", "Unknown gallery image.")
    return FileResponse(p, media_type="image/png")
