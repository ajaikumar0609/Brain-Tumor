"""Backend tests. Run from the project root:  app/backend/.venv/bin/python -m pytest app/backend/tests -q
Classification/generation tests exercise the plumbing with real checkpoints; their outputs are NOT scientific results."""
import base64, io, json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.backend import config
from app.backend.main import app

client = TestClient(app)
EXP = config.EXP


def _png(size=(128, 128), mode="L", fmt="PNG"):
    buf = io.BytesIO(); Image.new(mode, size, 90).save(buf, fmt); return buf.getvalue()


def test_health():
    r = client.get("/api/health"); assert r.status_code == 200 and r.json()["status"] == "ok"


def test_status_models_load():
    d = client.get("/api/models/status").json()
    assert d["generator"]["available"] and d["generator"]["generator_updates"] == 3000
    assert d["classifier"]["available"] and d["classifier"]["weights_sha256_verified"]
    assert d["project_classifiers_abc"]["available"] is False


def test_metrics_match_saved_files():
    d = client.get("/api/metrics").json()
    g = json.load(open(EXP / "gan_comparison/metrics.json"))["aggregate"]
    s = json.load(open(EXP / "downstream/summary.json"))
    gan = {r["model"]: r for r in d["gan"]}
    assert gan["DCGAN"]["fid"] == g["DCGAN"]["fid_mean"] and gan["DCGAN"]["kid_x1000"] == pytest.approx(g["DCGAN"]["kid_mean"] * 1000)
    assert gan["cWGAN-GP"]["fid"] == g["cWGAN-GP"]["fid_mean"] and gan["WGAN-GP"]["fid"] == g["WGAN-GP"]["fid_mean"]
    clf = {r["id"]: r for r in d["classifiers"]}
    assert clf["A"]["accuracy"] == s["conditions"]["A_real_only"]["accuracy"]["mean"]
    assert clf["C"]["accuracy"] == s["conditions"]["C_real_only_matched_composition"]["accuracy"]["mean"]
    assert d["n_test"] == 492
    # headline values the report states
    assert [round(gan[m]["fid"], 1) for m in ("DCGAN", "cWGAN-GP", "WGAN-GP")] == [154.7, 177.8, 182.1]
    assert [round(gan[m]["kid_x1000"], 1) for m in ("DCGAN", "cWGAN-GP", "WGAN-GP")] == [129.4, 157.6, 163.5]
    assert [round(clf[k]["accuracy"], 3) for k in "ABC"] == [0.817, 0.839, 0.869]


def test_experiment_lists_limitations():
    d = client.get("/api/experiment").json()
    text = " ".join(d["limitations"]).lower()
    for must in ("tumour-specific", "patient-level", "matched-size", "non-finite", "492"):
        assert must in text
    assert "not a clinically validated" in d["status"].lower()


def test_generate_shape_class_and_seed_reproducible():
    a = client.post("/api/generate", json={"class": "meningioma", "count": 4, "seed": 5}).json()
    b = client.post("/api/generate", json={"class": "meningioma", "count": 4, "seed": 5}).json()
    c = client.post("/api/generate", json={"class": "glioma", "count": 4, "seed": 5}).json()
    assert a["count"] == 4 and a["synthetic"] is True and a["image_size"] == 64
    im = Image.open(io.BytesIO(base64.b64decode(a["images"][0].split(",", 1)[1]))); assert im.size == (64, 64) and im.mode == "L"
    assert a["images"] == b["images"]            # same seed and class -> identical images
    assert a["images"] != c["images"]            # the class label changes the output


@pytest.mark.parametrize("body,code", [({"class": "bogus", "count": 4}, 422), ({"class": "glioma", "count": 99}, 422), ({"class": "glioma", "count": 0}, 422), ({"class": "glioma", "count": 4, "seed": -1}, 422)])
def test_generate_rejects_bad_input(body, code):
    r = client.post("/api/generate", json=body); assert r.status_code == code and "error" in r.json()


def test_classify_valid_png_and_jpeg():
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
        r = client.post("/api/classify", files={"file": (f"x.{fmt.lower()}", _png(fmt=fmt), mime)})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["predicted_class"] in config.CLASSES
        assert sum(p["probability"] for p in d["probabilities"]) == pytest.approx(1.0, abs=1e-4)
        assert d["confidence"] == max(p["probability"] for p in d["probabilities"])
        assert d["caveats"] and "not a medical diagnosis" in d["disclaimer"].lower()


def test_classify_real_dataset_images_if_present():
    root = config.PROJECT_ROOT / "archive" / "Testing"
    if not root.exists():
        pytest.skip("local dataset not present")
    for cls in config.CLASSES:
        f = sorted((root / cls).glob("*.jpg"))[0]
        r = client.post("/api/classify", files={"file": (f.name, f.read_bytes(), "image/jpeg")})
        assert r.status_code == 200 and r.json()["predicted_class"] in config.CLASSES


def test_classify_error_handling():
    assert client.post("/api/classify", files={"file": ("a.txt", b"hello", "text/plain")}).json()["error"]["code"] == "not_an_image"
    assert client.post("/api/classify", files={"file": ("a.png", b"", "image/png")}).json()["error"]["code"] == "empty_file"
    gif = io.BytesIO(); Image.new("L", (64, 64)).save(gif, "GIF")
    assert client.post("/api/classify", files={"file": ("a.gif", gif.getvalue(), "image/gif")}).json()["error"]["code"] == "unsupported_type"
    assert client.post("/api/classify", files={"file": ("tiny.png", _png((8, 8)), "image/png")}).json()["error"]["code"] == "bad_dimensions"
    big = b"\x89PNG" + b"0" * (config.MAX_UPLOAD_BYTES + 10)
    r = client.post("/api/classify", files={"file": ("big.png", big, "image/png")}); assert r.status_code == 413
    assert client.post("/api/classify").status_code == 422


def test_unavailable_model_is_reported_not_substituted(monkeypatch, tmp_path):
    from app.backend import ml
    svc = ml.GeneratorService(); monkeypatch.setattr(config, "CWGAN_CKPT", tmp_path / "missing.pth")
    assert svc.ensure() is False and "not found" in svc.error
    bad = tmp_path / "bad.pth"; bad.write_bytes(b"not a checkpoint"); monkeypatch.setattr(config, "CLASSIFIER_CKPT", bad)
    csvc = ml.ClassifierService(); assert csvc.ensure() is False and "SHA-256" in csvc.error


def test_gallery_registry_only():
    d = client.get("/api/gallery").json()["items"]
    assert d and {i["kind"] for i in d} <= {"real", "synthetic", "mixed"}
    assert client.get(d[0]["url"]).headers["content-type"] == "image/png"
    for bad in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd", "nonexistent", "samples-dcgan-3000/../x"):
        assert client.get(f"/api/gallery/image/{bad}").status_code == 404


def test_cors_restricted_to_vite_origin():
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_no_hardcoded_user_paths():
    for p in Path(__file__).parents[1].glob("*.py"):
        assert "/Users/" not in p.read_text(), p
