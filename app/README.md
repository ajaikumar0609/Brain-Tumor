# NeuroVision AI: web app for the Brain Tumor MRI IA3 project

An **academic research prototype** (React + TypeScript + Vite + Tailwind + shadcn/ui + Recharts frontend, FastAPI + PyTorch backend) around the completed research project. It does not retrain or re-evaluate anything, and it is **not a medical device**: nothing it shows is a diagnosis.

| Page | What it does | Backed by |
|---|---|---|
| Dashboard | Overview, FID chart, A/B/C chart, model readiness | saved result files, `/api/models/status` |
| Classify MRI | Upload a PNG/JPEG, get predicted class + softmax confidence | third-party EfficientNet-B3 (Apache-2.0) |
| Generate | Pick a class, generate 4-16 synthetic 64x64 images, download | project's cWGAN-GP checkpoint (3,000 updates) |
| Gallery | Saved real / synthetic / mixed image sheets with provenance labels | `outputs/submission/figures`, `gan/*/samples_step_3000.png` |
| Experiment results | FID/KID, A/B/C accuracy and macro-F1, class-wise F1, limitations | `outputs/deduplicated_experiment/` |

**Important honesty notes** (also shown in the UI): the classification page uses a *third-party pretrained* model because the project's own A/B/C CNNs never saved checkpoints; its accuracy on this project's cluster-disjoint split is unknown. The generator did not demonstrate reliable tumour-specific anatomy; the class is a requested label. See `docs/WEB_APP_AUDIT.md`.

## Run it (Apple Silicon Mac)

One-time setup (the app has its own environments; the research `.venv` is not touched):

```bash
cd ~/Desktop/brain
python3 -m venv app/backend/.venv
app/backend/.venv/bin/pip install -r app/backend/requirements.txt
(cd app/frontend && npm install)
```

Start both servers:

```bash
./app/run_dev.sh
```

Then open **http://localhost:5173** (API docs: http://127.0.0.1:8000/docs). Ctrl+C stops both.

Separately, if you prefer two terminals:

```bash
app/backend/.venv/bin/python -m uvicorn app.backend.main:app --port 8000   # from ~/Desktop/brain
cd app/frontend && npm run dev                                              # http://localhost:5173
```

Requirements: Python 3.14 (what the research used), Node 20+ (tested on Node 25), checkpoints already in the repository (`outputs/deduplicated_experiment/gan/cwgangp/checkpoint_step_3000.pth` and `external_models/brain_tumor_classifier/weights/best_model.pth`). If a checkpoint is missing or its hash differs, that feature reports itself unavailable instead of falling back to anything else. Inference uses MPS when available, otherwise CPU.

## Tests

```bash
app/backend/.venv/bin/python -m pytest app/backend/tests -q      # 16 backend tests (real checkpoints, saved metrics)
(cd app/frontend && npm run build)                               # type-check + production build
./app/run_dev.sh &  sleep 8; ./app/smoke_test.sh                 # frontend -> proxy -> backend integration smoke test
```

## API

`GET /api/health`, `GET /api/models/status`, `POST /api/classify` (multipart `file`), `POST /api/generate` (`{"class","count","seed?"}`), `GET /api/metrics`, `GET /api/experiment`, `GET /api/gallery` (+ `/api/gallery/image/{id}`). Errors are JSON `{"error": {"code", "message"}}`. Paths are relative to the project root; configuration is in `app/.env.example`.

## Layout

```
app/
  backend/  main.py  ml.py  results.py  config.py  requirements.txt  tests/test_api.py
  frontend/ src/{pages,components,lib}  vite.config.ts  package.json  components.json
  run_dev.sh  smoke_test.sh  .env.example
docs/WEB_APP_AUDIT.md
```
