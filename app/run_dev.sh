#!/usr/bin/env bash
# Start the NeuroVision AI backend (port 8000) and frontend (port 5173). Ctrl+C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -x app/backend/.venv/bin/python ] || { echo "Backend venv missing. Run: python3 -m venv app/backend/.venv && app/backend/.venv/bin/pip install -r app/backend/requirements.txt"; exit 1; }
[ -d app/frontend/node_modules ] || { echo "Frontend deps missing. Run: (cd app/frontend && npm install)"; exit 1; }
if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 || lsof -iTCP:5173 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8000 or 5173 is already in use. Stop the other process first."; exit 1
fi
app/backend/.venv/bin/python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 &
BACK=$!
(cd app/frontend && npm run dev) &
FRONT=$!
trap 'kill $BACK $FRONT 2>/dev/null || true' EXIT INT TERM
echo "Backend: http://127.0.0.1:8000/docs   Frontend: http://localhost:5173"
wait
