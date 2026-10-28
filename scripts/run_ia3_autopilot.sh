#!/bin/bash
# Launcher for the IA3 autopilot. Usage: ./scripts/run_ia3_autopilot.sh --status | --plan | --dry-run | --start | --resume | --stop | ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" "$ROOT/scripts/ia3_autopilot.py" --root "$ROOT" "$@"
