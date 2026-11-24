#!/usr/bin/env bash
# Start API + Vite UI for local development (Linux/macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

API_PORT="${YOLOFLOW_API_PORT:-8000}"
UI_PORT="${YOLOFLOW_UI_PORT:-5173}"

uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!
cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT/frontend"
npm run dev -- --host 127.0.0.1 --port "$UI_PORT"
