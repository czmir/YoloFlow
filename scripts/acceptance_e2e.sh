#!/usr/bin/env bash
# Start FastAPI + Vite, run Playwright E2E, stop. Linux only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false
export YOLOFLOW_E2E_BASE="${YOLOFLOW_E2E_BASE:-http://127.0.0.1:5173}"
export YOLOFLOW_E2E_SCREENSHOT_DIR="${YOLOFLOW_E2E_SCREENSHOT_DIR:-$ROOT/docs/images}"
export YOLOFLOW_E2E_RESULT="${YOLOFLOW_E2E_RESULT:-$ROOT/data/runs/e2e-result.json}"

python3 "$ROOT/scripts/prepare_e2e_zip.py"
export YOLOFLOW_E2E_ZIP="${YOLOFLOW_E2E_ZIP:-$ROOT/data/runs/e2e-coco8.zip}"

uvicorn app.main:app --app-dir "$ROOT/backend" --host 127.0.0.1 --port 8000 >/tmp/yoloflow-e2e-api.log 2>&1 &
API_PID=$!
(cd "$ROOT/frontend" && npm run dev -- --host 127.0.0.1 --port 5173) >/tmp/yoloflow-e2e-ui.log 2>&1 &
UI_PID=$!
cleanup() {
  kill "$API_PID" "$UI_PID" 2>/dev/null || true
  pkill -f 'vite --host 127.0.0.1 --port 5173' 2>/dev/null || true
  pkill -f 'uvicorn app.main:app' 2>/dev/null || true
  wait "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ok=0
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null && curl -fsS http://127.0.0.1:5173 >/dev/null; then
    ok=1
    break
  fi
  sleep 1
done
if [[ "$ok" -ne 1 ]]; then
  echo "stack did not become ready" >&2
  tail -n 40 /tmp/yoloflow-e2e-api.log /tmp/yoloflow-e2e-ui.log >&2 || true
  exit 3
fi

pytest -q "$ROOT/e2e" -m e2e --tb=short
echo "e2e ok"
