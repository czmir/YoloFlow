#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
export YOLOFLOW_E2E_BASE="${YOLOFLOW_E2E_BASE:-http://127.0.0.1:5173}"
pytest -q "$ROOT/e2e/test_ui.py" -m e2e
