#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

echo "== backend pytest =="
cd "$ROOT/backend"
pytest -q

echo "== frontend typecheck + build =="
cd "$ROOT/frontend"
npm run typecheck
npm run build

echo "YoloFlow quickstart verification OK."