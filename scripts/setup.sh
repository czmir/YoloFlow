#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if python3 -m venv .venv 2>/dev/null; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
  pip install -r backend/requirements.txt -r backend/requirements-dev.txt
else
  echo "note: python3-venv unavailable; install deps with: pip install -r backend/requirements.txt"
  pip3 install --user -U pip
  pip3 install --user torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
  pip3 install --user -r backend/requirements.txt -r backend/requirements-dev.txt || true
fi

(cd frontend && npm install)
mkdir -p data/datasets data/workspaces data/uploads
echo "YoloFlow setup complete."