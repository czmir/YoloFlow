#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
(cd frontend && npm install)

mkdir -p data/datasets data/workspaces data/uploads
echo "YoloFlow setup complete."