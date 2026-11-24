#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

tmux -f /exec-daemon/tmux.portal.conf has-session -t "=yoloflow-dev" 2>/dev/null || \
  tmux -f /exec-daemon/tmux.portal.conf new-session -d -s yoloflow-dev -c "$ROOT" -- bash -l

tmux -f /exec-daemon/tmux.portal.conf send-keys -t yoloflow-dev:0.0 "cd '$ROOT' && source .venv/bin/activate && export PYTHONPATH='$ROOT/backend' && uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000" C-m
tmux -f /exec-daemon/tmux.portal.conf send-keys -t yoloflow-dev:0.0 "cd '$ROOT/frontend' && npm run dev -- --host 127.0.0.1 --port 5173" C-m

echo "API: http://127.0.0.1:8000  Frontend: http://127.0.0.1:5173 (tmux session: yoloflow-dev)"