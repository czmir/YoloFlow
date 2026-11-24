#!/usr/bin/env bash
# Start repo docker compose on Linux, hit health/API, capture logs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${YOLOFLOW_COMPOSE_ARTIFACT_DIR:-$ROOT/data/runs/compose}"
mkdir -p "$OUT"

if ! command -v docker >/dev/null 2>&1 && ! sudo -n docker info >/dev/null 2>&1; then
  echo "docker missing" >&2
  exit 2
fi

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
fi

echo "using: ${DOCKER[*]}" | tee "$OUT/meta.txt"
"${DOCKER[@]}" compose version | tee -a "$OUT/meta.txt"
uname -a | tee -a "$OUT/meta.txt"

cleanup() {
  "${DOCKER[@]}" compose logs --no-color >"$OUT/compose-logs.txt" 2>&1 || true
  "${DOCKER[@]}" compose ps -a >"$OUT/compose-ps.txt" 2>&1 || true
  "${DOCKER[@]}" compose down --remove-orphans >"$OUT/compose-down.txt" 2>&1 || true
}
trap cleanup EXIT

"${DOCKER[@]}" compose up --build -d | tee "$OUT/compose-up.txt"

ok=0
for i in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8000/health >"$OUT/health.json" 2>/dev/null; then
    ok=1
    break
  fi
  sleep 2
done
if [[ "$ok" -ne 1 ]]; then
  echo "API health did not become ready" >&2
  exit 3
fi

curl -fsS http://127.0.0.1:8000/health | tee "$OUT/health.json"
curl -fsS http://127.0.0.1:8000/api/v1/datasets >"$OUT/datasets.json" || true
curl -fsS http://127.0.0.1:8000/api/v1/training/jobs >"$OUT/jobs.json" || true
curl -fsS http://127.0.0.1:8000/api/v1/models >"$OUT/models.json" || true
curl -sS -o /dev/null -w "frontend_http=%{http_code}\n" http://127.0.0.1:5173/ | tee "$OUT/frontend-http.txt" || true

echo "compose acceptance ok" | tee "$OUT/result.txt"
