#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv-flowpilot/bin/python ]]; then
  python -m venv --system-site-packages .venv-flowpilot
fi

REQ_FILE="flowpilot_webapp/backend/requirements.txt"
REQ_HASH="$(sha256sum "$REQ_FILE" | cut -d' ' -f1)"
REQ_STAMP=".venv-flowpilot/.flowpilot-requirements"
if [[ ! -f "$REQ_STAMP" ]] || [[ "$(cat "$REQ_STAMP")" != "$REQ_HASH" ]]; then
  .venv-flowpilot/bin/python -m pip install -r "$REQ_FILE"
  printf '%s' "$REQ_HASH" > "$REQ_STAMP"
fi

if [[ ! -x flowpilot_webapp/frontend/node_modules/.bin/vite ]]; then
  npm --prefix flowpilot_webapp/frontend ci --cache /tmp/flowpilot-npm-cache
fi
npm --prefix flowpilot_webapp/frontend run build

exec .venv-flowpilot/bin/python -m uvicorn flowpilot_webapp.backend.app:app \
  --host 0.0.0.0 \
  --port "${FLOWPILOT_PORT:-8510}" \
  --reload \
  --reload-dir flora_translate \
  --reload-dir flowpilot_webapp/backend
