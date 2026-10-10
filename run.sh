#!/usr/bin/env bash
# Feelable — one-command start (builds the web app if needed, then serves everything on one port).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PORT="${PORT:-8000}"

if [ ! -f web/dist/index.html ]; then
  if [ -d web/node_modules ]; then
    echo "→ Building the web app (first run)…"
    (cd web && npm run build)
  else
    echo "! web/dist is missing and web/node_modules isn't installed — serving the API only."
    echo "  (Install with: cd web && npm install && npm run build)"
  fi
fi

if [ -n "${GEMINI_API_KEY:-}" ]; then AI_MODE="Gemini key found"; else AI_MODE="offline mode (set GEMINI_API_KEY for smarter names)"; fi

cat <<BANNER

  🍌  Feelable — Every diagram, feelable.
  ─────────────────────────────────────────────
  Open:    http://localhost:${PORT}/
  API:     http://localhost:${PORT}/api/health
  AI:      ${AI_MODE}
  Stop:    Ctrl+C

BANNER

cd backend
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
