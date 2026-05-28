#!/usr/bin/env bash
set -euo pipefail

# Project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Colors
info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }

# 1) Python env and deps
if [ ! -d "venv" ]; then
  info "Creating Python virtual environment..."
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip >/dev/null

info "Installing Python dependencies..."
# Install base + backend + embedding deps if files exist
pip install -r requirements.txt >/dev/null || warn "Base requirements.txt not found or install issue"
[ -f inference/requirements.txt ] && pip install -r inference/requirements.txt >/dev/null || true
[ -f embedding/requirements.txt ] && pip install -r embedding/requirements.txt >/dev/null || true

# 2) Ensure embeddings are available (skip if already downloaded)
EMB_FILE="embedding/vectors/embeddings_20250728_221826.npy"
if [ ! -f "$EMB_FILE" ]; then
  info "Downloading embeddings from Hugging Face (one-time ~700MB)..."
  python embedding/download_from_hf.py || warn "Embedding download script failed; backend may start without embeddings"
else
  info "Embeddings already present, skipping download."
fi

# 3) Start backend (uvicorn)
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}

info "Starting backend on http://localhost:${API_PORT} ..."
uvicorn inference.backend.app:app \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --reload \
  --log-level info &
BACKEND_PID=$!

# Ensure we kill background processes on exit
cleanup() {
  warn "Shutting down dev processes..."
  kill "$BACKEND_PID" 2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

# 4) Start frontend (React dev server)
cd "$ROOT_DIR/web_ui/frontend"
info "Installing frontend dependencies..."
if command -v npm >/dev/null 2>&1; then
  npm install >/dev/null
  export PORT=${PORT:-3000}
  export REACT_APP_API_URL=${REACT_APP_API_URL:-http://localhost:${API_PORT}}
  info "Starting frontend on http://localhost:${PORT} ..."
  npm start &
  FRONTEND_PID=$!
else
  warn "npm not found. Please install Node.js (https://nodejs.org) to run the frontend."
  FRONTEND_PID=$BACKEND_PID # dummy to satisfy trap
fi

cd "$ROOT_DIR"
info "Dev environment is up:"
info "- Backend:  http://localhost:${API_PORT}/docs"
info "- Frontend: http://localhost:${PORT:-3000}"

# Keep script alive while children run
wait $BACKEND_PID $FRONTEND_PID
