#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "No existe entorno virtual en $ROOT_DIR/.venv"
  echo "Crea e instala dependencias con:"
  echo "  cd fase3_3/local_tryon_api && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

export PYTHONPATH="$ROOT_DIR"
cd "$ROOT_DIR"

exec "$ROOT_DIR/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8021 --reload
