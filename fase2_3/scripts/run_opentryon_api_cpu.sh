#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENTRYON_DIR="$ROOT_DIR/fase2_3/vendor/opentryon"
VENV_DIR="$ROOT_DIR/fase2_3/vendor/.venv_api_cpu"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "No existe el entorno API/CPU en $VENV_DIR"
  echo "Ejecuta primero: bash fase2_3/scripts/setup_opentryon_api_cpu.sh"
  exit 1
fi

export PYTHONPATH="$OPENTRYON_DIR"
cd "$OPENTRYON_DIR"

exec "$VENV_DIR/bin/uvicorn" api_server:app --host 127.0.0.1 --port 8011
