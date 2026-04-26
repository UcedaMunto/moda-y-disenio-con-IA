#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENTRYON_DIR="$ROOT_DIR/fase2_3/vendor/opentryon"
VENV_DIR="$ROOT_DIR/fase2_3/vendor/.venv_api_cpu"
REQ_FILE="$ROOT_DIR/fase2_3/requirements-api-cpu.txt"

if [[ ! -d "$OPENTRYON_DIR" ]]; then
  echo "OpenTryOn no esta clonado en $OPENTRYON_DIR"
  echo "Ejecuta: .venv/bin/python fase2_3/scripts/bootstrap_opentryon.py --clone"
  exit 1
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE"

PYTHONPATH="$OPENTRYON_DIR" "$VENV_DIR/bin/python" - <<'PY'
import importlib

mod = importlib.import_module("api_server")
app = getattr(mod, "app", None)
if app is None:
    raise RuntimeError("api_server.app no esta disponible")
print("Smoke check ok: api_server.app importado")
PY

echo "Instalacion API/CPU completada en: $VENV_DIR"
echo "Siguiente paso: bash fase2_3/scripts/run_opentryon_api_cpu.sh"
