#!/bin/bash
set -euo pipefail

# Verifica que todas las correcciones de Modo 3 sigan presentes.

WORKSPACE_ROOT="/home/uceda/Documents/difusion/proyecto-alterno-desde-cero"
cd "$WORKSPACE_ROOT"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "❌ No se encontró intérprete en $PYTHON_BIN"
  exit 1
fi

echo "==============================================="
echo "🔎 Verificación de correcciones Modo 3"
echo "==============================================="

echo "\n1) Verificando corrección HSV en pipeline..."
if grep -q "np.array(\[0, 50, 90\]" fase2_2/core/tryon/pipeline_v2.py \
  && grep -q "np.array(\[20, 120, 240\]" fase2_2/core/tryon/pipeline_v2.py \
  && grep -q "np.array(\[160, 50, 90\]" fase2_2/core/tryon/pipeline_v2.py \
  && grep -q "np.array(\[180, 120, 240\]" fase2_2/core/tryon/pipeline_v2.py; then
  echo "   ✅ Rangos HSV corregidos detectados"
else
  echo "   ❌ No se encontraron los rangos HSV corregidos"
  exit 1
fi

echo "\n2) Verificando integración API..."
if grep -q "remove_clothing_and_apply_texture: bool = False" apps/api/main.py \
  && grep -q "remove_clothing_and_apply_texture=request.remove_clothing_and_apply_texture" apps/api/main.py; then
  echo "   ✅ Campo y passthrough API detectados"
else
  echo "   ❌ Faltan cambios de integración API"
  exit 1
fi

echo "\n3) Verificando integración UI..."
if grep -q "id=\"removeClothingToggle\"" apps/api/ui/tryon.html \
  && grep -q "remove_clothing_and_apply_texture: removeClothing" apps/api/ui/tryon.html; then
  echo "   ✅ Checkbox y envío UI detectados"
else
  echo "   ❌ Faltan cambios de integración UI"
  exit 1
fi

echo "\n4) Ejecutando diagnóstico rápido en imagen real..."
bash scripts/test_modo3.sh

echo "\n5) Ejecutando smoke tests API..."
PYTHONPATH=. "$PYTHON_BIN" -m pytest -q tests/test_looks_api.py

echo "\n==============================================="
echo "✅ Todas las correcciones de Modo 3 verificadas"
echo "==============================================="
