#!/bin/bash
# Script rápido para probar Modo 3

WORKSPACE_ROOT="/home/uceda/Documents/difusion/proyecto-alterno-desde-cero"
cd "$WORKSPACE_ROOT"

PYTHON_BIN=".venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "❌ No se encontró intérprete en $PYTHON_BIN"
    exit 1
fi

# Buscar una foto de prueba
FOTO=$(find fotos_personas -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.avif' \) | head -1)
if [ -z "$FOTO" ]; then
    echo "❌ No se encontró foto de prueba en fotos_personas/"
    exit 1
fi

echo "📸 Foto encontrada: $FOTO"

# Buscar una textura
TEXTURA=$(find data/looks data/processed -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | head -1)
if [ -z "$TEXTURA" ]; then
    echo "❌ No se encontró textura en data/looks o data/processed"
    exit 1
fi

OUTPUT="/tmp/modo3_resultado_$(date +%s).png"

echo ""
echo "🚀 Ejecutando diagnóstico..."
echo "  Foto: $FOTO"
echo "  Textura: $TEXTURA"
echo "  Salida: $OUTPUT"
echo ""

"$PYTHON_BIN" scripts/diagnose_clothing_removal.py "$FOTO" "$TEXTURA" "$OUTPUT"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo ""
    echo "✅ Resultado guardado en: $OUTPUT"
    echo "Puedes abrir la imagen para verificar que se vea bien"
else
    echo ""
    echo "❌ Error en el diagnóstico. Ver detalles arriba."
fi

exit $STATUS
