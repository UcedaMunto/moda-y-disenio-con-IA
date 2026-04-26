#!/usr/bin/env python3
"""Script de diagnóstico para Modo 3: Remover Ropa y Aplicar Textura."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure local packages are importable when this script is executed from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("🔍 DIAGNÓSTICO: Detección y Reemplazo de Ropa (Modo 3)")
print("=" * 70)

# Detectar si tenemos argumentos
if len(sys.argv) < 4:
    print("\nUso:")
    print("  python3 diagnose_clothing_removal.py <FOTO> <TEXTURA> <SALIDA>")
    print("\nEjemplos:")
    print("  python3 diagnose_clothing_removal.py fotos_personas/foto1.jpg data/looks/terre/texture.png resultado.png")
    sys.exit(1)

foto_path = Path(sys.argv[1])
textura_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

# Step 1: Verificar archivos
print("\n1️⃣  Verificando archivos de entrada...")
if not foto_path.exists():
    print(f"   ❌ Foto no encontrada: {foto_path}")
    sys.exit(1)
print(f"   ✅ Foto: {foto_path}")

if not textura_path.exists():
    print(f"   ❌ Textura no encontrada: {textura_path}")
    sys.exit(1)
print(f"   ✅ Textura: {textura_path}")

# Step 2: Cargar imágenes
print("\n2️⃣  Cargando imágenes...")
try:
    foto_img = np.array(Image.open(foto_path).convert("RGB"))
    print(f"   ✅ Foto cargada: {foto_img.shape}")
except Exception as e:
    print(f"   ❌ Error cargando foto: {e}")
    sys.exit(1)

try:
    textura_img = np.array(Image.open(textura_path).convert("RGB"))
    print(f"   ✅ Textura cargada: {textura_img.shape}")
except Exception as e:
    print(f"   ❌ Error cargando textura: {e}")
    sys.exit(1)

# Step 3: Importar funciones
print("\n3️⃣  Importando funciones...")
try:
    from fase2_2.core.tryon.pipeline_v2 import (
        _detect_clothing_region_all,
        _apply_texture_to_clothing_region,
    )
    print("   ✅ Funciones importadas")
except Exception as e:
    print(f"   ❌ Error importando funciones: {e}")
    sys.exit(1)

# Step 4: Crear máscara de cuerpo simulada
print("\n4️⃣  Creando máscara de cuerpo...")
body_mask = np.ones(foto_img.shape[:2], dtype=np.uint8) * 255
body_mask[:, :50] = 0  # Bordes
body_mask[:, -50:] = 0
print(f"   ✅ Máscara creada: {body_mask.shape}")

# Step 5: Detectar ropa
print("\n5️⃣  Detectando ropa...")
try:
    clothing_mask = _detect_clothing_region_all(foto_img, body_mask)
    if clothing_mask is not None:
        ropa_pixels = np.count_nonzero(clothing_mask > 0)
        print(f"   ✅ Ropa detectada: {ropa_pixels} píxeles")
        print(f"   └─ %ropa: {100 * ropa_pixels / np.count_nonzero(body_mask > 0):.1f}% del cuerpo")
    else:
        print(f"   ⚠️  No se detectó ropa (retornó None)")
        print(f"   └─ Posible causa: rango HSV no coincide con tonos en foto")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error detectando ropa: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Aplicar textura
print("\n6️⃣  Aplicando textura...")
try:
    result = _apply_texture_to_clothing_region(
        foto_img,
        clothing_mask,
        str(textura_path),
        scale=1.5
    )
    if result is not None:
        print(f"   ✅ Textura aplicada")
        print(f"   └─ Forma: {result.shape}")
        
        # Verificar cambios
        diff = np.sum(np.abs(result.astype(int) - foto_img.astype(int)) > 10)
        print(f"   └─ Píxeles modificados: {diff}")
        
        if diff > 0:
            print(f"   └─ ✅ Imagen fue modificada!")
        else:
            print(f"   └─ ⚠️  Imagen NO fue modificada")
    else:
        print(f"   ❌ Aplicación de textura retornó None")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error aplicando textura: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Guardar resultado
print("\n7️⃣  Guardando resultado...")
try:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.astype(np.uint8), mode="RGB").save(output_path)
    print(f"   ✅ Guardado en: {output_path}")
    print(f"   └─ Tamaño: {output_path.stat().st_size} bytes")
except Exception as e:
    print(f"   ❌ Error guardando resultado: {e}")
    sys.exit(1)

# Step 8: Resumen
print("\n" + "=" * 70)
print("✅ DIAGNÓSTICO COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nResultado guardado en: {output_path}")
print("\nPróximos pasos:")
print("  1. Verifica la imagen resultado")
print("  2. Si se ve correcta, el Modo 3 funciona!")
print("  3. Si ve mal, revisa:")
print("     - Los colores de piel en la foto")
print("     - El contraste entre piel y ropa")
print("     - Prueba con otra foto")

