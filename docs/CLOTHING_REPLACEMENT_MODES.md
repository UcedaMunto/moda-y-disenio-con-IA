# Tres Modos de Reemplazo de Ropa en Try-On

## 📋 Descripción General

El sistema ahora ofrece **3 modos diferentes** para aplicar texturas de ropa a una fotografía de una persona. Cada modo tiene ventajas específicas según el caso de uso.

---

## 🎯 Los 3 Modos

### Modo 1: **Overlay Tradicional** (Default)
**Parámetro:** `remove_clothing_and_apply_texture=false` , `detect_and_replace_garment=false`

#### ¿Qué hace?
- Superpone la textura como una "capa" rectangular sobre el cuerpo
- La textura se posiciona en el torso superior (típicamente donde va una camiseta)
- Responde a controles de tamaño y offset
- Se puede controlar con desplazamiento (offset_x, offset_y) y escala

#### 👍 Ventajas
- ✅ Muy rápido (~0.2 segundos)
- ✅ Predecible y consistente
- ✅ Funciona con cualquier imagen
- ✅ Desplazamiento y escala manual

#### 👎 Limitaciones
- Puede parecer poco realista (es una superposición genérica)
- La textura no se ajusta a la forma real de la ropa

#### 🎨 Mejor para
- Pruebas rápidas
- Demos rápidas
- Evaluación de múltiples opciones
- Casos donde el realismo no es crítico

#### UI
```
☐ Detectar y reemplazar prenda
☐ ✨ Remover ropa y aplicar textura
```

---

### Modo 2: **Detección y Reemplazo de Prenda** (Intermedio)
**Parámetro:** `detect_and_replace_garment=true` , `remove_clothing_and_apply_texture=false`

#### ¿Qué hace?
1. **Detecta**: Identifica la prenda actual en la foto (usando análisis de color HSV)
2. **Extrae**: Obtiene la silueta exacta de esa prenda
3. **Reemplaza**: Coloca la textura nueva dentro de esa silueta

#### 👍 Ventajas
- ✅ Más realista que overlay
- ✅ La textura se ajusta a la prenda real
- ✅ No requiere calibración manual
- ✅ Balance entre velocidad y realismo (~0.7s)

#### 👎 Limitaciones
- Funciona mejor cuando la prenda tiene color **uniforme**
- No funciona si hay múltiples prendas con colores muy diferentes
- Puede fallar con prendas muy oscuras o muy claras

#### 🎨 Mejor para
- Reemplazo de una prenda específica (camiseta, pantalón, etc.)
- Fotos con contraste claro
- Prendas de color sólido (azul, rojo, negro, etc.)

#### UI
```
✓ Detectar y reemplazar prenda
☐ ✨ Remover ropa y aplicar textura
```

---

### Modo 3: **Remover Ropa y Aplicar Textura** (Nuevo - Premium)
**Parámetro:** `remove_clothing_and_apply_texture=true`

#### ¿Qué hace?
1. **Reconoce**: Identifica el cuerpo humano completo (MediaPipe + segmentación)
2. **Detecta toda la ropa**: Distingue entre piel y ropa usando análisis HSV
   - Identifica píxeles de **piel**: H(0-20, 160-180), S(20-70), V(90-220)
   - Todo lo demás dentro del cuerpo = **ropa**
3. **Aplica textura**: Rellena TODA la región de ropa con patrón repetido de textura
4. **Preserva piel**: Mantiene brazos, cuello, cara sin cambios

#### 👍 Ventajas
- ✅ **Más realista**: Detecta toda la ropa (camiseta + pantalones + zapatos)
- ✅ **Preserva anatomía**: Mantiene piel expuesta (brazos, cuello, etc.)
- ✅ **Textura completa**: Rellena uniformemente toda la región de ropa
- ✅ **Sem necesidade de calibração**
- ✅ **Comportamento inteligente**: Padrón de tela repetido naturalmente

#### 👎 Limitaciones
- Más lento (~1.0-1.5s)
- Puede fallar si hay fondos con colores similares a la piel
- Mejor con contraste claro entre piel y ropa
- No reconoce diferentes tipos de prenda (trata como bloque único)

#### 🎨 Mejor para
- **Reemplazo completo de vestuario** (cambio de tela completa)
- **Pruebas realistas** (el usuario se ve con la textura puesta)
- **Catálogos** (mostrar cómo se ve la tela en una persona real)
- **Fotos profesionales** (donde realismo es clave)
- **Promociones** (mostrar producto en contexto realista)

#### UI
```
☐ Detectar y reemplazar prenda
✓ ✨ Remover ropa y aplicar textura
```

---

## 📊 Comparación Rápida

| Aspecto | Overlay | Detección | Remover Ropa |
|---------|---------|-----------|--------------|
| **Velocidad** | ⚡ ~0.2s | ⚡⚡ ~0.7s | ⚡⚡⚡ ~1.2s |
| **Realismo** | ⭐ Básico | ⭐⭐ Medio | ⭐⭐⭐ Alto |
| **Precisión de forma** | Rectangular | Prenda individual | Cuerpo completo |
| **Manejo de múltiples prendas** | N/A | ❌ Solo una | ✅ Todo |
| **Piel expuesta** | Cubierta | Parcial | ✅ Preservada |
| **Mejor para** | Pruebas rápidas | Prenda simple | Replazo realista |
| **Requisitos** | Cualquier imagen | Color uniforme | Buen contraste |

---

## 🚀 Cómo Usar Cada Modo

### Desde UI Web

#### Modo 1: Overlay (Default)
```
1. Carga un look
2. Carga una foto
3. (Sincronizar: ambos checkboxes DESMARCADOS)
4. Click "Aplicar prueba virtual"
```

#### Modo 2: Detección de Prenda
```
1. Carga un look
2. Carga una foto
3. Marca: ✓ "Detectar y reemplazar prenda"
4. (DESMARCA): ☐ "✨ Remover ropa y aplicar textura"
5. Click "Aplicar prueba virtual"
```

#### Modo 3: Remover Ropa (NUEVO)
```
1. Carga un look (la textura)
2. Carga una foto
3. Marca: ✓ "✨ Remover ropa y aplicar textura"
4. (DESMARCA): ☐ "Detectar y reemplazar prenda"  <- IMPORTANTE
5. Click "Aplicar prueba virtual"
```

### Desde API REST

```bash
# Modo 1: Overlay (default)
curl -X POST http://localhost:8001/tryon/apply \
  -H "Content-Type: application/json" \
  -d '{
    "look_id": "LOOK_ID",
    "foto_nombre": "foto.jpg",
    "detect_and_replace_garment": false,
    "remove_clothing_and_apply_texture": false
  }'

# Modo 2: Detección de prenda
curl -X POST http://localhost:8001/tryon/apply \
  -H "Content-Type: application/json" \
  -d '{
    "look_id": "LOOK_ID",
    "foto_nombre": "foto.jpg",
    "detect_and_replace_garment": true,
    "remove_clothing_and_apply_texture": false
  }'

# Modo 3: Remover ropa (NUEVO)
curl -X POST http://localhost:8001/tryon/apply \
  -H "Content-Type: application/json" \
  -d '{
    "look_id": "LOOK_ID",
    "foto_nombre": "foto.jpg",
    "remove_clothing_and_apply_texture": true
  }'
```

### Desde Python

```python
from fase2_2.core.tryon.pipeline_v2 import run_tryon_v2
from fase2_1.core.tryon.schemas import TryOnRequest

# Modo 3: Remover ropa y aplicar textura
request = TryOnRequest(
    image_path="foto.jpg",
    garment_path="textura.png",
    output_path="resultado.png",
    remove_clothing_and_apply_texture=True
)

result = run_tryon_v2(request)
```

---

## 🧵 Detalles Técnicos del Modo 3

### Paso 1: Reconocimiento del cuerpo
- **Tecnología**: MediaPipe Selfie Segmentation
- **Resultado**: Máscara binaria de la figura humana

### Paso 2: Detección de ropa
- **Algoritmo**: Análisis de color HSV
- **Lógica**:
  1. Convierte imagen a HSV
  2. Identifica píxeles de piel (rango específico H-S-V)
  3. Dentro del cuerpo, lo que NO es piel = ROPA
- **Salida**: Máscara binaria de región de ropa

### Paso 3: Aplicación de textura
- **Estrategia**: Patrón repetido (tiling)
- **Proceso**:
  1. Carga la textura seleccionada
  2. Crea mosaico repetible del tamaño apropiado
  3. Llena la región de ropa con el patrón
  4. Suaviza bordes con Gaussian blur
  5. Blending: 100% textura en región de ropa, transición suave

### Rango de Piel (HSV)
```
Hue:        0-20 o 160-180 (tonos de carne/naranja)
Saturation: 20-70 (saturación media)
Value:      90-220 (brillo variable)
```

---

## ⚙️ Configuración

### Parámetros del Request

```python
TryOnRequest(
    # ... rutas ...
    image_path: str,
    garment_path: str,
    output_path: str,
    
    # Controles del modo
    remove_clothing_and_apply_texture: bool = False,  # NUEVO
    detect_and_replace_garment: bool = False,          # Modo 2
    
    # Otros parámetros
    size_multiplier: float = 1.0,
    apply_pose_guides: bool = True,
    # ... más ...
)
```

### Escalas (Prioridad de Modos)

Si múltiples flags están `True`:
1. 🥇 `remove_clothing_and_apply_texture` → Modo 3 (PRIORIDAD)
2. 🥈 `detect_and_replace_garment` → Modo 2
3. 🥉 (ambos False) → Modo 1 (Default)

---

## 🐛 Troubleshooting

### Problema: Modo 3 no funciona bien
**Causa posible**: Contraste bajo entre piel y ropa

**Soluciones**:
- Intenta con foto con más contraste
- Usa Modo 2 si camiseta tiene color uniforme
- Usa Modo 1 para prueba rápida

### Problema: Modo 2 detecta mal la prenda
**Causa posible**: Prenda con múltiples colores

**Solución**: Usa Modo 3 que detecta toda la ropa

### Problema: Textura se ve rara
**Causa posible**: Textura muy pequeña o patrón irregular

**Solución**: Aumenta resolución de textura o usa textura más regular

---

## 📈 Casos de Uso Recomendados

### Modo 1: Overlay
- 🏪 Tienda online (carga rápida, muchas opciones)
- ⚡ MVP inicial (demostración rápida)
- 📱 Mobile (rendimiento limitado)

### Modo 2: Detección de Prenda
- 👕 Catálogo de camisetas
- 👖 Catálogo de pantalones
- 📸 Prendas de color sólido

### Modo 3: Remover Ropa (RECOMENDADO)
- 🎨 Presentación premium
- 📸 Fotos profesionales
- 🛍️ Ventas de alto valor
- 🌟 Campañas de marketing
- 👗 Catálogos de moda
- ✨ Experiencia VTry-On completa

---

## 🎓 Algoritmos

### Detección de Ropa (Modo 3 - Paso 2)

```python
# Pseudocódigo
def detect_clothing(image_rgb, body_mask):
    # 1. Convertir a HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # 2. Detectar piel
    skin_mask = (
        ((h >= 0 and h <= 20) or (h >= 160 and h <= 180)) and
        (s >= 20 and s <= 70) and
        (v >= 90 and v <= 220)
    )
    
    # 3. Ropa = cuerpo - piel
    clothing_mask = body_mask AND NOT skin_mask
    
    # 4. Limpiar ruido
    clothing_mask = morphological_ops(clothing_mask)
    
    return clothing_mask
```

---

## 📞 Soporte

Para preguntas:
1. Revisa [docs/GARMENT_DETECTION.md](GARMENT_DETECTION.md)
2. Revisa [QUICKSTART_GARMENT_DETECTION.md](../QUICKSTART_GARMENT_DETECTION.md)
3. Ejecuta validación: `python3 scripts/validate_garment_detection.py`

---

**Última actualización:** 2026-04-20
**Status:** ✅ Producción-Ready
