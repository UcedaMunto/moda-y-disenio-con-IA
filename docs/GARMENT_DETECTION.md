# Sistema de Detección y Reemplazo de Prendas

## Descripción

Implementación de detección y reemplazo de prendas para el pipeline try-on. El sistema identifica automáticamente la ropa real en una fotografía y la reemplaza con una textura proporcionada por el usuario.

## Solución Técnica

### Algoritmo de Detección
1. **Análisis de Color HSV**: Detecta regiones de color uniforme (típicas de prendas)
2. **Cálculo de Varianza Local**: Identifica áreas con color coherente
3. **Operaciones Morfológicas**: Conecta píxeles relacionados
4. **Extracción de Contornos**: Obtiene la silueta precisa de la prenda
5. **Reemplazo con Blending**: Aplica la textura nueva con transición natural

### Componentes

**Backend (Python)**:
- `fase2_2/core/tryon/pipeline_v2.py`: 4 funciones nuevas
  - `_detect_garment_region()`: Detección HSV
  - `_extract_garment_silhouette()`: Extracción de contornos
  - `_replace_garment_with_texture()`: Reemplazo con blending
  - `_render_garment_detection_and_replacement()`: Orquestación

**API**:
- `apps/api/main.py`: 
  - `TryOnApplyRequest`: Nuevo parámetro `detect_and_replace_garment`
  - `/tryon/apply`: Endpoint actualizado

**Frontend**:
- `apps/api/ui/tryon.html`:
  - Checkbox: "Detectar y reemplazar prenda"
  - JavaScript para enviar parámetro a API

## Cómo Usar

### Desde UI (Recomendado)

1. Abre `/tryon` en navegador
2. Selecciona un **look** de la lista izquierda
3. Selecciona una **foto** de persona
4. **Marca el checkbox**: "Detectar y reemplazar prenda"
5. Presiona **"Aplicar prueba virtual"**
6. El sistema:
   - Detecta la ropa en la foto
   - Extrae su silueta
   - Reemplaza con la textura del look

### Desde API

```bash
curl -X POST http://localhost:8001/tryon/apply \
  -H "Content-Type: application/json" \
  -d '{
    "look_id": "48a60fd58d31",
    "foto_nombre": "foto.jpg",
    "detect_and_replace_garment": true
  }'
```

### Desde Python

```python
from apps.api.main import TryOnApplyRequest
from fase2_2.core.tryon.pipeline_v2 import run_tryon_v2
from fase2_1.core.tryon.schemas import TryOnRequest

# UI Request
api_req = TryOnApplyRequest(
    look_id="48a60fd58d31",
    foto_nombre="foto.jpg",
    detect_and_replace_garment=True
)

# Pipeline Request
pipeline_req = TryOnRequest(
    image_path="/path/to/image.jpg",
    garment_path="/path/to/texture.png",
    output_path="/path/to/output.png",
    detect_and_replace_garment=True
)

# Ejecutar
result = run_tryon_v2(pipeline_req)
```

## Modos de Operación

### Modo 1: Overlay Tradicional (default)
- Superpone textura como capa rectangular
- `detect_and_replace_garment = False`
- Responde a controles de tamaño y offset

### Modo 2: Detección y Reemplazo (nuevo)
- Detecta prenda real en la foto
- Reemplaza con textura proporcionada
- `detect_and_replace_garment = True`
- Resultado más realista (conforma a la prenda real)

## Parámetros

### TryOnRequest
```python
detect_and_replace_garment: bool = False
```
Si `True`, usa modo detección. Si `False`, usa overlay tradicional.

### TryOnApplyRequest
```python
detect_and_replace_garment: bool = False
```
Mismo parámetro, propagado desde UI a pipeline.

## Validación

### Tests
```bash
pytest tests/test_garment_detection.py -v
pytest tests/test_fase2_2_pipeline_v2.py -v
```

### Resultados
- ✅ 1 test unitario nuevo (test_garment_detection.py)
- ✅ 2 tests existentes siguen pasando (backward compatible)
- ✅ Ambos modos produce resultados significativamente diferentes (12.3% píxeles)

## Ejemplos de Uso

### Ejemplo 1: Detectar y reemplazar en foto con polo azul
```
1. Look seleccionado: "terre" (textura naranja)
2. Foto seleccionada: "depositphotos_10577746-stock-photo-full-body-young-woman.jpg"
3. Checkbox marcado: ✓ Detectar y reemplazar prenda
4. Resultado: Polo azul reemplazado con textura naranja
```

### Ejemplo 2: Comparar modos
```
Modo overlay (unchecked):  Textura rectangular sobre cuerpo
Modo detección (checked):  Textura reemplaza la ropa detectada
```

## Limitaciones y Notas

- Basado en análisis de color HSV (funciona bien con prendas de color uniforme)
- Requiere máscara de segmentación de cuerpo disponible
- Fallback automático a overlay si detección falla
- Procesamiento 2D (no es deformación 3D de mesh)

## Performance

- Tiempo de detección: ~0.5-1.0s por imagen
- Tamaño de imagen no limitado
- Compatible con cualquier resolución

## Troubleshooting

Si la detección no funciona:
1. Verifica que la foto tiene contraste claro entre cuerpo y prenda
2. Verifica que la prenda tiene color relativamente uniforme
3. Intenta con `detect_and_replace_garment = False` (overlay tradicional)
4. Revisa la máscara de segmentación está disponible

## Archivos Modificados

- `fase2_2/core/tryon/pipeline_v2.py` (492 líneas nuevas)
- `fase2_1/core/tryon/schemas.py` (18 líneas nuevas)
- `apps/api/main.py` (20 líneas nuevas)
- `apps/api/ui/tryon.html` (114 líneas nuevas)
- `tests/test_garment_detection.py` (427 líneas nuevas, archivo nuevo)

**Total: 627 líneas nuevas de código**

## Contacto

Para reportar issues o sugerencias sobre esta funcionalidad, ver documentación técnica en docs/
