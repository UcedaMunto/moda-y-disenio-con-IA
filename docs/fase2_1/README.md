# FASE 2 - Virtual Try-On

## Objetivo

Extender Fabric2Mesh AI para permitir:

- Superponer una prenda sobre una persona en imagen.
- Dejar la base lista para video o secuencia de frames.
- Ajustar automáticamente la prenda al cuerpo usando IA.
- Reutilizar modelos texturizados y assets creados en Fase 1.
- Mantenernos en herramientas gratuitas, open source y entrenables.

---

## Estado actual (arranque 2.1)

Avance implementado en esta rama de trabajo:

1. Carpeta aislada de fase: `fase2_1/`.
2. Módulo baseline de try-on en `fase2_1/core/tryon/`.
3. API dedicada de fase en `fase2_1/api/main.py`.
4. Endpoint puente en API principal: `POST /fase2_1/tryon/run`.
5. Test de contrato del endpoint: `tests/test_fase2_1_api.py`.
6. Script de evaluación batch: `fase2_1/scripts/evaluate_batch.py`.
7. Configuración versionada de inferencia: `fase2_1/config/inference.yaml`.
8. Checklist de calidad visual: `fase2_1/config/quality_checklist.json`.

Comandos rápidos:

```bash
# API separada de fase 2.1
uvicorn fase2_1.api.main:app --host 0.0.0.0 --port 8010 --reload

# API principal con endpoint puente
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

# Test de Fase 2.1
python -m pytest -q tests/test_fase2_1_api.py

# Evaluación batch (baseline)
python fase2_1/scripts/evaluate_batch.py \
   --input-dir data/raw/personas \
   --garment-path data/raw/models/TShirts.obj \
   --garment-type shirt

# Preparar set interno de validación (20-50 imágenes)
python fase2_1/scripts/prepare_validation_dataset.py \
   --input-dir data/raw/personas \
   --manifest-path data/processed/fase2_1_eval/validation_manifest.json \
   --min-images 20 \
   --max-images 50

# Preparar dataset de entrenamiento para segmentación (Semana 3)
python fase2_1/scripts/prepare_segmentation_dataset.py \
   --images-dir data/raw/fase2_1_seg/images \
   --masks-dir data/raw/fase2_1_seg/masks \
   --output-dir data/processed/fase2_1_train/segmentation \
   --train-ratio 0.8 \
   --val-ratio 0.1 \
   --test-ratio 0.1 \
   --seed 42 \
   --strict

# Entrenar baseline de fit_regression (Semana 3)
python fase2_1/scripts/train_fit_regression_baseline.py \
   --dataset-jsonl data/processed/fase2_1_train/fit_regression/train.jsonl \
   --output-model data/processed/fase2_1_train/fit_regression/model_baseline.json \
   --l2 0.001

# Benchmark de latencia (Semana 4)
python fase2_1/scripts/benchmark_latency.py \
   --input-dir data/raw/personas \
   --garment-path data/raw/models/TShirts.obj \
   --garment-type shirt \
   --target-max-ms 2500
```

Metrica de validez visual manual:

- `GET /fase2_1/tryon/evaluate-summary` acepta `valid_score_threshold` (default `85.0`).
- `visually_valid_rate` = casos con `score >= valid_score_threshold` sobre total manual.
- Criterio de aceptacion Semana 4: `manual_visually_valid_rate >= 90.0`.

Salida del pipeline de datos de segmentacion:

- `manifest.json` con conteos, ratios y trazabilidad del split.
- `train.jsonl`, `val.jsonl`, `test.jsonl` con pares `image_path` + `mask_path`.

Diagrama PUML completo del flujo trabajado:

- `fase2_1/docs/proceso_completo_fase2_1.puml`

---

## Secuencia de trabajo del sistema (2.1)

```mermaid
sequenceDiagram
    participant U as Usuario/UI
    participant API as FastAPI Bridge
    participant P as Pipeline TryOn
    participant Q as Quality/Review
    participant FS as Artifacts (JSON/JSONL)

    U->>API: POST /fase2_1/tryon/run (image_path, garment_path, garment_type)
    API->>P: run_tryon(request)
    P->>P: detect_pose + segment_person
    P->>P: compute_scale(landmarks, garment_type)
    P-->>API: TryOnResult(status, output_path, scale, meta)
    API-->>U: respuesta try-on

    U->>API: POST /fase2_1/tryon/batch
    API->>P: run_tryon_batch(..., garment_type)
    P->>FS: report.json (summary/results)
    P-->>API: TryOnBatchResult
    API-->>U: resumen batch

    U->>API: POST /fase2_1/tryon/evaluate
    API->>Q: append_manual_review(criteria_scores)
    Q->>FS: manual_reviews.jsonl
    Q-->>API: score manual
    API-->>U: evaluacion guardada

    U->>API: GET /fase2_1/tryon/evaluate-summary
   API->>Q: summarize_manual_reviews(project_id, valid_score_threshold)
    Q-->>API: resumen manual
    API-->>U: metricas manuales

    U->>API: GET /fase2_1/tryon/evaluate-consolidated
   API->>Q: build_consolidated_evaluation_report(valid_score_threshold)
    Q->>FS: consolidated_report.json (opcional)
    Q-->>API: resumen combinado batch+manual
    API-->>U: metricas consolidadas
```

## Contrato canonico de landmarks y transforms

El pipeline de `run` y `batch` devuelve en `meta` dos bloques versionados:

- `landmarks_contract`: puntos clave en espacio normalizado (`image_normalized`).
- `transform_contract`: parametros de transformacion de prenda (escala, rotacion, traslacion).

Version actual: `1.0`.

```mermaid
flowchart TD
   A[Pose Landmarks - Mediapipe] --> B[build_landmarks_contract v1.0]
   C[Scale + Garment Type] --> D[build_transform_contract v1.0]
   B --> E[TryOnResult.meta.landmarks_contract]
   D --> F[TryOnResult.meta.transform_contract]
   E --> G[API /fase2_1/tryon/run]
   F --> G
   G --> H[Batch Report + Evaluacion Consolidada]
```

## Secuencia de inicializacion (hardening API)

```mermaid
sequenceDiagram
   participant VS as Uvicorn/FastAPI
   participant LF as lifespan(app)
   participant DB as init_database()
   participant API as Endpoints

   VS->>LF: startup
   LF->>DB: intento de inicializacion
   alt DB disponible
      DB-->>LF: ok
   else DB no disponible
      DB-->>LF: exception controlada
      LF-->>LF: continuar sin tumbar API
   end
   LF-->>VS: app lista
   VS->>API: acepta requests
```

## Limites conocidos y reglas de fallback

Limites actuales del MVP 2.1:

- `segment_person` usa `mediapipe_selfie_segmentation` y fallback `fallback_ellipse` si falla backend.
- `run_tryon` aplica composicion por capas cuando `garment_path` es imagen raster (`png/jpg/webp`).
- Si la prenda no es raster (por ejemplo `.obj`) o falla el render, mantiene fallback placeholder (copia de imagen base).
- Fallback activo cuando DB no inicializa en startup: la API sigue operativa.
- El modelo `fit_regression` baseline predice escala de prenda; no predice offset horizontal/vertical (pendiente 2.2).
- El dataset de segmentacion depende de que existan pares imagen-mascara preexistentes; no genera mascaras automaticamente.
- MLflow tracking solo se activa con `MLFLOW_TRACKING_ENABLED=1`; en produccion se debe configurar antes de correr el script.
- La primera corrida de entrenamiento usa 5 muestras sinteticas; para produccion se requiere dataset real de 100+ pares.

### Backlog Fase 2.2

Mejoras identificadas fuera del alcance del MVP 2.1:

| Item | Descripcion |
|---|---|
| Prediccion de offset | Extender `fit_regression` para predecir tambien offset X/Y, no solo escala. |
| Dataset real de segmentacion | Recolectar y etiquetar pares imagen/mascara de personas reales con prendas. |
| Fine-tuning de segmentacion | Usar el dataset preparado por `prepare_segmentation_dataset.py` para ajustar el modelo de segmentacion. |
| Deformacion de tela | Aplicar warp o thin-plate spline sobre la prenda antes de componer, para simular como cae la tela. |
| Inferencia en video | Extender el pipeline para procesar secuencias de frames con consistencia temporal. |
| Evaluacion humana automatizada | Integrar formulario de evaluacion Web con guardado automatico en PostgreSQL/MLflow. |

```mermaid
flowchart TD
   A[Request /tryon/run] --> B{Pose detectada?}
   B -- No --> E[Error controlado: No landmarks]
   B -- Si --> C{Segmentacion disponible?}
   C -- No --> F[Usar fallback_ellipse]
   C -- Si --> G[Usar mascara real]
   F --> H[Computar scale + contracts]
   G --> H
   H --> I[Output baseline placeholder]
```

---

## Criterio técnico de esta fase

En esta fase conviene evitar dos errores:

- Intentar simulación física completa desde el inicio.
- Depender de servicios cerrados o pipelines difíciles de entrenar localmente.

La estrategia correcta para Fase 2 es:

1. Resolver primero un try-on robusto en imagen estática.
2. Introducir IA entrenable en pose, segmentación y ajuste.
3. Añadir oclusión, máscara y composición limpia antes de pensar en tela física realista.
4. Recién después pasar a video y deformación más avanzada.

---

## Enfoque recomendado

### Arquitectura propuesta

```text
Foto de persona
   ↓
Pose estimation
   ↓
Segmentación / parsing de cuerpo
   ↓
Cálculo de escala, rotación y anclajes
   ↓
Warp / alineación de la prenda
   ↓
Composición con oclusión
   ↓
Imagen final
```

### Qué sí hacemos

- Ajuste geométrico de la prenda.
- Estimación de puntos clave del cuerpo.
- Segmentación del cuerpo o de la ropa existente.
- Composición final con máscaras.
- Entrenamiento de componentes propios con datasets abiertos.

### Qué no hacemos todavía

- Simulación física avanzada de tela.
- Reconstrucción completa de cuerpo 3D.
- Cloth dynamics en tiempo real.

---

## Mejora clave sobre el plan inicial

El documento anterior planteaba un overlay demasiado simple.

Eso sirve para un demo técnico, pero no para un try-on mínimamente usable, porque faltaban tres piezas críticas:

1. Segmentación del cuerpo y la ropa.
2. Manejo de oclusiones.
3. Componentes entrenables reales más allá de MediaPipe.

La versión corregida de Fase 2 propone un MVP más serio sin salirnos de herramientas gratuitas.

---

## Stack recomendado con herramientas gratuitas

### IA / visión por computadora

- PyTorch
- OpenCV
- NumPy
- MediaPipe
- torchvision
- albumentations

### Modelos open source entrenables

- RTMPose o MMPose para pose entrenable.
- YOLOv8-seg o Detectron2 para segmentación.
- U2Net o MODNet si se necesita recorte de silueta.
- Segment Anything solo como apoyo, no como núcleo entrenable principal.

### 3D y render

- Blender
- trimesh
- Three.js
- GLTF / GLB

### Backend

- FastAPI

### Entrenamiento

- PyTorch Lightning opcional.
- MLflow ya presente en el proyecto para experiment tracking.

---

## Datasets gratuitos recomendados

### Para ropa y landmarks

- DeepFashion2
  - categorías
  - bounding boxes
  - landmarks
  - masks

### Para parsing corporal o segmentación humana

- LIP
- ATR
- CIHP

### Para robustecer el sistema

- Datos propios generados por ustedes:
  - renders sintéticos desde Blender
  - fotos reales de usuario con poses controladas
  - pares prenda-persona para calibración

---

## Componentes IA que sí conviene entrenar

## 1. Pose del cuerpo

### Baseline gratuito

- MediaPipe Pose

### Mejora entrenable

- MMPose o RTMPose fine-tuned con datos de ropa y poses objetivo.

### Por qué

MediaPipe sirve para arrancar, pero no está pensado para tunear fino a tu dominio. Si el objetivo es usar IA y entrenamiento real, conviene mover el pose a un stack entrenable.

---

## 2. Segmentación de persona y oclusión

### Recomendado

- YOLOv8-seg o Detectron2 entrenado o fine-tuned con máscaras de ropa y cuerpo.

### Por qué

Sin segmentación, el overlay se ve falso. La oclusión correcta es lo que hace que una manga, cabello o brazo tapen parte de la prenda cuando corresponde.

---

## 3. Alineación y warp de la prenda

### MVP

- ajuste por hombros, torso, cadera y bounding boxes.

### Mejora entrenable

- red pequeña de regresión para parámetros de transformación:
  - escala
  - traslación
  - rotación
  - deformación simple

### Objetivo

No aprender una prenda nueva, sino aprender a colocar mejor una prenda existente.

---

## 4. Composición final

### MVP

- alpha blending con máscara.

### Mejora

- orden de capas:
  - fondo
  - cuerpo
  - prenda virtual
  - regiones ocluyentes como brazos, cabello o ropa original

---

## Pipeline recomendado de Fase 2

## MVP usable

1. Usuario sube foto.
2. Detectar pose.
3. Segmentar cuerpo.
4. Obtener anclas: hombros, torso, cadera.
5. Escalar y posicionar la prenda.
6. Aplicar warp simple.
7. Componer con máscara y oclusión.
8. Devolver imagen final.

## Versión mejorada

1. Pose entrenable.
2. Segmentación entrenada.
3. Estimación de parámetros de ajuste con red propia.
4. Refinamiento visual final.

---

## Estructura sugerida

```text
core/
├── tryon/
│   ├── pose.py
│   ├── parsing.py
│   ├── align.py
│   ├── warp.py
│   ├── occlusion.py
│   ├── overlay.py
│   ├── pipeline.py
│   └── schemas.py
├── training/
│   ├── datasets/
│   ├── pose/
│   ├── segmentation/
│   └── fit_regression/
```

Esto separa inferencia de entrenamiento, que en la versión anterior estaba demasiado básica.

---

## Ejemplo mínimo de pose

Archivo: `core/tryon/pose.py`

```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


def detect_pose(image_path: str):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

    return results.pose_landmarks
```

Esto sirve como baseline, no como solución final entrenable.

---

## Ejemplo mejorado de escala

Archivo: `core/tryon/align.py`

```python
def compute_scale(landmarks):
    left_shoulder = landmarks.landmark[11]
    right_shoulder = landmarks.landmark[12]
    left_hip = landmarks.landmark[23]
    right_hip = landmarks.landmark[24]

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    torso_width = abs(left_hip.x - right_hip.x)

    return max(shoulder_width, torso_width) * 2.2
```

Mejora respecto a la versión anterior porque no depende solo de hombros.

---

## Ejemplo de overlay utilizable

Archivo: `core/tryon/overlay.py`

```python
import cv2
import numpy as np


def alpha_overlay(background, overlay_rgba, x, y):
    h, w = overlay_rgba.shape[:2]
    roi = background[y:y+h, x:x+w]

    rgb = overlay_rgba[:, :, :3]
    alpha = overlay_rgba[:, :, 3:4] / 255.0

    blended = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    background[y:y+h, x:x+w] = blended
    return background
```

Esto es mejor que copiar píxeles sin máscara.

---

## Integración con Three.js

### Qué debe devolver backend

- landmarks
- bounding box corporal
- escala
- rotación
- posición
- máscara o regiones de oclusión si ya están disponibles

### Qué hace el frontend

- carga modelo GLB
- ajusta transformación
- renderiza preview
- opcionalmente mezcla canvas del render con imagen original

Ejemplo:

```javascript
const loader = new THREE.GLTFLoader();

loader.load('model.glb', function (gltf) {
  const model = gltf.scene;
  model.scale.set(scale, scale, scale);
  model.position.set(x, y, z);
  model.rotation.set(rx, ry, rz);
  scene.add(model);
});
```

---

## Plan de entrenamiento realista

## Etapa A - Baseline sin entrenamiento fuerte

- MediaPipe para pose.
- Segmentación con modelo abierto preentrenado.
- Reglas geométricas para escala y posición.

Objetivo: validar pipeline visual.

## Etapa B - Entrenamiento útil

- Fine-tuning de segmentación de cuerpo y ropa.
- Fine-tuning o reemplazo de pose por MMPose o RTMPose.
- Dataset interno con ejemplos de colocación correcta.

Objetivo: mejorar precisión.

## Etapa C - Ajuste aprendido

- Entrenar un modelo pequeño para predecir parámetros de transformación.

Input sugerido:

- landmarks
- máscara corporal
- categoría de prenda

Output sugerido:

- escala
- desplazamiento x/y
- rotación
- deformación ligera

---

## Video y semi-tiempo real

Esto sí es viable, pero no como primer entregable.

Orden correcto:

1. Imagen estática robusta.
2. Lote de imágenes.
3. Video por frames.
4. Optimización de inferencia.

Pipeline futuro:

```text
Video
   ↓
Frames
   ↓
Pose por frame
   ↓
Tracking temporal
   ↓
Actualizar prenda
   ↓
Render
```

La pieza extra aquí es tracking temporal para evitar jitter.

---

## Limitaciones reales de Fase 2

- No habrá simulación física de tela realista todavía.
- Las poses extremas van a degradar el resultado.
- La calidad depende mucho de segmentación y landmarks.
- La prenda no va a caer físicamente sobre el cuerpo.

Eso es aceptable para esta fase si el objetivo es demostrar un try-on funcional con IA.

---

## Buenas prácticas

- Usar GLB como formato frontend.
- Normalizar escala y orientación de prendas en Blender.
- Mantener texturas entre 1K y 2K para evitar sobrecarga.
- Entrenar primero con una sola categoría, por ejemplo tops o camisas.
- Medir todo con ejemplos propios, no solo con dataset público.

---

## Recomendación concreta para este proyecto

Si el objetivo es mantenernos en gratis + IA + entrenamiento, la mejor ruta es:

1. Baseline con MediaPipe + segmentación open source.
2. Integrar DeepFashion2 solo como fuente de labels 2D, no como fuente de modelos 3D.
3. Entrenar segmentación de ropa y persona con dataset abierto y datos propios.
4. Entrenar un módulo pequeño de ajuste geométrico.
5. Dejar simulación física para Fase 3 o posterior.

Ese camino sí es viable en este repositorio y sí aprovecha IA entrenable de verdad.

---

## Resultado esperado de Fase 2

```text
Usuario sube foto
Selecciona una prenda
Sistema detecta pose y cuerpo
Sistema ajusta prenda
Sistema compone la imagen final
Resultado visual funcional y repetible
```

---

## Conclusión

- Es viable hacerlo con herramientas gratuitas.
- Sí se puede incorporar IA entrenable real en esta fase.
- La clave no es una simulación compleja, sino pose + segmentación + alineación + composición.
- El MVP debe apuntar a calidad visual consistente, no a realismo físico total.

---

## FASE 2.1 - Inicio inmediato (MVP ejecutable)

Objetivo de 2.1: dejar una primera versión funcional de try-on en imagen estática, medible y repetible.

### Alcance de 2.1

Incluye:

1. Endpoint de inferencia para try-on en imagen.
2. Pipeline base con:
   - pose
   - segmentación
   - alineación geométrica
   - overlay con alpha
3. Estructura de entrenamiento preparada para:
   - segmentación
   - ajuste geométrico
4. Métricas iniciales y registro de experimentos.

No incluye todavía:

1. Video tiempo real de producción.
2. Simulación física de tela.
3. Deformación compleja de malla con física.

### Entregables de 2.1

1. API try-on para imagen estática.
2. Primer módulo core/tryon integrado al proyecto.
3. Plan de entrenamiento incremental con datasets gratuitos.
4. Checklist de validación visual con casos controlados.

### KPI iniciales de 2.1

1. Tiempo por imagen menor o igual a 2.5 segundos en modo CPU baseline.
2. Tasa de overlays válidos mayor o igual a 90 por ciento en dataset de prueba interna.
3. Reducción visible de errores de alineación frente al baseline simple.