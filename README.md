# 🧵 Fabric2Mesh AI (Local + AMD + Docker Hybrid)

Sistema de IA para generar texturas de telas desde pocas imágenes (2–10) y aplicarlas automáticamente a modelos 3D existentes.

---

## ✅ Objetivo real del proyecto (alineado a negocio)

Flujo deseado:

* Tomar fotos de telas reales
* Generar un patron/textura digital equivalente
* Transferir ese patron a modelos 3D de prendas
* Prendas objetivo principales: **camisa**, **falda**, **pantalon**

El sistema ya prioriza este flujo en API/UI y permite trabajar con 1 imagen por caso para pruebas rapidas.

---

## 🎯 Objetivo del sistema

Permitir:

* 📸 Aprender una tela desde 2–10 fotos
* 🧠 Generar una textura limpia y tileable usando IA local
* 👕 Aplicar la textura a un modelo 3D existente
* ⚡ Ejecutar procesamiento pesado en GPU AMD (host)
* 🐳 Usar Docker solo para servicios auxiliares

---

## 🧠 Enfoque técnico

❗ No usamos IA 3D pura

✔ Usamos enfoque híbrido:

```text
Fotos → IA 2D (LoRA) → Textura tileable → Blender → Modelo 3D
```

---

## ⚙️ Arquitectura híbrida

```text
                ┌──────────────────────┐
                │      Usuario         │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │     FastAPI (API)    │  ← Docker
                └─────────┬────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                                   ▼
┌───────────────┐                 ┌──────────────────┐
│ Preprocessing │                 │  Texture Engine  │
│ (OpenCV)      │                 │ (PyTorch ROCm)   │ ← HOST GPU
└───────────────┘                 └──────────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │   Blender Engine │ ← HOST
                                  └──────────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │ Export GLB/OBJ   │
                                  └──────────────────┘
```

---

## ⚙️ Requisitos

### Hardware

* GPU AMD (compatible con ROCm)
* 16–32 GB RAM

### OS

* Ubuntu 22.04+

---

## 🔥 Configuración GPU AMD (HOST)

### Instalar ROCm

```bash
sudo apt update
sudo apt install rocm-dev rocm-libs
```

Verificar:

```bash
rocminfo
```

---

### Instalar PyTorch ROCm

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

---

## 🧱 Stack Tecnológico

### 🧠 IA local (HOST)

| Herramienta | Para que se usa en el proyecto |
|---|---|
| torch (ROCm) | Framework base de deep learning para entrenamiento e inferencia en GPU AMD con ROCm. |
| diffusers | Pipelines de modelos de difusion para generar texturas e imagenes a partir de prompts o referencias. |
| transformers | Soporte para modelos tipo Transformer (texto/vision) usados dentro de pipelines modernos de generacion. |
| accelerate | Optimiza y simplifica ejecucion en GPU/CPU (memoria, precision, batch, distribucion). |
| peft (LoRA) | Fine-tuning eficiente con LoRA para adaptar modelos grandes al dominio de telas sin reentrenar todo. |
| safetensors | Formato seguro y rapido para cargar/guardar pesos de modelos sin usar serializacion insegura. |

---

### 🧵 Procesamiento de imagen

| Herramienta | Para que se usa en el proyecto |
|---|---|
| opencv-python | Limpieza y preprocesamiento de imagen: lectura/escritura, conversiones de color, filtros, mascaras y ajustes geometricos antes de generar texturas. |
| pillow | Manipulacion simple de imagenes para cargar, guardar, redimensionar y componer previews o exportaciones intermedias. |
| numpy | Base numerica para representar imagenes como arreglos y aplicar operaciones rapidas sobre pixeles, canales y mascaras. |
| scikit-image | Utilidades de procesamiento mas especializadas, como metricas visuales, filtros adicionales, segmentacion y transformaciones complementarias. |
| albumentations | Pipeline de aumentacion de datos para entrenamiento y pruebas, con variaciones controladas de iluminacion, rotacion, recorte, ruido y deformacion. |

---

### 🧊 3D / Render

* Blender (Python API)
* trimesh
* xatlas

---

### 🌐 Backend (Docker)

* fastapi
* uvicorn
* pydantic

---

### 🧠 Tracking

* mlflow (Docker)

---

### 💾 Almacenamiento

* PostgreSQL (Docker)
* MinIO (Docker)

---

### 🌐 Frontend

* three.js

---

## 🐳 Uso de Docker (solo servicios)

❗ IMPORTANTE
Docker NO se usa para IA pesada.

Se usa para:

* API
* Base de datos
* Storage
* MLflow

---

## 🐳 docker-compose.yml (base)

```yaml
version: "3.9"

services:

  api:
    build: ./apps/api
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    depends_on:
      - db
      - minio

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-almapa}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-asdfghjkl}
      POSTGRES_DB: ${POSTGRES_DB:-fabricdb}

  minio:
    image: minio/minio
    command: server /data
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: password
    ports:
      - "9000:9000"

  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports:
      - "5000:5000"
```

---

## 📁 Estructura del proyecto

```text
fabric2mesh-ai/
│
├── apps/
│   ├── api/
│   └── worker/
│
├── core/
│   ├── preprocessing/
│   ├── texture/
│   ├── projection/
│   └── rendering/
│
├── models/
│   └── lora_fabrics/
│
├── scripts/
│
├── data/
│
└── docker/
```

---

## 🔄 Pipeline real

```text
1. Subir 2–10 fotos
2. Limpieza (OpenCV)
3. Generación de textura (Diffusion + LoRA)
4. Crear textura tileable
5. Aplicar en Blender
6. Exportar GLB
```

---

## 🧠 Entrenamiento IA (LoRA)

### Cuándo usar LoRA

* Tela única → LoRA específica
* Tipo de tela → LoRA general

---

### Script entrenamiento

```bash
python scripts/train_lora.py --dataset data/fabrics/denim
```

---

### Generar textura

```bash
python scripts/generate_texture.py --input fotos/
```

---

## 🎨 Blender automático

Script:

```bash
blender -b -P scripts/apply_texture.py -- model.obj textura.png
```

---

## 🧪 Fases del proyecto

### Fase 1 — MVP

* Aplicar textura directa
* Exportar modelo

---

### Fase 2 — IA

* LoRA entrenamiento
* texturas seamless

---

### Fase 3 — mejora

* selección automática
* ranking de texturas

---

### Fase 4 — avanzada

* control por máscaras
* mejor UV mapping

---

## ⚠️ Problemas reales

### Distorsión

→ UV mapping incorrecto

### Textura falsa

→ falta de mapas PBR

### Repetición visible

→ no seamless

---

## 🔥 Buenas prácticas

* No entrenar modelos 3D
* Separar IA 2D de 3D
* usar meshes base
* controlar escala de patrón

---

## 🚀 Flujo ideal

```bash
# 1. generar textura
python scripts/generate_texture.py

# 2. aplicar a modelo
python scripts/apply_texture.py

# 3. exportar
python scripts/export_glb.py
```

---

## 🧠 Conclusión

✔ Con 2–10 fotos es suficiente
✔ IA local mejora calidad
✔ Blender hace el trabajo 3D
✔ Docker solo para servicios

---

## 📄 Licencia

MIT

---

## ✅ Estado actual del repositorio

### Actualizacion reciente (2026-04-18)

Novedades implementadas y validadas:

* Integracion de modelos DeepFashion desde dos fuentes:
  * `data_deepfashon/point_cloud/...`
  * `data_deepfasho_antiguo/pointcloud/...`
* Etiquetado automatico por tipo de prenda (`shirt`, `dress`, `pants`) usando `cloth_type_list.txt`
* Deteccion de `point cloud` en API/UI con metadatos (`source`, `is_point_cloud`, `garment_type`)
* Conversion on-demand de nubes de puntos (`.ply`) a malla proxy para operaciones backend:
  * `POST /projects/preview-3d`
  * `POST /projects/select`
* Cache de conversion en `data/processed/pointcloud_meshes/`
* En scripts Blender se intenta UV automatico (`smart_project`) cuando la malla no tiene UV

Notas:

* `deepfashion2/` es un dataset 2D (imagenes + anotaciones), no una fuente de modelos 3D para texturizar.
* La conversion de point cloud a malla es un proxy rapido para pruebas; puede perder detalle fino respecto a la nube original.

Implementado en esta iteracion:

* Estructura base de carpetas (`apps`, `core`, `scripts`, `models`, `data`, `docker`)
* API FastAPI minima con endpoints:
  * `GET /health`
  * `POST /projects/generate`
  * `POST /projects/select`
  * `POST /projects/feedback`
  * `GET /projects/{project_id}/feedback-summary`
  * `GET /projects/{project_id}/ranking`
  * `GET /projects/{project_id}/metrics`
  * `GET /projects/{project_id}/candidates`
  * `POST /projects/{project_id}/preview-sheet`
  * `POST /projects/preview-3d`
  * `GET /assets/catalog`
  * `GET /assets/model-search`
  * `POST /assets/import-from-downloads`
  * `POST /assets/upload-fabric`
* Preproceso funcional: valida, normaliza y exporta referencias a `data/processed/<project>/preprocessed`
* Generador funcional baseline: crea N texturas PNG candidatas en `data/processed/<project>/textures`
* Exportador con Blender real (`blender -b`) cuando esta disponible y fallback de compatibilidad cuando no
* Feedback persistido en PostgreSQL (con fallback `jsonl` solo si DB no esta disponible)
* Validacion de contrato de modelo 3D en `POST /projects/select` (soporta OBJ/GLB/GLTF/FBX y chequeo base de UV)
* Soporte de conversion de point clouds a malla proxy en demanda para preview/export
* Catalogo de modelos clasificado por tipo de prenda (`shirt`, `skirt`, `pants`, `other`)
* Busqueda de modelos con autocomplete y metadatos por fuente/tipo
* Ranking automatico de candidatos basado en feedback historico
* Metricas de proyecto (tasa de aprobacion y top candidato)
* Metricas de performance por lote (tiempo promedio, ultimo tiempo y reprocesos)
* Tracking opcional en MLflow para generacion, seleccion y feedback
* Postproceso seamless configurable para texturas candidatas
* `docker-compose.yml` con API + Postgres + MinIO + MLflow
* Scripts base:
  * `scripts/generate_texture.py`
  * `scripts/apply_texture.py`
  * `scripts/export_glb.py`
  * `scripts/train_lora.py`
  * `scripts/blender_apply_texture.py`

Nota: aun no se conecta diffusion+LoRA real. El generador actual es baseline deterministico para cerrar flujo de punta a punta.

Actualizacion: ya existe modo `diffusion` opcional (best effort) con fallback automatico a baseline si faltan dependencias o modelo.

## 🚀 Quickstart (Fase 0)

### 1) Levantar servicios

```bash
cp .env.example .env
docker compose up --build -d
```

### 2) Interfaz web (recomendado)

Abrir en navegador:

```
http://localhost:8000/
```

**Flujo típico desde la UI:**

1. Click en **Import From Downloads** para cargar assets desde tu carpeta binaria
2. En **Asset Browser**, selecciona:
   - Una tela (ejm: `gris.webp`)
   - Un modelo (ejm: `TShirts.obj` o `TShirts.FBX`)
   - Al seleccionar modelo, ves el **estado UV inmediatamente**
3. Presiona **Generate Candidates** para crear 4-8 variantes
4. En **Candidates**, click **Select** en una variante
5. Presiona **Build 3D Preview** para ver la prenda en visor **3D interactivo** (rotar/zoom/pan)
6. Presiona **Export Selected** para guardar el modelo 3D con la textura aplicada
7. (Opcional) **Approve/Reject** para feedback y reranking automático

### 3) API CLI (para automatización)

```bash
# Health check
curl http://localhost:8000/health

# Validar modelo (ver UV status INMEDIATO)
curl 'http://localhost:8000/projects/validate-model?model_path=data/raw/models/TShirts.obj'

# Generar candidatos
curl -X POST http://localhost:8000/projects/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "demo-001",
    "image_paths": ["data/raw/telas/gris.webp"],
    "n_candidates": 8
  }'

# Preview 2D (sheet de candidatos)
curl -X POST http://localhost:8000/projects/demo-001/preview-sheet

# Preview 3D server
curl -X POST http://localhost:8000/projects/preview-3d \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "demo-001",
    "model_path": "data/raw/models/TShirts.obj",
    "texture_path": "data/processed/demo-001/textures/candidate_01.png",
    "convert_point_cloud": true
  }'

# Export y aplicar textura
curl -X POST http://localhost:8000/projects/select \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "demo-001",
    "model_path": "data/raw/models/TShirts.obj",
    "selected_texture_path": "data/processed/demo-001/textures/candidate_01.png",
    "output_path": "data/exports/demo-001.glb",
    "convert_point_cloud": true
  }'

# Feedback (aprobar/rechazar)
curl -X POST http://localhost:8000/projects/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "demo-001",
    "candidate_path": "data/processed/demo-001/textures/candidate_01.png",
    "label": "approve",
    "score": 5
  }'

# Ranking y métricas
curl http://localhost:8000/projects/demo-001/ranking?limit=10
curl http://localhost:8000/projects/demo-001/metrics
```

**Nota técnica: Validación de UV**

* `GET /projects/validate-model?model_path=...` devuelve estado UV instant sin necesidad de preview/export
* Estados posibles: `missing` (sin UV), `ok` (con UV), `unknown` (no detectable)
* La UI actualiza el estado al instante cuando seleccionas un modelo

**Nota técnica: Point Clouds en backend**

* Si envias un `.ply` a `preview-3d` o `select`, el backend puede convertirlo a malla proxy en el momento con `"convert_point_cloud": true`.
* La respuesta incluye rutas y metadata de conversion: `input_model_path`, `render_model_path`, `point_cloud_conversion`.

## ✅ Validacion operativa reciente (2026-04-18)

### Test integral completo ejecutado: `case-fullflow-20260418`

**Flujo probado:**

1. ✅ Validación de modelo OBJ: `uv_status: missing` (detectado inmediatamente)
2. ✅ Validación de modelo FBX: `uv_status: unknown` (detectado inmediatamente)
3. ✅ Generación de 4 candidatos desde tela `gris.webp`
4. ✅ Preview 2D: mosaico de candidatos generado
5. ✅ Preview 3D server: imagen renderizada con Blender
6. ✅ Export OBJ: modelo + textura → GLB
7. ✅ Export FBX: modelo + textura → GLB
8. ✅ Feedback: aprobación registrada en PostgreSQL
9. ✅ Reranking: automático basado en feedback

**Características validadas en esta sesión:**

- [x] Visor 3D interactivo en navegador (Three.js + OrbitControls)
  - Carga modelos OBJ/FBX/GLTF
  - Aplica textura seleccionada
  - Permite rotación/zoom/pan
  
- [x] Estado UV del modelo visible al instante
  - Endpoint `GET /projects/validate-model` agregado
  - UI actualiza estado al seleccionar modelo
  - Colores codificados: verde (ok), amarillo (unknown), rojo (missing)
  
- [x] Deduplicación de assets
  - Telas: 19 → 11 (removidas Blender map duplicates)
  - Modelos: 32 → 8 (removidas numbered variants _1/_2/_3)
  - Asset Browser mucho más limpio
  
- [x] Pipeline end-to-end robusto
  - Generación + Preview 2D/3D + Export en ambos formatos
  - Feedback y reranking operativo
  - Métricas y tracking funcionales

### Casos anteriormente probados:

- OBJ test (`case-continue-20260418`): todos los pasos exitosos
- FBX test (`case-fbx-20260418`): preview y export OK, UV status unknown

### Deduplicacion ejecutada (2026-04-18 continuacion):

* Fabric images: reducidas de 19 a 11 (eliminadas 8 Blender map duplicates)
* Models: reducidas de 32 a 8 (eliminadas 24 numbered variants _1, _2, _3)
* Resultado: catalogo mucho más limpio en Asset Browser


## 🧥 Modelos 3D objetivo cargados desde Downloads

Cargados al proyecto en `data/raw/models`:

* `uploads_files_2105208_TShirts_OBJ.zip` (extraido, incluye `TShirts.obj`)
* `uploads_files_2105208_T_Shirts_FBX.zip` (extraido, incluye `TShirts.FBX`)
* `uploads_files_6483441_Basic+bra+top+CT25-151.fbx`
* `uploads_files_6490317_Basic+Raglan+T-shirt+short+sleeve+FormX+BT25-29.fbx`
* `uploads_files_6490317_Basic+Raglan+T-shirt+short+sleeve+FormX+BT25-29.obj`

Nota:

* Parte de tus modelos en Downloads vienen en `.rar`; para incorporarlos automaticamente (por ejemplo opciones de falda/pantalon), primero hay que extraerlos al filesystem y luego apareceran en `/assets/catalog`.

## 📥 Importacion automatica desde Downloads

El servicio API monta `~/Downloads` como lectura en `/mnt/downloads` y permite importar con:

```bash
curl -X POST http://localhost:8000/assets/import-from-downloads \
  -H "Content-Type: application/json" \
  -d '{}'
```

Comportamiento:

* Copia modelos soportados directos (`.obj`, `.fbx`, `.glb`, `.gltf`) a `data/raw/models`
* Copia imagenes soportadas (`.jpg`, `.jpeg`, `.png`, `.webp`) a `data/raw/telas`
* Extrae `.zip` automaticamente
* Intenta extraer `.rar` si `unrar` esta disponible en entorno

Resultado actual con tus archivos:

* Camisas/T-Shirts ya detectadas y operativas
* Faldas/pantalones dependen de extraer paquetes `.rar` que aun no se pueden abrir sin herramienta adicional

### 2) Probar salud de API

```bash
curl http://localhost:8000/health
```

Respuesta esperada:

```json
{"status":"ok","database":"ok"}
```

### 3) Probar generacion de candidatos (API)

```bash
curl -X POST http://localhost:8000/projects/generate \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "demo-001",
    "image_paths": ["data/raw/sample1.jpg", "data/raw/sample2.jpg"],
    "n_candidates": 8
  }'
```

### 4) Probar scripts CLI (placeholder)

```bash
python scripts/generate_texture.py --project-id demo-001 --input data/raw/sample1.jpg data/raw/sample2.jpg --n 8
python scripts/apply_texture.py --model data/raw/model.obj --texture data/processed/demo-001/textures/candidate_01.png --output data/exports/demo-001.glb
```

### 5) Guardar feedback humano (APROBAR/RECHAZAR)

```bash
curl -X POST http://localhost:8000/projects/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "demo-001",
    "candidate_path": "data/processed/demo-001/textures/candidate_01.png",
    "label": "approve",
    "score": 5,
    "comment": "se apega al patron"
  }'
```

```bash
curl http://localhost:8000/projects/demo-001/feedback-summary
```

### 6) Listar candidatos actuales de un proyecto

```bash
curl http://localhost:8000/projects/demo-001/candidates
```

### 6.1) Obtener ranking automatico por feedback acumulado

```bash
curl "http://localhost:8000/projects/demo-001/ranking?limit=10"
```

### 6.2) Consultar metricas operativas del proyecto

```bash
curl "http://localhost:8000/projects/demo-001/metrics"
```

### 7) Aplicar textura seleccionada a modelo 3D

```bash
curl -X POST http://localhost:8000/projects/select \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "demo-001",
    "model_path": "data/raw/model.obj",
    "selected_texture_path": "data/processed/demo-001/textures/candidate_01.png",
    "output_path": "data/exports/demo-001.glb"
  }'
```

La respuesta incluye `model_report` con formato detectado y estado UV.

### 8) Generar hoja de preview 2D de candidatos

```bash
curl -X POST http://localhost:8000/projects/demo-001/preview-sheet
```

### 8.1) Generar preview 3D de un candidato

```bash
curl -X POST http://localhost:8000/projects/preview-3d \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "demo-001",
    "model_path": "data/raw/model.obj",
    "texture_path": "data/processed/demo-001/textures/candidate_01.png",
    "output_path": "data/processed/demo-001/preview/preview_01.png"
  }'
```

Si Blender no esta disponible, se genera fallback de preview basado en textura para no bloquear el flujo.

### 9) Ejecutar prueba basica de API

```bash
pip install -r apps/api/requirements.txt -r requirements-dev.txt
pytest -q
```

### 10) Habilitar motor diffusion opcional

Instalar dependencias IA del host:

```bash
pip install -r requirements-ai.txt
```

Activar en `.env`:

```bash
TEXTURE_ENGINE=diffusion
```

Si diffusion falla por entorno/modelo, el sistema vuelve automaticamente al modo baseline.

Control seamless:

* `TEXTURE_SEAMLESS_ENABLED=1` aplica offset/blend para hacer texturas mas tileables

Control LoRA opcional:

* `LORA_WEIGHTS_PATH=/ruta/a/lora` para cargar pesos LoRA al pipeline
* `LORA_SCALE=0.8` para ajustar influencia LoRA

## 📈 Tracking MLflow (opcional)

Por defecto esta deshabilitado:

* `MLFLOW_TRACKING_ENABLED=0`

Para activarlo:

* cambiar `MLFLOW_TRACKING_ENABLED=1` en `.env`
* mantener `MLFLOW_TRACKING_URI=http://mlflow:5000`

## 🗃️ Configuracion actual de PostgreSQL local

Variables activas (archivo `.env`):

* `POSTGRES_USER=almapa`
* `POSTGRES_PASSWORD=asdfghjkl`
* `POSTGRES_DB=fabricdb`
* `POSTGRES_PORT=5432`

## 🧩 Proximos pasos tecnicos (inmediatos)

* Reemplazar `core/texture/engine.py` con inferencia diffusion + LoRA real
* Agregar preview 3D rapido para comparar candidatos
