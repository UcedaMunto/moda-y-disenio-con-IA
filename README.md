# Fabric2Mesh AI

Repositorio para generacion de texturas, aplicacion en modelos 3D y prueba virtual sobre fotos.

## Estado Actual

- Editor web principal: GET /
- Flujo de texturas y 3D: operativo
- Flujo Fase 2.1 (foto): operativo
- Flujo Fase 2.2 (pipeline v2 para try-on): operativo
- Guardado de looks y prueba virtual en fotos_personas: operativo

## Objetivo del Proyecto

Flujo principal:

1. Cargar o importar telas
2. Generar candidatos de textura
3. Visualizar preview 2D y 3D
4. Exportar modelo final
5. Guardar look
6. Aplicar look en fotos de personas

## Arquitectura Resumida

- Backend: FastAPI en apps/api/main.py
- Motor textura: core/texture/*
- Render/export 3D: core/rendering/*
- Catalogo y assets: core/assets/*
- Fase 2.1 try-on: fase2_1/core/tryon/*
- Fase 2.2 try-on v2: fase2_2/core/tryon/*
- UI: apps/api/ui/index.html y apps/api/ui/tryon.html

## Nueva Linea: Fase 2.3 (SAM 3D Body)

Objetivo inmediato de mejora del try-on:

1. Tomar una foto de persona.
2. Reconstruir una malla 3D humana con la pose de la foto usando `sam-3d-body/`.
3. Convertir esa salida a un "maniqui posed" util para vestir la prenda diseñada.
4. Usar ese maniqui en el pipeline de prenda (ajuste/proyeccion/render) para mejorar resultado frente a overlay 2D.

Documentacion de esta linea:

- [docs/fase2_3/PLAN-SAM3D-BODY.md](docs/fase2_3/PLAN-SAM3D-BODY.md)
- [docs/fase2_3/INTEGRACION-SAM3D-BODY-AMD.md](docs/fase2_3/INTEGRACION-SAM3D-BODY-AMD.md)

Scripts iniciales Fase 2.3:

- `scripts/run_sam3d_body_smoke.py` (smoke demo)
- `scripts/run_sam3d_body_single.py` (inferencia por imagen + export de malla)

## Ejecucion Local

### 1) Activar entorno

```bash
source .venv/bin/activate
```

### 2) Levantar API

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3) Abrir interfaz

- Editor: http://localhost:8000/
- Prueba virtual: http://localhost:8000/tryon

## Documentacion de APIs

Base URL local: http://localhost:8000

### Salud y UI

| Metodo | Endpoint | Descripcion |
|---|---|---|
| GET | / | Sirve la interfaz principal index.html |
| GET | /tryon | Sirve la interfaz de prueba virtual tryon.html |
| GET | /health | Estado de API y disponibilidad de base de datos |

### Catalogo de Assets

| Metodo | Endpoint | Descripcion |
|---|---|---|
| GET | /assets/catalog | Lista telas, modelos y estadisticas de indexado |
| GET | /assets/model-search | Busqueda de modelos por texto con limite |
| POST | /assets/import-from-downloads | Importa assets desde carpeta de descargas |
| POST | /assets/upload-fabric | Sube una tela en data_url base64 |
| GET | /assets/deepfashion-coverage | Reporte de cobertura entre datasets DeepFashion |

### Proyectos de Textura y 3D

| Metodo | Endpoint | Descripcion |
|---|---|---|
| GET | /projects/validate-model | Valida contrato de modelo 3D |
| POST | /projects/generate | Genera candidatos de textura por proyecto |
| POST | /projects/select | Selecciona textura y exporta modelo |
| POST | /projects/preview-3d | Genera preview 3D en imagen |
| POST | /projects/{project_id}/preview-sheet | Genera hoja 2D de candidatos |
| GET | /projects/{project_id}/candidates | Lista candidatos guardados del proyecto |
| POST | /projects/feedback | Registra feedback approve/reject |
| GET | /projects/{project_id}/feedback-summary | Resumen de feedback |
| GET | /projects/{project_id}/ranking | Ranking por feedback |
| GET | /projects/{project_id}/metrics | Metricas agregadas del proyecto |

### Fase 2.1 Try-On

| Metodo | Endpoint | Descripcion |
|---|---|---|
| POST | /fase2_1/tryon/run | Ejecuta try-on de una imagen |
| POST | /fase2_1/tryon/batch | Ejecuta evaluacion batch |
| POST | /fase2_1/tryon/evaluate | Guarda evaluacion manual por imagen |
| GET | /fase2_1/tryon/evaluate-summary | Resume evaluaciones manuales |
| GET | /fase2_1/tryon/evaluate-consolidated | Consolida batch + evaluacion manual |

### Fase 2.3 Body Reconstruction

| Metodo | Endpoint | Descripcion |
|---|---|---|
| GET | /fase2_3/body/preflight | Verifica entorno/config SAM 3D Body (root, checkpoints, script) |
| POST | /fase2_3/body/reconstruct | Reconstruye cuerpo 3D desde foto (SAM 3D Body) y guarda artefactos |

### Looks y Prueba Virtual

| Metodo | Endpoint | Descripcion |
|---|---|---|
| POST | /looks/save | Guarda look (modelo + textura + metadatos) |
| GET | /looks | Lista looks guardados |
| DELETE | /looks/{look_id} | Elimina look |
| GET | /fotos-personas | Lista fotos disponibles de fotos_personas |
| POST | /tryon/apply | Aplica look sobre foto usando pipeline_v2 |

## Contratos Principales de Request

### POST /projects/generate

```json
{
  "project_id": "case-demo",
  "image_paths": ["data/raw/telas/gris.webp"],
  "n_candidates": 6
}
```

### POST /projects/select

```json
{
  "project_id": "case-demo",
  "model_path": "data/raw/models/TShirts.obj",
  "selected_texture_path": "data/processed/textures/case-demo/candidate_01.png",
  "output_path": "data/exports/case-demo.glb",
  "convert_point_cloud": true
}
```

### POST /projects/preview-3d

```json
{
  "project_id": "case-demo",
  "model_path": "data/raw/models/TShirts.obj",
  "texture_path": "data/processed/textures/case-demo/candidate_01.png",
  "output_path": "data/processed/preview/preview_3d.png",
  "convert_point_cloud": true
}
```

### POST /projects/feedback

```json
{
  "project_id": "case-demo",
  "candidate_path": "data/processed/textures/case-demo/candidate_01.png",
  "label": "approve",
  "score": 5,
  "comment": "textura aprobada"
}
```

### POST /assets/upload-fabric

```json
{
  "name": "tela-rayas-azul",
  "data_url": "data:image/png;base64,iVBOR..."
}
```

### POST /looks/save

```json
{
  "name": "Look Verano 01",
  "model_path": "data/raw/models/TShirts.obj",
  "texture_path": "data/processed/textures/case-demo/candidate_01.png",
  "garment_type": "shirt",
  "project_id": "case-demo",
  "notes": "look para pruebas internas"
}
```

### POST /tryon/apply

```json
{
  "look_id": "a1b2c3d4e5f6",
  "foto_nombre": "woman-in-a-dress-full-body-1615891441oP5.jpg"
}
```

### POST /fase2_3/body/reconstruct

```json
{
  "image_path": "fotos_personas/2f7b90fbaaa9476253d6d993e6ddf487.jpg",
  "case_id": "case-fase2_3-0001",
  "checkpoint_path": "sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt",
  "mhr_path": "sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
  "sam3d_root": "sam-3d-body",
  "sam3d_python_bin": "python",
  "detector_name": "",
  "bbox_thresh": 0.8,
  "use_mask": false,
  "export_glb": true,
  "use_mock": false
}
```

Notas:
- `checkpoint_path` y `mhr_path` pueden omitirse si defines `SAM3D_CHECKPOINT_PATH` y `SAM3D_MHR_PATH` en entorno.
- Usa `GET /fase2_3/body/preflight` para validar configuración antes de correr inferencia real.

## Storage y Rutas Estaticas

- /artifacts -> carpeta data
- /ui-static -> carpeta apps/api/ui
- /deepfashion-artifacts -> carpeta data_deepfashon (si existe)
- /deepfashion-antiguo-artifacts -> carpeta data_deepfasho_antiguo (si existe)
- /fotos-personas-static -> carpeta fotos_personas (si existe)

## Consola de Trabajo Recomendada

```bash
source .venv/bin/activate
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Luego abrir:

- http://localhost:8000/
- http://localhost:8000/tryon

## Indice de Documentacion en docs

Todos los archivos Markdown secundarios fueron movidos a la carpeta docs.

- [docs/ANALYSIS_410_MODEL.md](docs/ANALYSIS_410_MODEL.md)
- [docs/ANALYSIS_POINT_CLOUDS_TEXTURE.md](docs/ANALYSIS_POINT_CLOUDS_TEXTURE.md)
- [docs/BACKLOG_TECNICO.md](docs/BACKLOG_TECNICO.md)
- [docs/INFORME_DE_DUDAS.md](docs/INFORME_DE_DUDAS.md)
- [docs/PLAN_DE_TRABAJO.md](docs/PLAN_DE_TRABAJO.md)
- [docs/README-FASE2.md](docs/README-FASE2.md)
- [docs/SESION_RESUMEN.md](docs/SESION_RESUMEN.md)
- [docs/apps/worker/README.md](docs/apps/worker/README.md)
- [docs/fase2_1/README.md](docs/fase2_1/README.md)
- [docs/fase2_1/PLAN-TRABAJO.md](docs/fase2_1/PLAN-TRABAJO.md)
- [docs/fase2_1/api/README.md](docs/fase2_1/api/README.md)
- [docs/fase2_1/config/README.md](docs/fase2_1/config/README.md)
- [docs/fase2_1/core/training/fit_regression/README.md](docs/fase2_1/core/training/fit_regression/README.md)
- [docs/fase2_1/core/training/segmentation/README.md](docs/fase2_1/core/training/segmentation/README.md)
- [docs/fase2_2/README.md](docs/fase2_2/README.md)
- [docs/fase2_2/PLAN-TRABAJO.md](docs/fase2_2/PLAN-TRABAJO.md)
- [docs/fase2_3/PLAN-SAM3D-BODY.md](docs/fase2_3/PLAN-SAM3D-BODY.md)
- [docs/fase2_3/INTEGRACION-SAM3D-BODY-AMD.md](docs/fase2_3/INTEGRACION-SAM3D-BODY-AMD.md)
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

| Herramienta | Para que se usa en el proyecto |
|---|---|
| Blender (Python API) | Motor principal de renderizado y exportacion 3D. Se invoca via `blender -b -P` para aplicar texturas a mallas, ejecutar UV unwrap automatico y exportar a GLB/OBJ. |
| trimesh | Carga, inspeccion y manipulacion de mallas 3D en Python. Permite detectar UV, convertir nubes de puntos a malla proxy y hacer operaciones geometricas sin abrir Blender. |
| xatlas | Generacion automatica de UV maps sobre mallas sin coordenadas de textura, como paso previo a la aplicacion de texturas generadas. |

---

### 🌐 Backend (Docker)

| Herramienta | Para que se usa en el proyecto |
|---|---|
| fastapi | Framework web principal para exponer todos los endpoints REST del sistema: generacion, seleccion, feedback, preview 2D/3D y modelos 3D. |
| uvicorn | Servidor ASGI de alto rendimiento que ejecuta la aplicacion FastAPI dentro del contenedor Docker. |
| pydantic | Validacion y serializacion de contratos de datos (schemas de request/response), tambien usado para los contratos internos del pipeline (landmarks_contract, transform_contract). |

---

### 🧠 Tracking

| Herramienta | Para que se usa en el proyecto |
|---|---|
| mlflow (Docker) | Registro de experimentos de entrenamiento: parametros (l2, feature_dim), metricas (train_mae, train_rmse) y artefactos de modelo. Activado con `MLFLOW_TRACKING_ENABLED=1`. UI disponible en `http://localhost:5000`. |

---

### 💾 Almacenamiento

| Herramienta | Para que se usa en el proyecto |
|---|---|
| PostgreSQL (Docker) | Base de datos relacional para persistir feedback humano (aprobacion/rechazo), scores por candidato y metricas de proyecto. Con fallback JSONL si no esta disponible. |
| MinIO (Docker) | Storage de objetos compatible con S3 para guardar texturas generadas, modelos 3D exportados y artefactos de entrenamiento. |

---

### 🌐 Frontend

| Herramienta | Para que se usa en el proyecto |
|---|---|
| three.js | Visor 3D interactivo en navegador. Carga modelos OBJ/FBX/GLTF, aplica la textura seleccionada y permite rotacion, zoom y paneo sin necesidad de Blender. |

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

### Actualizacion reciente (2026-04-19)

Novedades implementadas y validadas:

* **Fuente de modelos 3D cambiada**: el índice ahora usa mallas OBJ reales desde `data_deepfasho_antiguo/mesh/`
  * Estructura: `mesh/<garment_id>-<pose>/model_cleaned.obj` (con `.mtl` y textura PNG adjunta)
  * Reemplaza el uso anterior de nubes de puntos `.ply` de `data_deepfasho_antiguo/pointcloud/`
  * **1212 modelos OBJ** distribuidos en **~492 IDs** únicos, todos con malla triangulada lista para texturizar
* Fuente secundaria de nubes de puntos `data_deepfashon/point_cloud/` se mantiene disponible en el índice
* La UI muestra correctamente los modelos de mesh antiguo con badge OBJ (no POINT CLOUD)
* URL de servicio de archivos estáticos: `/deepfashion-antiguo-artifacts/mesh/...`
* El campo `is_point_cloud` es `false` para todos los modelos en `mesh/`
* El ranking de fuentes fue ajustado: `local > deepfashion_antiguo (mesh) > deepfashion (point_cloud)`

### Actualizacion anterior (2026-04-18)

Novedades implementadas y validadas:

* Integracion de modelos DeepFashion desde dos fuentes (ahora actualizadas):
  * `data_deepfashon/point_cloud/...` (nubes de puntos)
  * `data_deepfasho_antiguo/mesh/...` (mallas OBJ — fuente primaria)
* Etiquetado automatico por tipo de prenda (`shirt`, `dress`, `pants`) usando `cloth_type_list.txt`
* Deteccion de `point cloud` en API/UI con metadatos (`source`, `is_point_cloud`, `garment_type`)
* Conversion on-demand de nubes de puntos (`.ply`) a malla proxy para operaciones backend:
  * `POST /projects/preview-3d`
  * `POST /projects/select`
* Cache de conversion en `data/processed/pointcloud_meshes/`
* En scripts Blender se intenta UV automatico (`smart_project`) cuando la malla no tiene UV
* Auditoria reproducible de cobertura de IDs DeepFashion con script dedicado:
  * `python -m scripts.analyze_deepfashion_coverage --old-root data_deepfasho_antiguo --new-root data_deepfashon --output data/processed/deepfashion/coverage_report.json`
  * Resultado actual: `556` IDs compartidos, `42` IDs nuevos en dataset nuevo, cobertura de `100%` del antiguo dentro del nuevo

Notas:

* `data_deepfasho_antiguo/mesh/` contiene **mallas trianguladas reales** (OBJ) con UV y textura PNG por carpeta.
* `data_deepfashon/point_cloud/` son nubes de puntos densas; útiles solo si no hay OBJ equivalente.
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
  * `scripts/analyze_deepfashion_coverage.py`

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
