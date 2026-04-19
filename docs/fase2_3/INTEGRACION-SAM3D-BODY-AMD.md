# Integracion SAM 3D Body en Fabric2Mesh (AMD/ROCm)

## Resumen

Este documento aterriza como usar `sam-3d-body/` dentro de este repositorio para generar una malla 3D de persona desde foto y convertirla en un "maniqui posed" util para vestir prendas.

Ruta base del proyecto externo (ya clonado):

- `sam-3d-body/`

## Fuente y referencias

Basado en:

1. `sam-3d-body/README.md`
2. `sam-3d-body/INSTALL.md`
3. `sam-3d-body/demo.py`
4. `sam-3d-body/sam_3d_body/sam_3d_body_estimator.py`

## Hallazgos tecnicos relevantes

1. El modelo corre sobre `torch.device("cuda")` cuando `torch.cuda.is_available()` es true.
2. En ROCm, PyTorch tambien expone backend via `cuda` API (HIP), por lo que ese punto es compatible.
3. `process_one_image(...)` soporta pasar `bboxes` y `masks` externas.
4. Si no hay detector, el estimador puede usar bbox full-image como fallback.
5. El demo oficial guarda imagen de visualizacion, pero no exporta malla por defecto.
6. En notebook hay utilidades para exportar mallas (`save_mesh_results`) en PLY.

## Implicacion para AMD

Lo critico no es SAM 3D Body puro, sino dependencias auxiliares:

1. Detectron2 en AMD puede ser el principal bloqueo.
2. Si detectron2 falla, podemos seguir con una ruta MVP:
   - detector desactivado
   - bbox externa (de otro modulo) o full-image
3. Esto permite validar el objetivo central (malla 3D desde foto) sin bloquear por detector.

## Estrategia de integracion recomendada

## Etapa A - Inferencia minima robusta

1. Instalar SAM 3D Body en entorno separado (host, no en contenedor API).
2. Usar checkpoint aprobado de Hugging Face.
3. Ejecutar inferencia con fallback sin detector cuando sea necesario.

Comando orientativo:

```bash
cd sam-3d-body
python demo.py \
  --image_folder ../fotos_personas \
  --output_folder ../data/processed/fase2_3/sam3d_body_demo \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name ""
```

Nota:

- `--detector_name ""` evita depender de detectron2 en etapa inicial.

Alternativa recomendada en este repo (wrapper local):

```bash
python scripts/run_sam3d_body_smoke.py \
  --checkpoint-path sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr-path sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

Este wrapper:

1. Valida rutas de entrada/checkpoints.
2. Ejecuta `sam-3d-body/demo.py` desde su carpeta.
3. Guarda resultados en `data/processed/fase2_3/sam3d_body_smoke/`.

## Etapa B - Export de malla estandar

1. Crear wrapper en este repo que:
   - llame al estimador
   - extraiga `pred_vertices` + `faces`
   - exporte `mesh.ply` por persona
2. Guardar metadata minima:
   - `bbox`
   - `focal_length`
   - `run_time_ms`
   - `backend` (rocm/cpu)

Salida recomendada por caso:

```text
data/processed/fase2_3/<case_id>/
  input.jpg
  overlay.jpg
  person_000_mesh.ply
  person_000_meta.json
  run_meta.json
```

Script implementado para esta etapa:

- `scripts/run_sam3d_body_single.py`

Uso directo:

```bash
python scripts/run_sam3d_body_single.py \
  --repo-root . \
  --sam3d-root sam-3d-body \
  --image-path fotos_personas/2f7b90fbaaa9476253d6d993e6ddf487.jpg \
  --output-dir data/processed/fase2_3/case-manual-0001 \
  --checkpoint-path sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr-path sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector-name ""
```

Este script genera:

1. `*_mesh_000.ply` (una por persona detectada)
2. `*_mesh_000.json` (bbox/focal/cam)
3. `*_overlay.jpg`
4. `result.json`

## Etapa C - Maniqui posed para pipeline de prenda

1. Convertir `ply` a `glb` con script Blender existente o nuevo script de conversion.
2. Normalizar orientacion/escala para contrato interno.
3. Exponer ruta de salida como "avatar base" para etapa de vestir prenda.

## Etapa D - API y orquestacion

Endpoint propuesto:

- `GET /fase2_3/body/preflight`
- `POST /fase2_3/body/reconstruct`

Request minimo:

```json
{
  "image_path": "fotos_personas/2f7b90fbaaa9476253d6d993e6ddf487.jpg",
  "case_id": "case-fase2_3-0001",
  "use_mask": true
}
```

Response minimo:

```json
{
  "status": "ok",
  "case_id": "case-fase2_3-0001",
  "mesh_paths": ["data/processed/fase2_3/case-fase2_3-0001/person_000_mesh.ply"],
  "preview_path": "data/processed/fase2_3/case-fase2_3-0001/overlay.jpg",
  "meta_path": "data/processed/fase2_3/case-fase2_3-0001/run_meta.json"
}
```

Notas operativas:

1. `checkpoint_path` y `mhr_path` se pueden enviar en request o resolver desde entorno (`SAM3D_CHECKPOINT_PATH`, `SAM3D_MHR_PATH`).
2. `preflight` valida existencia de script, repo `sam-3d-body`, checkpoint y asset MHR antes de ejecutar inferencia.
3. La reconstruccion puede exportar `glb` (`export_glb=true`) para usarlo como avatar base en pipeline posterior.

## Compatibilidad y decisiones (AMD first)

1. Mantener API FastAPI en Docker para servicios ligeros.
2. Ejecutar inferencia SAM 3D Body en host (entorno Python dedicado ROCm).
3. API invoca wrapper local via subprocess/controlado, no mete toda la cadena SAM dentro del contenedor por ahora.

Razon:

- Reduce friccion con ROCm + detectron2 + drivers.
- Acelera el primer resultado funcional.

## Checklist operativo

1. Acceso autorizado a checkpoints en Hugging Face.
2. Entorno Python `sam_3d_body` creado y activo.
3. PyTorch ROCm verificado (`torch.cuda.is_available() == True` en AMD).
4. Demo ejecutando sobre al menos una foto local.
5. Export de al menos un `.ply` por imagen.

## Criterios de aceptacion inicial

1. Para una foto valida, se genera malla 3D sin error.
2. Se obtiene al menos un artefacto visual de control.
3. El output queda en `data/processed/fase2_3/` con metadata trazable.
4. El flujo es repetible por comando.

## Estado actual (iteracion en curso)

1. Wrapper de reconstruccion single-image implementado en `scripts/run_sam3d_body_single.py`.
2. Endpoint `POST /fase2_3/body/reconstruct` operativo con modo `use_mock` para pruebas sin checkpoints.
3. Endpoint `GET /fase2_3/body/preflight` agregado para validar entorno/config.
4. UI `/tryon` incluye boton "Generar maniqui 3D" y accion de preflight 3D.
