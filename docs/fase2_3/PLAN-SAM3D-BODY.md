# Plan de Trabajo - Fase 2.3 (SAM 3D Body)

## Contexto

El flujo actual de try-on en foto (Fase 2.1/2.2) depende de alineacion 2D y composicion por capas. Eso permite demos, pero no resuelve de forma robusta geometria real del cuerpo en poses complejas.

Para mejorar, integramos `sam-3d-body/` como motor de reconstruccion de malla humana desde una sola foto.

## Objetivo de Negocio

Pasar de "overlay 2D" a "drapeado sobre maniqui posed" para obtener:

1. Mejor ajuste en hombros, torso y cadera.
2. Mejor consistencia en poses inclinadas, brazos levantados y oclusiones.
3. Base reutilizable para pipeline de prenda 3D.

## Objetivo Tecnico

Dado `foto_persona`, producir:

1. `malla_persona_pose.ply` (o equivalente).
2. `metadata_pose.json` con bbox, intrinsecos/focal y score.
3. `maniqui_pose.glb` normalizado para la etapa de vestir prenda.

## Alcance de Fase 2.3

Incluye:

1. Integracion por inferencia de `sam-3d-body` (sin entrenamiento propio en esta fase).
2. Endpoint nuevo para generar malla de persona desde foto.
3. Conversor a formato de trabajo del repositorio (`glb` + metadata).
4. Demo interna de "foto -> maniqui posed -> preview".

No incluye:

1. Fine-tuning de SAM 3D Body.
2. Simulacion fisica avanzada de tela.
3. Calidad final de produccion en todas las poses extremas.

## Arquitectura Propuesta

```text
Foto persona
  -> Detector/segmentacion (SAM 3D Body)
  -> SAM 3D Body inference
  -> pred_vertices + faces
  -> export PLY/OBJ
  -> normalizacion de escala/orientacion
  -> maniqui_pose.glb
  -> pipeline de prenda (fase posterior)
```

## Plan por Iteraciones

## Iteracion 0 - Preparacion y smoke test

1. Validar entorno `sam-3d-body` en host Linux.
2. Descargar checkpoints autorizados desde Hugging Face.
3. Ejecutar demo sobre 3-5 fotos reales de `fotos_personas/`.
4. Guardar resultados en `data/processed/fase2_3/sam3d_body_smoke/`.

Criterio de salida:

- Al menos 3 casos con malla generada sin error.

## Iteracion 1 - Adaptador del repositorio

1. Crear wrapper local (script o modulo) para correr inferencia de forma reproducible.
2. Estandarizar salida:
   - `mesh.ply`
   - `overlay.jpg`
   - `focal_length.json`
   - `run_meta.json`
3. Normalizar naming por `case_id`.

Criterio de salida:

- Salidas estables y reproducibles para un mismo input.

## Iteracion 2 - Endpoint API

1. Crear endpoint nuevo en API principal, por ejemplo:
   - `POST /fase2_3/body/reconstruct`
2. Entrada minima:
   - `image_path`
   - opcional: `bbox`, `use_mask`
3. Respuesta:
   - paths de malla y artefactos
   - estado y metadatos de ejecucion

Criterio de salida:

- Endpoint funcional invocable desde UI o curl.

## Iteracion 3 - Maniqui posed util para vestir

1. Convertir malla inferida a `maniqui_pose.glb`.
2. Definir contrato geometrico de intercambio con pipeline de prenda.
3. Probar overlay/proyeccion de una prenda sobre ese maniqui.

Criterio de salida:

- Demo "foto -> maniqui -> prenda" en al menos 5 casos.

## Iteracion 4 - Calidad y fallback

1. Definir metricas de calidad minima (deteccion, completitud, estabilidad).
2. Fallback cuando no haya deteccion:
   - bbox full-image
   - degradar a pipeline 2D actual
3. Reporte comparativo 2D vs 3D.

Criterio de salida:

- Regla clara de seleccion de pipeline (2D/3D) por caso.

## Riesgos y Mitigaciones

1. Dependencias de GPU/ROCm y compatibilidad de librerias.
   - Mitigar con guia separada de AMD y modo CPU fallback.
2. Detectron2 puede complicar builds en AMD.
   - Mitigar usando bbox externas/full-image en primera iteracion.
3. Latencia alta por imagen.
   - Mitigar con colas batch y cache de resultados por foto.

## Entregables

1. Documentacion tecnica y operativa de Fase 2.3.
2. Script/adaptador de inferencia SAM 3D Body.
3. Endpoint API de reconstruccion.
4. Artefactos de prueba y reporte interno comparativo.

## Definicion de Hecho (DoD)

1. Un usuario puede enviar una foto y obtener malla/maniqui posed.
2. El artefacto queda guardado y versionado en `data/processed/fase2_3/`.
3. Existe demo reproducible con comando unico.
4. Se documenta claramente cuando usar pipeline 2D y cuando usar 3D.
