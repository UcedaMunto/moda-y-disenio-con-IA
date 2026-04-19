# Plan de Trabajo - Fase 2.2

## Objetivo

Extender el MVP 2.1 con prediccion de offset geometrico completo (escala + offset X/Y),
inicio de fine-tuning de segmentacion y preparacion de deformacion de tela.

---

## Duracion estimada

4 semanas (iteracion corta con entregables semanales)

---

## Entregables finales de la 2.2

1. Modelo `fit_regression` extendido con prediccion de escala + offset X/Y.
2. Script de entrenamiento actualizado con nuevas metricas (MAE por dimension).
3. Dataset real o semi-real de 50+ pares para segmentacion.
4. Primer intento de fine-tuning de segmentacion con evaluacion de IoU.
5. Documento de resultados y decisiones para pasar a 2.3.

---

## Cronograma

## Semana 1 - Offset geometrico completo

Objetivo:

- Extender fit_regression para predecir escala + offset_x + offset_y.

Tareas:

1. [x] Modelo multisalida en `fit_regression/offset.py` (3 targets: scale, offset_x, offset_y).
2. [x] Script de entrenamiento `train_fit_regression_offset.py` con MLflow.
3. [x] Dataset sintetico de offset con variacion controlada.
4. [x] Tests del modelo multisalida.

Criterios de aceptacion:

1. train_mae separado por dimension (scale, offset_x, offset_y).
2. Al menos 1 corrida registrada con MLflow.

Progreso actual Semana 1:

- [x] Modelo multisalida `fit_regression/offset.py` implementado y testeado (11 tests).
- [x] Script `train_fit_regression_offset.py` con MLflow optional tracking.
- [x] Primera corrida: `model_offset_run1.json` (train_mae_scale=0.000496, train_mae_offset_x=0.0, n_samples=5, feature_dim=17).

---

## Semana 2 - Dataset real de segmentacion

Objetivo:

- Construir o recolectar 50+ pares imagen-mascara reales para segmentacion.

Tareas:

1. [x] Script de descarga/preparacion de subset publico (LIP o supervisely-person).
2. [x] Validacion automatica de calidad de mascaras (IoU con segmentacion automatica).
3. [x] Manifest JSONL con metadatos por par.
4. [x] Tests de pipeline de carga.

Progreso actual Semana 2:

- [x] Script `prepare_segmentation_real_dataset.py` para preparar subset local de dataset publico ya descargado.
- [x] Validacion por muestra con `iou_auto` contra segmentacion automatica (`segment_person`).
- [x] Generacion de `train.jsonl`, `val.jsonl`, `test.jsonl`, `low_iou.jsonl` y `manifest.json`.
- [x] Tests `test_fase2_2_prepare_segmentation_real_dataset.py` (4 tests, 4 passed).

---

## Semana 3 - Fine-tuning de segmentacion

Objetivo:

- Ajustar el modelo de segmentacion con el dataset preparado.

Tareas:

1. [x] Adaptar `parsing.py` para aceptar modelo fine-tuneado.
2. [ ] Script de fine-tuning sobre backbone ligero (MobileNet o similar).
3. [ ] Evaluacion de IoU en conjunto de validacion.
4. [ ] Registrar experimento en MLflow.

Progreso actual Semana 3:

- [x] Adaptador `core/tryon/parsing_adapter.py` con `segment_person_v2` y carga de config de modelo fine-tuneado.
- [x] Soporte para `model_config_path` (threshold/backend/model_name) sin romper fallback robusto de Fase 2.1.
- [x] Tests `test_fase2_2_parsing_adapter.py` (5 tests, 5 passed).

---

## Semana 4 - Integracion y cierre

Objetivo:

- Integrar offset real en pipeline y cerrar iteracion.

Tareas:

1. [ ] Integrar prediccion de offset en `pipeline.py`.
2. [ ] Benchmark comparativo 2.1 vs 2.2 (visual + metrico).
3. [ ] Documentar limites conocidos y backlog 2.3.
4. [ ] Demo interna con casos representativos.

---

## Backlog tecnico 2.2

1. [ ] Deformacion de tela (warp/thin-plate spline).
2. [ ] Inferencia en video (consistencia temporal).
3. [ ] Evaluacion humana Web con guardado automatico.
