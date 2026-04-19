# Fase 2.2 - Virtual Try-On (offset + segmentacion real)

## Estado

- Semana 1: completada (fit_regression multisalida: scale + offset_x + offset_y)
- Semana 2: completada (preparacion de dataset real con validacion IoU)
- Semana 3: en ejecucion (adaptador de parsing + fine-tuning ligero por calibracion)

## Scripts principales

### 0) Pipeline 2.2 con offset

Modulo nuevo: `fase2_2.core.tryon.pipeline_v2`.

`run_tryon_v2(...)` agrega:

- Prediccion de `scale + offset_x + offset_y` desde modelo multisalida.
- Segmentacion v2 via `segment_person_v2` con `model_config_path` opcional.
- Metadata de trazabilidad en `meta.transform_source`.

### 1) Entrenar fit_regression multisalida

```bash
python -m fase2_2.scripts.train_fit_regression_offset \
  --dataset-jsonl /tmp/demo_dataset.jsonl \
  --output-model data/processed/fase2_2_train/fit_regression/model_offset_run1.json \
  --l2 1e-3
```

Salida: modelo JSON con metricas por dimension:

- `train_mae_scale`
- `train_mae_offset_x`
- `train_mae_offset_y`

### 2) Preparar dataset real de segmentacion

Este script asume que ya descargaste un dataset publico (por ejemplo LIP/Supervisely)
y tienes carpetas locales separadas para imagenes y mascaras.

```bash
python -m fase2_2.scripts.prepare_segmentation_real_dataset \
  --source-images-dir data/raw/public_seg/images \
  --source-masks-dir data/raw/public_seg/masks \
  --output-dir data/processed/fase2_2_train/segmentation_real \
  --max-samples 200 \
  --min-iou 0.25 \
  --seed 42
```

Archivos generados:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `low_iou.jsonl`
- `manifest.json`

### 3) Fine-tuning ligero de segmentacion (calibracion de threshold)

```bash
python -m fase2_2.scripts.train_segmentation_lite \
  --train-jsonl data/processed/fase2_2_train/segmentation_real/train.jsonl \
  --val-jsonl data/processed/fase2_2_train/segmentation_real/val.jsonl \
  --output-config data/processed/fase2_2_train/segmentation_lite/segmentation_model_config.json \
  --candidate-thresholds 0.25,0.30,0.35,0.40,0.45
```

Salida:

- Config JSON de modelo (`model_name`, `backend`, `threshold`).
- Metricas de IoU en train y validacion.
- Historial de pruebas por threshold (`trials`).

Tracking opcional MLflow:

- `MLFLOW_TRACKING_ENABLED=1`
- `MLFLOW_TRACKING_URI=http://localhost:5000`

Campos por muestra (JSONL):

- `image_path`, `mask_path`, `auto_mask_path`
- `iou_auto`
- `auto_backend`
- `auto_mask_coverage`
- `quality_pass`

## Testing

```bash
python -m pytest tests/test_fase2_2_fit_regression_offset.py -q
python -m pytest tests/test_fase2_2_prepare_segmentation_real_dataset.py -q
```
