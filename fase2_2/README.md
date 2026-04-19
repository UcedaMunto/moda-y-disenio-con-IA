# Fase 2.2 - Virtual Try-On (offset + segmentacion real)

## Estado

- Semana 1: completada (fit_regression multisalida: scale + offset_x + offset_y)
- Semana 2: completada (preparacion de dataset real con validacion IoU)

## Scripts principales

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
