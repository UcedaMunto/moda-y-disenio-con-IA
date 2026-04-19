# Fase 2.2 - Virtual Try-On (offset + segmentacion real)

## Estado

- Semana 1: completada (fit_regression multisalida: scale + offset_x + offset_y)
- Semana 2: completada (preparacion de dataset real con validacion IoU)
- Semana 3: completada (adaptador de parsing + fine-tuning ligero por calibracion)
- Semana 4: completada (pipeline v2 + benchmark comparativo + demo interna)

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

### 4) Benchmark comparativo 2.1 vs 2.2

```bash
python -m fase2_2.scripts.benchmark_compare_v21_v22 \
  --input-dir data/raw/person_images \
  --garment-path data/raw/garments/shirt.png \
  --garment-type shirt \
  --output-dir data/processed/fase2_2_benchmark \
  --report data/processed/fase2_2_benchmark/compare_report.json
```

Reporte:

- Resumen de latencia y success rate para v2.1 y v2.2.
- Delta de performance (`avg_elapsed_ms`, `success_rate`).
- Resultados por imagen y trazabilidad de offsets en v2.2.

### 5) Demo interna reproducible

```bash
python -m fase2_2.scripts.run_internal_demo \
  --output-dir data/processed/fase2_2_demo \
  --report data/processed/fase2_2_demo/demo_report.json \
  --garment-type shirt
```

Resultado de la corrida actual:

- `samples=4`
- `v21.success_rate=100.0`
- `v22.success_rate=100.0`
- `delta.avg_elapsed_ms=-0.001`
- Reporte guardado en `data/processed/fase2_2_demo/demo_report.json`

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
python -m pytest tests/test_fase2_2_parsing_adapter.py -q
python -m pytest tests/test_fase2_2_train_segmentation_lite.py -q
python -m pytest tests/test_fase2_2_benchmark_compare.py -q
python -m pytest tests/test_fase2_2_pipeline_v2.py -q
python -m pytest tests/test_fase2_2_run_internal_demo.py -q
```

## Limites conocidos (cierre 2.2)

- El fine-tuning de segmentacion actual es una calibracion de threshold sobre backend preentrenado; no reentrena pesos de red.
- La demo interna se ejecuta en modo `mock_demo` para asegurar reproducibilidad en CI/entornos sin MediaPipe.
- La prediccion de offset usa clamp en rango normalizado (`[-0.3, 0.3]`) para evitar artefactos por extrapolacion.
- El benchmark comparativo metrico no incluye aun una metrica perceptual robusta (SSIM/LPIPS) entre outputs.

## Backlog 2.3 propuesto

1. Entrenamiento real de segmentacion (backbone ligero) sobre dataset etiquetado con validacion IoU por categoria.
2. Warp/deformacion de prenda guiada por landmarks (TPS o malla 2D) para mejorar ajuste visual.
3. Integracion de metrica perceptual automatica (SSIM/LPIPS) en benchmark de calidad.
4. Pipeline video con consistencia temporal y suavizado de jitter.
5. Evaluacion humana web integrada con persistencia y analitica de experimentos.
