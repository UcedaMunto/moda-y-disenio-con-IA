# Fit Regression - Fase 2.1

Este modulo entrena un baseline regresivo para estimar parametros geometricos de ajuste.

Entradas del baseline actual:

- landmarks canonicos (`landmarks_contract.points`)
- categoria de prenda (`garment_type`) como one-hot

Salida actual:

- `scale` de ajuste

## Entrenamiento baseline (implementado)

Script disponible:

- `fase2_1/scripts/train_fit_regression_baseline.py`

Formato esperado del dataset (`.jsonl`):

- `landmarks_contract`
- `transform_contract.scale`
- `transform_contract.garment_type`

Ejemplo:

```bash
python fase2_1/scripts/train_fit_regression_baseline.py \
  --dataset-jsonl data/processed/fase2_1_train/fit_regression/train.jsonl \
  --output-model data/processed/fase2_1_train/fit_regression/model_baseline.json \
  --l2 0.001
```

Salida:

- pesos del modelo lineal regularizado (ridge)
- bias
- metricas de entrenamiento (`train_mae`, `train_rmse`)
