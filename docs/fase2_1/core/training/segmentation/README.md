# Segmentacion - Fase 2.1

Este modulo guardara scripts de entrenamiento/fine-tuning de segmentacion para try-on.

Objetivo de 2.1:

1. Preparar dataset de ropa/persona.
2. Ejecutar un primer fine-tuning reproducible.
3. Exportar pesos para inferencia en core/tryon.

## Pipeline de datos (implementado)

Script disponible:

- `fase2_1/scripts/prepare_segmentation_dataset.py`

Genera:

1. Pairing imagen-mascara por ruta relativa.
2. Split reproducible train/val/test con `seed`.
3. Manifests `train.jsonl`, `val.jsonl`, `test.jsonl` y `manifest.json`.

Ejemplo:

```bash
python fase2_1/scripts/prepare_segmentation_dataset.py \
	--images-dir data/raw/fase2_1_seg/images \
	--masks-dir data/raw/fase2_1_seg/masks \
	--output-dir data/processed/fase2_1_train/segmentation \
	--train-ratio 0.8 \
	--val-ratio 0.1 \
	--test-ratio 0.1 \
	--seed 42 \
	--strict
```
