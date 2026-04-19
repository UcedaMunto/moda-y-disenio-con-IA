# Fase 2.1 API

Arranque minimo para ejecutar el pipeline de try-on de Fase 2.1 sin mezclarlo todavia con apps/api principal.

## Ejecutar local

```bash
uvicorn fase2_1.api.main:app --host 0.0.0.0 --port 8010 --reload
```

## Endpoints

- GET /health
- POST /tryon/run
- POST /tryon/batch
- POST /tryon/evaluate
- GET /tryon/evaluate-summary
- GET /tryon/evaluate-consolidated

Tambien disponible desde API principal:

- POST /fase2_1/tryon/run
- POST /fase2_1/tryon/batch
- POST /fase2_1/tryon/evaluate
- GET /fase2_1/tryon/evaluate-summary
- GET /fase2_1/tryon/evaluate-consolidated

Query params adicionales para evaluacion:

- `valid_score_threshold` (float, default `85.0`) en `GET /tryon/evaluate-summary` y `GET /tryon/evaluate-consolidated`.
- Se usa para calcular `visually_valid_rate` y la bandera consolidada `manual_acceptance_passed` (objetivo >= 90%).

Ejemplo payload:

```json
{
	"image_path": "data/raw/personas/persona_001.jpg",
	"garment_path": "data/raw/models/TShirts.obj",
	"output_path": "data/processed/fase2_1/outputs/persona_001.jpg"
}
```

Ejemplo payload batch:

```json
{
	"input_dir": "data/raw/personas",
	"garment_path": "data/raw/models/TShirts.obj",
	"garment_type": "shirt",
	"output_dir": "data/processed/fase2_1_eval/outputs",
	"report_path": "data/processed/fase2_1_eval/report.json",
	"checklist_path": "fase2_1/config/quality_checklist.json",
	"limit": 100
}
```

Categorias permitidas en `garment_type`: `shirt`, `skirt`, `pants`, `dress`, `other`.

La respuesta de `tryon/run` incluye en `meta`:

- `landmarks_contract` (versionado, espacio normalizado)
- `transform_contract` (scale, rotation_deg, translation_norm, garment_type)

Ejemplo payload evaluacion manual:

```json
{
	"image": "data/processed/fase2_1_eval/outputs/persona_001.jpg",
	"output_path": "data/processed/fase2_1_eval/outputs/persona_001.jpg",
	"criteria_scores": {
		"alignment_shoulders": 1.0,
		"torso_scale": 0.8,
		"occlusion_consistency": 0.7,
		"edge_artifacts": 0.9
	},
	"reviewer": "qa_user",
	"comment": "alineacion correcta, bordes aceptables",
	"report_path": "data/processed/fase2_1_eval/manual_reviews.jsonl"
}
```
