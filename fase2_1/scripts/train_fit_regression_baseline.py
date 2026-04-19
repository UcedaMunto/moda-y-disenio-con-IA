from __future__ import annotations

import argparse
import json
from pathlib import Path

from fase2_1.core.training.fit_regression.baseline import (
    extract_feature_vector,
    extract_target_scale,
    fit_ridge_regression,
    load_rows_from_jsonl,
    save_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento baseline de fit_regression (escala)")
    parser.add_argument(
        "--dataset-jsonl",
        required=True,
        help="JSONL con landmarks_contract y transform_contract",
    )
    parser.add_argument(
        "--output-model",
        default="data/processed/fase2_1_train/fit_regression/model_baseline.json",
        help="Ruta de salida del modelo entrenado",
    )
    parser.add_argument("--l2", type=float, default=1e-3, help="Regularizacion ridge")
    args = parser.parse_args()

    rows = load_rows_from_jsonl(args.dataset_jsonl)
    features = [extract_feature_vector(row) for row in rows]
    targets = [extract_target_scale(row) for row in rows]

    model = fit_ridge_regression(features=features, targets=targets, l2=args.l2)
    model["dataset_jsonl"] = str(Path(args.dataset_jsonl))
    model["l2"] = args.l2

    saved = save_model(model, args.output_model)
    payload = {
        "status": "ok",
        "saved_model_path": saved,
        "metrics": model.get("metrics", {}),
        "feature_dim": model.get("feature_dim", 0),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
