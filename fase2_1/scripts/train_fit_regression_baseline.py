from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fase2_1.core.training.fit_regression.baseline import (
    extract_feature_vector,
    extract_target_scale,
    fit_ridge_regression,
    load_rows_from_jsonl,
    save_model,
)


def _mlflow_log(params: dict, metrics: dict, artifact_path: str | None) -> None:
    tracking_enabled = os.environ.get("MLFLOW_TRACKING_ENABLED", "0").strip() not in ("", "0", "false", "False")
    if not tracking_enabled:
        return
    try:
        import mlflow  # type: ignore

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fit_regression_baseline")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if artifact_path:
                mlflow.log_artifact(artifact_path)
    except Exception as exc:  # pragma: no cover
        print(f"[mlflow] tracking omitido: {exc}", flush=True)


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

    metrics = model.get("metrics", {})
    _mlflow_log(
        params={
            "l2": args.l2,
            "dataset_jsonl": str(Path(args.dataset_jsonl).name),
            "feature_dim": model.get("feature_dim", 0),
            "n_samples": metrics.get("n_samples", 0),
        },
        metrics={
            "train_mae": metrics.get("train_mae", 0.0),
            "train_rmse": metrics.get("train_rmse", 0.0),
        },
        artifact_path=saved,
    )

    payload = {
        "status": "ok",
        "saved_model_path": saved,
        "metrics": metrics,
        "feature_dim": model.get("feature_dim", 0),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
