from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from fase2_2.core.tryon.parsing_adapter import segment_person_v2


def load_rows_from_jsonl(path: str) -> list[dict]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"No existe manifest jsonl: {path}")

    rows: list[dict] = []
    for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_binary_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr >= 127).astype(np.uint8)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError("Mascaras con dimensiones distintas")

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def evaluate_threshold(samples: list[dict], threshold: float, model_config_path: str | None = None) -> dict:
    if not samples:
        return {"mean_iou": 0.0, "n_samples": 0}

    ious: list[float] = []
    backend_count: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="seg_ft_") as tmpdir:
        tmp = Path(tmpdir)
        for idx, row in enumerate(samples, start=1):
            image_path = row["image_path"]
            gt_mask_path = row["mask_path"]
            auto_path = tmp / f"mask_{idx:06d}.png"

            seg = segment_person_v2(
                image_path=image_path,
                output_mask_path=str(auto_path),
                threshold=threshold,
                model_config_path=model_config_path,
            )
            backend_count[seg.backend] = backend_count.get(seg.backend, 0) + 1

            gt = _read_binary_mask(gt_mask_path)
            pred = _read_binary_mask(str(auto_path))
            ious.append(compute_iou(gt, pred))

    return {
        "mean_iou": round(float(np.mean(ious)), 6),
        "n_samples": len(ious),
        "backend_count": backend_count,
    }


def tune_threshold(
    train_samples: list[dict],
    candidate_thresholds: list[float],
    model_config_path: str | None = None,
) -> dict:
    if not candidate_thresholds:
        raise ValueError("candidate_thresholds no puede estar vacio")

    trials: list[dict] = []
    for thr in candidate_thresholds:
        metrics = evaluate_threshold(train_samples, threshold=thr, model_config_path=model_config_path)
        trials.append({"threshold": thr, **metrics})

    best = max(trials, key=lambda x: x["mean_iou"])
    return {
        "best_threshold": best["threshold"],
        "best_train_iou": best["mean_iou"],
        "trials": trials,
    }


def _mlflow_log(params: dict, metrics: dict, artifact_path: str | None) -> None:
    tracking_enabled = os.environ.get("MLFLOW_TRACKING_ENABLED", "0").strip() not in ("", "0", "false", "False")
    if not tracking_enabled:
        return
    try:
        import mlflow  # type: ignore

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "segmentation_lite_ft")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if artifact_path:
                mlflow.log_artifact(artifact_path)
    except Exception as exc:  # pragma: no cover
        print(f"[mlflow] tracking omitido: {exc}", flush=True)


def train_segmentation_lite(
    train_jsonl: str,
    val_jsonl: str,
    output_config_path: str,
    candidate_thresholds: list[float] | None = None,
    base_model_config_path: str | None = None,
) -> dict:
    if candidate_thresholds is None:
        candidate_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    train_rows = load_rows_from_jsonl(train_jsonl)
    val_rows = load_rows_from_jsonl(val_jsonl)

    tuned = tune_threshold(
        train_samples=train_rows,
        candidate_thresholds=candidate_thresholds,
        model_config_path=base_model_config_path,
    )

    best_threshold = float(tuned["best_threshold"])
    val_metrics = evaluate_threshold(
        samples=val_rows,
        threshold=best_threshold,
        model_config_path=base_model_config_path,
    )

    config = {
        "model_name": "segmentation_lite_ft_v1",
        "backend": "mediapipe_selfie_segmentation",
        "threshold": best_threshold,
        "train_manifest": str(Path(train_jsonl)),
        "val_manifest": str(Path(val_jsonl)),
        "metrics": {
            "train_best_iou": tuned["best_train_iou"],
            "val_iou": val_metrics["mean_iou"],
            "n_train": len(train_rows),
            "n_val": len(val_rows),
        },
        "trials": tuned["trials"],
    }

    out = Path(output_config_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    _mlflow_log(
        params={
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "candidate_thresholds": ",".join(str(x) for x in candidate_thresholds),
            "base_model_config": str(base_model_config_path or ""),
        },
        metrics={
            "train_best_iou": float(tuned["best_train_iou"]),
            "val_iou": float(val_metrics["mean_iou"]),
        },
        artifact_path=str(out),
    )

    return {
        "status": "ok",
        "output_config_path": str(out),
        "best_threshold": best_threshold,
        "train_best_iou": tuned["best_train_iou"],
        "val_iou": val_metrics["mean_iou"],
        "n_train": len(train_rows),
        "n_val": len(val_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tuning ligero de segmentacion por calibracion de threshold (MediaPipe)"
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument(
        "--output-config",
        default="data/processed/fase2_2_train/segmentation_lite/segmentation_model_config.json",
    )
    parser.add_argument(
        "--candidate-thresholds",
        default="0.25,0.30,0.35,0.40,0.45,0.50",
        help="Lista separada por comas",
    )
    parser.add_argument("--base-model-config", default=None)
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.candidate_thresholds.split(",") if x.strip()]

    result = train_segmentation_lite(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_config_path=args.output_config,
        candidate_thresholds=thresholds,
        base_model_config_path=args.base_model_config,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
