from __future__ import annotations

import json
from pathlib import Path

import numpy as np


GARMENT_ORDER = ["shirt", "skirt", "pants", "dress", "other"]
POINT_ORDER = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def garment_one_hot(garment_type: str) -> list[float]:
    g = (garment_type or "other").lower()
    out = [0.0] * len(GARMENT_ORDER)
    try:
        out[GARMENT_ORDER.index(g)] = 1.0
    except ValueError:
        out[-1] = 1.0
    return out


def extract_feature_vector(row: dict) -> list[float]:
    points = ((row.get("landmarks_contract") or {}).get("points") or {})

    vector: list[float] = []
    for name in POINT_ORDER:
        point = points.get(name, {})
        vector.append(_safe_float(point.get("x", 0.0)))
        vector.append(_safe_float(point.get("y", 0.0)))
        vector.append(_safe_float(point.get("visibility", 1.0), 1.0))

    garment_type = (
        (row.get("transform_contract") or {}).get("garment_type")
        or row.get("garment_type")
        or "other"
    )
    vector.extend(garment_one_hot(str(garment_type)))
    return vector


def extract_target_scale(row: dict) -> float:
    transform = row.get("transform_contract") or {}
    return _safe_float(transform.get("scale", 1.0), 1.0)


def fit_ridge_regression(features: list[list[float]], targets: list[float], l2: float = 1e-3) -> dict:
    if not features:
        raise ValueError("No hay features para entrenar")
    if len(features) != len(targets):
        raise ValueError("Features y targets deben tener misma longitud")

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)

    # Add bias term.
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    x_aug = np.concatenate([x, ones], axis=1)

    # Closed-form ridge: w = (X^T X + lambda I)^-1 X^T y
    eye = np.eye(x_aug.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0  # Do not regularize bias.
    w = np.linalg.pinv(x_aug.T @ x_aug + l2 * eye) @ x_aug.T @ y

    preds = x_aug @ w
    mae = float(np.mean(np.abs(preds - y)))
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))

    return {
        "status": "ok",
        "feature_dim": int(x.shape[1]),
        "weights": w[:-1, 0].tolist(),
        "bias": float(w[-1, 0]),
        "metrics": {
            "train_mae": round(mae, 6),
            "train_rmse": round(rmse, 6),
            "n_samples": int(x.shape[0]),
        },
    }


def predict_scale(model: dict, feature_vector: list[float]) -> float:
    weights = np.asarray(model.get("weights", []), dtype=np.float64)
    bias = _safe_float(model.get("bias", 0.0))
    x = np.asarray(feature_vector, dtype=np.float64)
    if x.shape[0] != weights.shape[0]:
        raise ValueError("Dimension de feature incompatible con el modelo")
    return float(x @ weights + bias)


def load_rows_from_jsonl(path: str) -> list[dict]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"No existe dataset jsonl: {path}")

    rows: list[dict] = []
    for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def save_model(model: dict, output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)
