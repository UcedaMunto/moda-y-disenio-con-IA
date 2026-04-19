"""Modelo multisalida de fit_regression para Fase 2.2.

Extiende el baseline de escala (Fase 2.1) prediciendo tres targets:
  - scale    : factor de escala de la prenda
  - offset_x : desplazamiento horizontal normalizado
  - offset_y : desplazamiento vertical normalizado

Implementacion: ridge regression multisalida (matriz Y de (n, 3)),
misma forma cerrada que el baseline: w = (X^T X + lambda I)^-1 X^T Y.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fase2_1.core.training.fit_regression.baseline import (
    GARMENT_ORDER,
    POINT_ORDER,
    _safe_float,
    extract_feature_vector,
    garment_one_hot,
)

TARGET_NAMES = ["scale", "offset_x", "offset_y"]


def extract_target_vector(row: dict) -> list[float]:
    """Extrae [scale, offset_x, offset_y] desde transform_contract."""
    transform = row.get("transform_contract") or {}
    scale = _safe_float(transform.get("scale", 1.0), 1.0)
    offset_x = _safe_float(transform.get("offset_x", 0.0))
    offset_y = _safe_float(transform.get("offset_y", 0.0))
    return [scale, offset_x, offset_y]


def fit_ridge_multitarget(
    features: list[list[float]],
    targets: list[list[float]],
    l2: float = 1e-3,
) -> dict:
    """Ridge regression multisalida (3 targets: scale, offset_x, offset_y).

    Returns:
        dict con weights (feature_dim x 3), bias (3,), metricas por dimension.
    """
    if not features:
        raise ValueError("No hay features para entrenar")
    if len(features) != len(targets):
        raise ValueError("Features y targets deben tener misma longitud")

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)  # (n, 3)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Augment with bias column.
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    x_aug = np.concatenate([x, ones], axis=1)  # (n, d+1)

    eye = np.eye(x_aug.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0  # Do not regularize bias.

    # w shape: (d+1, 3)
    w = np.linalg.pinv(x_aug.T @ x_aug + l2 * eye) @ x_aug.T @ y

    preds = x_aug @ w  # (n, 3)
    residuals = preds - y  # (n, 3)
    mae_per_dim = np.mean(np.abs(residuals), axis=0).tolist()
    rmse_per_dim = np.sqrt(np.mean(residuals ** 2, axis=0)).tolist()

    metrics: dict = {}
    for i, name in enumerate(TARGET_NAMES):
        metrics[f"train_mae_{name}"] = round(float(mae_per_dim[i]), 6)
        metrics[f"train_rmse_{name}"] = round(float(rmse_per_dim[i]), 6)
    metrics["n_samples"] = int(x.shape[0])

    return {
        "status": "ok",
        "feature_dim": int(x.shape[1]),
        "target_names": TARGET_NAMES,
        "weights": w[:-1, :].tolist(),   # (feature_dim, 3)
        "bias": w[-1, :].tolist(),        # (3,)
        "metrics": metrics,
    }


def predict_transform(model: dict, feature_vector: list[float]) -> dict[str, float]:
    """Predice scale, offset_x y offset_y desde un vector de features."""
    weights = np.asarray(model.get("weights", []), dtype=np.float64)  # (d, 3)
    bias = np.asarray(model.get("bias", [0.0, 0.0, 0.0]), dtype=np.float64)  # (3,)
    x = np.asarray(feature_vector, dtype=np.float64)

    if x.shape[0] != weights.shape[0]:
        raise ValueError(
            f"Dimension de feature incompatible: input={x.shape[0]}, modelo={weights.shape[0]}"
        )

    preds = x @ weights + bias  # (3,)
    return {name: float(preds[i]) for i, name in enumerate(TARGET_NAMES)}


def save_model(model: dict, output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


def load_model(model_path: str) -> dict:
    src = Path(model_path)
    if not src.exists():
        raise FileNotFoundError(f"No existe modelo: {model_path}")
    return json.loads(src.read_text(encoding="utf-8"))
