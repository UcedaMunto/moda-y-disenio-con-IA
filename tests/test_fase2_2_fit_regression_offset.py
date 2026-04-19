"""Tests para el modelo multisalida fit_regression_offset (Fase 2.2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from fase2_2.core.training.fit_regression.offset import (
    TARGET_NAMES,
    extract_target_vector,
    fit_ridge_multitarget,
    load_model,
    predict_transform,
    save_model,
)
from fase2_1.core.training.fit_regression.baseline import extract_feature_vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(scale: float, offset_x: float = 0.0, offset_y: float = 0.0,
         garment_type: str = "shirt", x_shift: float = 0.0) -> dict:
    return {
        "landmarks_contract": {
            "points": {
                "left_shoulder":  {"x": 0.22 + x_shift, "y": 0.20, "visibility": 1.0},
                "right_shoulder": {"x": 0.78 + x_shift, "y": 0.20, "visibility": 1.0},
                "left_hip":       {"x": 0.28 + x_shift, "y": 0.70, "visibility": 1.0},
                "right_hip":      {"x": 0.72 + x_shift, "y": 0.70, "visibility": 1.0},
            }
        },
        "transform_contract": {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "garment_type": garment_type,
        },
    }


def _make_dataset(n: int = 8) -> tuple[list[list[float]], list[list[float]]]:
    rows = [
        _row(
            1.0 + i * 0.05,
            offset_x=0.01 * i,
            offset_y=-0.01 * i,
            x_shift=i * 0.005,
        )
        for i in range(n)
    ]
    features = [extract_feature_vector(r) for r in rows]
    targets = [extract_target_vector(r) for r in rows]
    return features, targets


# ---------------------------------------------------------------------------
# extract_target_vector
# ---------------------------------------------------------------------------

def test_extract_target_vector_defaults():
    row = {"transform_contract": {}}
    result = extract_target_vector(row)
    assert result == [1.0, 0.0, 0.0]


def test_extract_target_vector_valores():
    row = _row(scale=1.2, offset_x=0.05, offset_y=-0.03)
    result = extract_target_vector(row)
    assert result == pytest.approx([1.2, 0.05, -0.03])


# ---------------------------------------------------------------------------
# fit_ridge_multitarget
# ---------------------------------------------------------------------------

def test_fit_multitarget_devuelve_estructura_correcta():
    features, targets = _make_dataset()
    model = fit_ridge_multitarget(features, targets, l2=1e-3)
    assert model["status"] == "ok"
    assert model["target_names"] == TARGET_NAMES
    assert model["feature_dim"] == len(features[0])
    assert len(model["weights"]) == len(features[0])
    assert len(model["bias"]) == 3


def test_fit_multitarget_metricas_por_dimension():
    features, targets = _make_dataset()
    model = fit_ridge_multitarget(features, targets, l2=1e-3)
    metrics = model["metrics"]
    for name in TARGET_NAMES:
        assert f"train_mae_{name}" in metrics
        assert f"train_rmse_{name}" in metrics
    assert metrics["n_samples"] == len(features)


def test_fit_multitarget_error_sin_features():
    with pytest.raises(ValueError, match="No hay features"):
        fit_ridge_multitarget([], [], l2=1e-3)


def test_fit_multitarget_error_longitud_distinta():
    features, targets = _make_dataset(4)
    with pytest.raises(ValueError, match="misma longitud"):
        fit_ridge_multitarget(features, targets[:-1], l2=1e-3)


# ---------------------------------------------------------------------------
# predict_transform
# ---------------------------------------------------------------------------

def test_predict_transform_devuelve_tres_targets():
    features, targets = _make_dataset(10)
    model = fit_ridge_multitarget(features, targets, l2=1e-3)
    result = predict_transform(model, features[0])
    assert set(result.keys()) == set(TARGET_NAMES)
    for v in result.values():
        assert isinstance(v, float)


def test_predict_transform_error_dimension():
    features, targets = _make_dataset(4)
    model = fit_ridge_multitarget(features, targets, l2=1e-3)
    with pytest.raises(ValueError, match="Dimension de feature"):
        predict_transform(model, [0.0] * (len(features[0]) + 1))


def test_predict_transform_baja_mae_en_train():
    """El modelo debe reproducir bien los targets de entrenamiento."""
    features, targets = _make_dataset(20)
    model = fit_ridge_multitarget(features, targets, l2=1e-6)
    for fv, tv in zip(features, targets):
        pred = predict_transform(model, fv)
        for i, name in enumerate(TARGET_NAMES):
            assert abs(pred[name] - tv[i]) < 0.05, f"MAE alto en {name}"


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_load_model(tmp_path):
    features, targets = _make_dataset()
    model = fit_ridge_multitarget(features, targets)
    path = str(tmp_path / "subdir" / "model_offset.json")
    saved = save_model(model, path)
    loaded = load_model(saved)
    assert loaded["status"] == "ok"
    assert loaded["target_names"] == TARGET_NAMES
    assert loaded["weights"] == model["weights"]


def test_load_model_archivo_inexistente():
    with pytest.raises(FileNotFoundError):
        load_model("/tmp/no_existe_xyz.json")
