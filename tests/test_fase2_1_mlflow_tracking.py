"""Tests para la capa de MLflow tracking del script train_fit_regression_baseline."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch, call
import pytest


def _load_script() -> ModuleType:
    """Importa (o recarga) el modulo del script de entrenamiento."""
    mod_name = "fase2_1.scripts.train_fit_regression_baseline"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# _mlflow_log deshabilitado (por defecto MLFLOW_TRACKING_ENABLED no esta)
# ---------------------------------------------------------------------------

def test_mlflow_log_disabled_por_defecto(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_ENABLED", raising=False)
    mod = _load_script()
    mock_mlflow = MagicMock()
    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        mod._mlflow_log({"l2": 0.001}, {"train_mae": 0.1}, None)
    mock_mlflow.start_run.assert_not_called()


def test_mlflow_log_deshabilitado_con_cero(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "0")
    mod = _load_script()
    mock_mlflow = MagicMock()
    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        mod._mlflow_log({"l2": 0.001}, {"train_mae": 0.1}, None)
    mock_mlflow.start_run.assert_not_called()


# ---------------------------------------------------------------------------
# _mlflow_log habilitado
# ---------------------------------------------------------------------------

def test_mlflow_log_habilitado_registra_params_y_metricas(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_exp")

    mod = _load_script()

    mock_mlflow = MagicMock()
    ctx_manager = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = lambda s: ctx_manager
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        mod._mlflow_log(
            params={"l2": 0.001, "feature_dim": 17},
            metrics={"train_mae": 0.05, "train_rmse": 0.07},
            artifact_path=None,
        )

    mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    mock_mlflow.set_experiment.assert_called_once_with("test_exp")
    mock_mlflow.log_params.assert_called_once_with({"l2": 0.001, "feature_dim": 17})
    mock_mlflow.log_metrics.assert_called_once_with({"train_mae": 0.05, "train_rmse": 0.07})
    mock_mlflow.log_artifact.assert_not_called()


def test_mlflow_log_registra_artefacto_si_artifact_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "1")
    mod = _load_script()

    artifact = tmp_path / "model.json"
    artifact.write_text("{}")

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        mod._mlflow_log({}, {}, str(artifact))

    mock_mlflow.log_artifact.assert_called_once_with(str(artifact))


def test_mlflow_log_ignora_excepcion_de_mlflow(monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "1")
    mod = _load_script()

    mock_mlflow = MagicMock()
    mock_mlflow.set_tracking_uri.side_effect = RuntimeError("no server")

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        # No debe lanzar excepcion
        mod._mlflow_log({"l2": 0.001}, {"train_mae": 0.1}, None)
