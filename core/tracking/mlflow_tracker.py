import os
from typing import Any


def _enabled() -> bool:
    return os.getenv("MLFLOW_TRACKING_ENABLED", "0") == "1"


def track_event(event_name: str, payload: dict[str, Any]) -> None:
    if not _enabled():
        return

    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT", "fabric2mesh")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=event_name):
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, str(value))
    except Exception:
        # Tracking is optional and should never block the core pipeline.
        return
