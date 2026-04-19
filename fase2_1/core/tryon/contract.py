from __future__ import annotations


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_landmarks_contract(landmarks, version: str = "1.0") -> dict:
    """Construye formato canónico de landmarks para Fase 2.1.

    La fuente esperada es mediapipe pose (normalized coordinates).
    """
    keypoints = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
    }

    points: dict[str, dict] = {}
    for name, idx in keypoints.items():
        lm = landmarks.landmark[idx]
        points[name] = {
            "x": _safe_float(getattr(lm, "x", 0.0)),
            "y": _safe_float(getattr(lm, "y", 0.0)),
            "z": _safe_float(getattr(lm, "z", 0.0)),
            "visibility": _safe_float(getattr(lm, "visibility", 1.0), 1.0),
        }

    return {
        "version": version,
        "space": "image_normalized",
        "backend": "mediapipe_pose",
        "points": points,
    }


def build_transform_contract(scale: float, garment_type: str, version: str = "1.0") -> dict:
    """Formato canónico de parámetros de transformación para prenda 2D."""
    return {
        "version": version,
        "garment_type": (garment_type or "other").lower(),
        "scale": _safe_float(scale),
        "rotation_deg": 0.0,
        "translation_norm": {"x": 0.0, "y": 0.0},
        "warp_mode": "baseline_affine_placeholder",
    }
