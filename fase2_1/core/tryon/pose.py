import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from PIL import Image


_POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _pose_landmarker_model_path() -> Path:
    model_dir = Path("data/processed/fase2_1_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "pose_landmarker_lite.task"


def _ensure_pose_landmarker_model() -> Path:
    model_path = _pose_landmarker_model_path()
    if not model_path.exists() or model_path.stat().st_size < 100_000:
        urlretrieve(_POSE_LANDMARKER_URL, model_path)
    return model_path


def _detect_pose_legacy(image_rgb: np.ndarray):
    import mediapipe as mp

    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
    return results.pose_landmarks


def _detect_pose_landmarker(image_rgb: np.ndarray):
    import mediapipe as mp

    model_path = _ensure_pose_landmarker_model()
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)
    if not result.pose_landmarks:
        return None

    class _Point:
        __slots__ = ("x", "y", "z", "visibility")

        def __init__(self, x, y, z=0.0, visibility=1.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.visibility = float(visibility)

    class _Landmarks:
        __slots__ = ("landmark",)

        def __init__(self, points):
            self.landmark = points

    points = [
        _Point(lm.x, lm.y, getattr(lm, "z", 0.0), getattr(lm, "visibility", 1.0))
        for lm in result.pose_landmarks[0]
    ]
    return _Landmarks(points)


def detect_pose(image_path: str, backend: str = "landmarker"):
    """Detecta landmarks de pose en imagen estatica (baseline 2.1).

    backend: legacy | landmarker | both (landmarker con fallback a legacy)
    """
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError(
            "mediapipe no disponible para Fase 2.1; instala dependencias de vision."
        ) from exc

    try:
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
    except Exception as exc:
        raise FileNotFoundError(f"No se pudo leer imagen: {image_path}")

    backend_norm = (backend or "landmarker").strip().lower()
    if backend_norm == "legacy":
        return _detect_pose_legacy(image_rgb)
    if backend_norm == "landmarker":
        return _detect_pose_landmarker(image_rgb)
    if backend_norm == "both":
        lm = _detect_pose_landmarker(image_rgb)
        return lm if lm is not None else _detect_pose_legacy(image_rgb)

    return _detect_pose_landmarker(image_rgb)
