import numpy as np
from PIL import Image


def detect_pose(image_path: str):
    """Detecta landmarks de pose en imagen estatica (baseline 2.1)."""
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

    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

    return results.pose_landmarks
