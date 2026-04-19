from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class SegmentationResult:
    def __init__(self, mask_path: str | None, backend: str, note: str, mask_coverage: float):
        self.mask_path = mask_path
        self.backend = backend
        self.note = note
        self.mask_coverage = mask_coverage


def _save_mask(mask: np.ndarray, output_mask_path: str | None) -> str | None:
    if not output_mask_path:
        return None
    out = Path(output_mask_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(out)
    return str(out)


def _mask_coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return round(float(mask.mean() / 255.0) * 100.0, 2)


def _segment_with_mediapipe(image_rgb: np.ndarray, threshold: float) -> np.ndarray:
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError("mediapipe no disponible para segmentacion") from exc

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        results = segmenter.process(image_rgb)

    seg_mask = getattr(results, "segmentation_mask", None)
    if seg_mask is None:
        raise RuntimeError("modelo de segmentacion no devolvio mascara")

    return ((seg_mask >= float(threshold)).astype(np.uint8)) * 255


def _fallback_mask(image_shape: tuple[int, int, int]) -> np.ndarray:
    # Fallback determinista: elipse central aproximando torso/cuerpo.
    h, w = image_shape[:2]
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    left = int(w * 0.18)
    right = int(w * 0.82)
    top = int(h * 0.06)
    bottom = int(h * 0.98)
    draw.ellipse([left, top, right, bottom], fill=255)
    return np.array(mask_img, dtype=np.uint8)


def segment_person(
    image_path: str,
    output_mask_path: str | None = None,
    threshold: float = 0.35,
) -> SegmentationResult:
    """Segmenta persona con backend preentrenado y fallback estable.

    Prioriza MediaPipe Selfie Segmentation. Si falla, devuelve mascara
    heuristica para no romper el pipeline ni contratos de metadata.
    """
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"No se pudo leer imagen: {image_path}")

    try:
        image_rgb = np.array(Image.open(src).convert("RGB"))
    except Exception as exc:
        raise FileNotFoundError(f"No se pudo leer imagen: {image_path}") from exc

    backend = "mediapipe_selfie_segmentation"
    note = "Mascara de persona generada con modelo preentrenado"
    try:
        mask = _segment_with_mediapipe(image_rgb=image_rgb, threshold=threshold)
    except Exception as exc:
        mask = _fallback_mask(image_rgb.shape)
        backend = "fallback_ellipse"
        note = f"Fallback activado por error de segmentacion: {exc}"

    mask_path = _save_mask(mask, output_mask_path)

    return SegmentationResult(
        mask_path=mask_path,
        backend=backend,
        note=note,
        mask_coverage=_mask_coverage(mask),
    )
