from __future__ import annotations

import json
from pathlib import Path

from fase2_1.core.tryon.parsing import SegmentationResult, segment_person


def _load_finetuned_segmentation_config(model_config_path: str | None) -> dict:
    """Carga config de modelo fine-tuneado para segmentacion.

    Formato esperado (JSON):
    {
      "backend": "mediapipe_selfie_segmentation",
      "threshold": 0.4,
      "model_name": "seg-ft-v1"
    }
    """
    if not model_config_path:
        return {}

    src = Path(model_config_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"No existe config de segmentacion: {model_config_path}")

    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config de segmentacion invalida: se esperaba objeto JSON")
    return data


def segment_person_v2(
    image_path: str,
    output_mask_path: str | None = None,
    threshold: float = 0.35,
    model_config_path: str | None = None,
) -> SegmentationResult:
    """Segmentacion compatible con Fase 2.2 usando config de fine-tuning.

    Estrategia:
    - Si existe `model_config_path`, toma `threshold` desde config (si existe).
    - Ejecuta segmentacion base de Fase 2.1 para mantener comportamiento robusto.
    - Anota metadata en `note` para trazabilidad del experimento.
    """
    config = _load_finetuned_segmentation_config(model_config_path)
    effective_threshold = float(config.get("threshold", threshold))

    result = segment_person(
        image_path=image_path,
        output_mask_path=output_mask_path,
        threshold=effective_threshold,
    )

    model_name = config.get("model_name")
    backend_hint = config.get("backend")

    if model_config_path:
        suffix = f" | config={Path(model_config_path).name}"
        if model_name:
            suffix += f" model_name={model_name}"
        if backend_hint:
            suffix += f" backend_hint={backend_hint}"
        result.note = f"{result.note}{suffix}"

    return result
