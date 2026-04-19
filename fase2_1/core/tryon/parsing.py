from __future__ import annotations

from pathlib import Path


class SegmentationResult:
    def __init__(self, mask_path: str | None, backend: str, note: str):
        self.mask_path = mask_path
        self.backend = backend
        self.note = note


def segment_person(image_path: str, output_mask_path: str | None = None) -> SegmentationResult:
    """Baseline segmentation stub for Fase 2.1.

    En esta iteracion crea solo contrato estable para integrar oclusion
    sin forzar aun una dependencia pesada de segmentacion.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"No se pudo leer imagen: {image_path}")

    return SegmentationResult(
        mask_path=output_mask_path,
        backend="stub",
        note="Segmentation baseline pendiente de modelo entrenado/preentrenado",
    )
