import shutil
from pathlib import Path

from .align import compute_scale
from .parsing import segment_person
from .pose import detect_pose
from .schemas import TryOnRequest, TryOnResult


def run_tryon(request: TryOnRequest) -> TryOnResult:
    """Pipeline baseline 2.1 para imagen estatica.

    Nota: en esta primera iteracion no aplica warp de prenda real;
    devuelve la imagen original como placeholder operativo para integrar API.
    """
    landmarks = detect_pose(request.image_path)
    if landmarks is None:
        raise ValueError("No se detectaron landmarks de pose")

    scale = compute_scale(landmarks)
    seg = segment_person(request.image_path)

    src = Path(request.image_path)
    out = Path(request.output_path)
    if not src.exists():
        raise FileNotFoundError(f"No se pudo leer imagen: {request.image_path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    return TryOnResult(
        status="ok",
        output_path=request.output_path,
        scale=scale,
        meta={
            "baseline": True,
            "note": "placeholder output",
            "segmentation_backend": seg.backend,
            "segmentation_note": seg.note,
        },
    )
