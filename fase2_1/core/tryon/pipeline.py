import shutil
from pathlib import Path

from .align import compute_scale
from .contract import build_landmarks_contract, build_transform_contract
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

    scale = compute_scale(landmarks, garment_type=request.garment_type)
    landmarks_contract = build_landmarks_contract(landmarks)
    transform_contract = build_transform_contract(scale=scale, garment_type=request.garment_type)
    mask_output_path = str(Path(request.output_path).with_suffix(".person_mask.png"))
    seg = segment_person(request.image_path, output_mask_path=mask_output_path)

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
            "garment_type": request.garment_type,
            "landmarks_contract": landmarks_contract,
            "transform_contract": transform_contract,
            "segmentation_backend": seg.backend,
            "segmentation_note": seg.note,
            "segmentation_mask_path": seg.mask_path,
            "segmentation_mask_coverage": seg.mask_coverage,
        },
    )
