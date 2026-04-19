import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .align import compute_scale
from .contract import build_landmarks_contract, build_transform_contract
from .overlay import build_upper_occlusion_mask, compose_tryon_layers
from .parsing import segment_person
from .pose import detect_pose
from .schemas import TryOnRequest, TryOnResult


RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _is_raster_garment(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.suffix.lower() in RASTER_EXTENSIONS


def _render_layered_overlay(
    image_path: str,
    garment_path: str,
    output_path: str,
    scale: float,
    segmentation_mask_path: str | None,
) -> dict:
    background = np.array(Image.open(image_path).convert("RGB"))
    garment_rgba = np.array(Image.open(garment_path).convert("RGBA"))

    bg_h, bg_w = background.shape[:2]
    g_h, g_w = garment_rgba.shape[:2]
    aspect = g_h / max(1, g_w)

    target_w = int(max(48, min(bg_w * 0.8, bg_w * 0.45 * max(0.5, min(scale, 2.0)))))
    target_h = int(max(48, target_w * aspect))

    resized = np.array(
        Image.fromarray(garment_rgba, mode="RGBA").resize((target_w, target_h), resample=Image.BILINEAR),
        dtype=np.uint8,
    )

    x = int((bg_w - target_w) / 2)
    y = int(bg_h * 0.2)

    occlusion_mask = None
    if segmentation_mask_path and Path(segmentation_mask_path).exists():
        seg_mask = np.array(Image.open(segmentation_mask_path).convert("L"), dtype=np.uint8)
        occlusion_mask = build_upper_occlusion_mask(seg_mask, upper_ratio=0.45)

    composed = compose_tryon_layers(
        background_rgb=background,
        garment_rgba=resized,
        x=x,
        y=y,
        occlusion_mask=occlusion_mask,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composed, mode="RGB").save(out)
    return {
        "mode": "layered_overlay",
        "overlay_position": {"x": x, "y": y},
        "overlay_size": {"w": target_w, "h": target_h},
        "occlusion_enabled": occlusion_mask is not None,
    }


def run_tryon(request: TryOnRequest) -> TryOnResult:
    """Pipeline baseline 2.1 para imagen estatica.

    Si `garment_path` es imagen raster, aplica composicion por capas.
    Si no, conserva fallback placeholder para mantener compatibilidad.
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

    render_meta = {"mode": "placeholder_copy", "occlusion_enabled": False}
    baseline = True
    note = "placeholder output"
    if _is_raster_garment(request.garment_path):
        try:
            render_meta = _render_layered_overlay(
                image_path=request.image_path,
                garment_path=request.garment_path,
                output_path=request.output_path,
                scale=scale,
                segmentation_mask_path=seg.mask_path,
            )
            baseline = False
            note = "layered overlay output"
        except Exception:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            render_meta = {"mode": "placeholder_copy", "occlusion_enabled": False, "fallback_reason": "overlay_failed"}
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)

    return TryOnResult(
        status="ok",
        output_path=request.output_path,
        scale=scale,
        meta={
            "baseline": baseline,
            "note": note,
            "garment_type": request.garment_type,
            "landmarks_contract": landmarks_contract,
            "transform_contract": transform_contract,
            "segmentation_backend": seg.backend,
            "segmentation_note": seg.note,
            "segmentation_mask_path": seg.mask_path,
            "segmentation_mask_coverage": seg.mask_coverage,
            "render": render_meta,
        },
    )
