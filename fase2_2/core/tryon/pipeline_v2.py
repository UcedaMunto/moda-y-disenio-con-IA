from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from fase2_1.core.training.fit_regression.baseline import extract_feature_vector
from fase2_1.core.tryon.align import compute_scale
from fase2_1.core.tryon.contract import build_landmarks_contract, build_transform_contract
from fase2_1.core.tryon.overlay import build_upper_occlusion_mask, compose_tryon_layers
from fase2_1.core.tryon.pose import detect_pose
from fase2_1.core.tryon.schemas import TryOnRequest, TryOnResult
from fase2_2.core.training.fit_regression.offset import load_model, predict_transform
from fase2_2.core.tryon.parsing_adapter import segment_person_v2

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _is_raster_garment(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.suffix.lower() in RASTER_EXTENSIONS


def _predict_transform_or_default(
    landmarks_contract: dict,
    garment_type: str,
    offset_model_path: str | None,
    fallback_scale: float,
) -> tuple[dict, dict]:
    if not offset_model_path:
        return (
            {
                "scale": float(fallback_scale),
                "offset_x": 0.0,
                "offset_y": 0.0,
            },
            {"mode": "heuristic", "reason": "no_model"},
        )

    try:
        model = load_model(offset_model_path)
        row = {
            "landmarks_contract": landmarks_contract,
            "transform_contract": {"garment_type": garment_type},
        }
        feature_vector = extract_feature_vector(row)
        pred = predict_transform(model, feature_vector)

        # Clamp para evitar valores extremos por extrapolacion.
        scale = float(max(0.5, min(2.5, pred.get("scale", fallback_scale))))
        offset_x = float(max(-0.3, min(0.3, pred.get("offset_x", 0.0))))
        offset_y = float(max(-0.3, min(0.3, pred.get("offset_y", 0.0))))

        return (
            {
                "scale": scale,
                "offset_x": offset_x,
                "offset_y": offset_y,
            },
            {"mode": "model", "model_path": str(offset_model_path)},
        )
    except Exception as exc:
        return (
            {
                "scale": float(fallback_scale),
                "offset_x": 0.0,
                "offset_y": 0.0,
            },
            {"mode": "heuristic", "reason": f"model_failed: {exc}"},
        )


def _render_layered_overlay_with_offset(
    image_path: str,
    garment_path: str,
    output_path: str,
    scale: float,
    offset_x: float,
    offset_y: float,
    rotation_deg: float,
    segmentation_mask_path: str | None,
) -> dict:
    background = np.array(Image.open(image_path).convert("RGB"))
    garment_rgba = np.array(Image.open(garment_path).convert("RGBA"))

    bg_h, bg_w = background.shape[:2]
    g_h, g_w = garment_rgba.shape[:2]
    aspect = g_h / max(1, g_w)

    target_w = int(max(48, min(bg_w * 0.8, bg_w * 0.45 * max(0.5, min(scale, 2.0)))))
    target_h = int(max(48, target_w * aspect))

    resized_img = Image.fromarray(garment_rgba, mode="RGBA").resize((target_w, target_h), resample=Image.BILINEAR)
    if abs(rotation_deg) > 0.1:
        # Rotacion sobre centro para seguir inclinacion corporal.
        resized_img = resized_img.rotate(-float(rotation_deg), resample=Image.BILINEAR, expand=True)
    resized = np.array(resized_img, dtype=np.uint8)
    actual_h, actual_w = resized.shape[:2]

    base_x = int((bg_w - actual_w) / 2)
    base_y = int(bg_h * 0.2)

    x = int(base_x + (offset_x * bg_w))
    y = int(base_y + (offset_y * bg_h))

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
        "mode": "layered_overlay_v2",
        "overlay_position": {"x": x, "y": y},
        "overlay_base_position": {"x": base_x, "y": base_y},
        "overlay_size": {"w": actual_w, "h": actual_h},
        "offset_applied": {"x": offset_x, "y": offset_y},
        "rotation_deg": float(rotation_deg),
        "occlusion_enabled": occlusion_mask is not None,
    }


def _landmark_xy(landmarks, idx: int) -> tuple[float, float]:
    lm = landmarks.landmark[idx]
    return float(lm.x), float(lm.y)


def _compute_pose_guides(landmarks) -> dict:
    # Indices MediaPipe Pose: hombros 11/12, caderas 23/24.
    ls = _landmark_xy(landmarks, 11)
    rs = _landmark_xy(landmarks, 12)
    lh = _landmark_xy(landmarks, 23)
    rh = _landmark_xy(landmarks, 24)

    shoulder_dx = rs[0] - ls[0]
    shoulder_dy = rs[1] - ls[1]
    shoulder_width = float(math.hypot(shoulder_dx, shoulder_dy))
    shoulder_angle_deg = float(math.degrees(math.atan2(shoulder_dy, shoulder_dx)))

    hip_width = float(math.hypot(rh[0] - lh[0], rh[1] - lh[1]))
    shoulder_mid = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
    hip_mid = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)
    torso_height = float(max(1e-4, abs(hip_mid[1] - shoulder_mid[1])))

    return {
        "shoulder_width_norm": shoulder_width,
        "hip_width_norm": hip_width,
        "torso_height_norm": torso_height,
        "shoulder_angle_deg": shoulder_angle_deg,
    }


def _apply_pose_guides(
    pred_transform: dict,
    pose_guides: dict,
    enabled: bool,
    strength: float,
) -> dict:
    adjusted = dict(pred_transform)
    if not enabled:
        adjusted["rotation_deg"] = 0.0
        return adjusted

    s = float(max(0.0, min(1.0, strength)))
    shoulder_angle = float(pose_guides.get("shoulder_angle_deg", 0.0))
    shoulder_width = float(pose_guides.get("shoulder_width_norm", 0.22))
    hip_width = float(pose_guides.get("hip_width_norm", 0.20))

    # Mezcla entre prediccion y tamano sugerido por anchura hombros/cadera.
    scale_hint = 0.5 * (shoulder_width / 0.22) + 0.5 * (hip_width / 0.20)
    scale_hint = float(max(0.75, min(1.35, scale_hint)))

    base_scale = float(pred_transform.get("scale", 1.0))
    guided_scale = base_scale * scale_hint
    adjusted["scale"] = float(max(0.5, min(2.5, (1.0 - s) * base_scale + s * guided_scale)))

    # Rotacion limitada para evitar artefactos en poses extremas.
    adjusted["rotation_deg"] = float(max(-30.0, min(30.0, shoulder_angle * (0.6 + 0.4 * s))))
    return adjusted


def _detect_pose_with_backend(image_path: str, backend: str):
    try:
        return detect_pose(image_path, backend=backend)
    except TypeError:
        # Compatibilidad con tests/mocks antiguos que aceptan solo image_path.
        return detect_pose(image_path)


def run_tryon_v2(
    request: TryOnRequest,
    offset_model_path: str | None = None,
    segmentation_model_config_path: str | None = None,
) -> TryOnResult:
    """Pipeline Fase 2.2.

    Cambios respecto a 2.1:
    - Segmentacion via `segment_person_v2` con config opcional de modelo.
    - Transformacion de prenda usando modelo multisalida (scale + offset_x + offset_y).
    """
    landmarks = _detect_pose_with_backend(request.image_path, request.pose_backend)
    if landmarks is None:
        raise ValueError("No se detectaron landmarks de pose")

    heuristic_scale = compute_scale(landmarks, garment_type=request.garment_type)
    landmarks_contract = build_landmarks_contract(landmarks)
    pose_guides = _compute_pose_guides(landmarks)

    pred_transform, transform_source = _predict_transform_or_default(
        landmarks_contract=landmarks_contract,
        garment_type=request.garment_type,
        offset_model_path=offset_model_path,
        fallback_scale=heuristic_scale,
    )

    pred_transform = _apply_pose_guides(
        pred_transform=pred_transform,
        pose_guides=pose_guides,
        enabled=bool(request.apply_pose_guides),
        strength=float(request.pose_guide_strength),
    )

    transform_contract = build_transform_contract(
        scale=pred_transform["scale"],
        garment_type=request.garment_type,
    )
    transform_contract["translation_norm"] = {
        "x": pred_transform["offset_x"],
        "y": pred_transform["offset_y"],
    }
    transform_contract["offset_x"] = pred_transform["offset_x"]
    transform_contract["offset_y"] = pred_transform["offset_y"]
    transform_contract["rotation_deg"] = pred_transform.get("rotation_deg", 0.0)

    mask_output_path = str(Path(request.output_path).with_suffix(".person_mask.png"))
    seg = segment_person_v2(
        image_path=request.image_path,
        output_mask_path=mask_output_path,
        model_config_path=segmentation_model_config_path,
    )

    src = Path(request.image_path)
    out = Path(request.output_path)
    if not src.exists():
        raise FileNotFoundError(f"No se pudo leer imagen: {request.image_path}")

    render_meta = {"mode": "placeholder_copy", "occlusion_enabled": False}
    baseline = True
    note = "placeholder output"

    if _is_raster_garment(request.garment_path):
        try:
            render_meta = _render_layered_overlay_with_offset(
                image_path=request.image_path,
                garment_path=request.garment_path,
                output_path=request.output_path,
                scale=pred_transform["scale"],
                offset_x=pred_transform["offset_x"],
                offset_y=pred_transform["offset_y"],
                rotation_deg=pred_transform.get("rotation_deg", 0.0),
                segmentation_mask_path=seg.mask_path,
            )
            baseline = False
            note = "layered overlay output v2"
        except Exception:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            render_meta = {
                "mode": "placeholder_copy",
                "occlusion_enabled": False,
                "fallback_reason": "overlay_failed",
            }
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)

    return TryOnResult(
        status="ok",
        output_path=request.output_path,
        scale=pred_transform["scale"],
        meta={
            "baseline": baseline,
            "note": note,
            "garment_type": request.garment_type,
            "landmarks_contract": landmarks_contract,
            "transform_contract": transform_contract,
            "transform_source": transform_source,
            "pose_backend": request.pose_backend,
            "pose_guides_enabled": bool(request.apply_pose_guides),
            "pose_guide_strength": float(request.pose_guide_strength),
            "pose_guides": pose_guides,
            "segmentation_backend": seg.backend,
            "segmentation_note": seg.note,
            "segmentation_mask_path": seg.mask_path,
            "segmentation_mask_coverage": seg.mask_coverage,
            "render": render_meta,
        },
    )
