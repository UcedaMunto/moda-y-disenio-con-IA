from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityTryOnResult:
    ok: bool
    mode: str
    status: str
    reason: str | None
    score: float | None
    iterations: int
    elapsed_seconds: float


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if not np.any(mask > 0):
        return np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest_idx] = 255
    return out


def _select_primary_body(seg_mask: np.ndarray) -> np.ndarray:
    mask = (seg_mask > 0).astype(np.uint8)
    if not np.any(mask):
        return (seg_mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8) * 255

    h, w = seg_mask.shape[:2]
    ref_x = w * 0.5
    ref_y = h * 0.45

    best_idx = 1
    best_score = -1.0
    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        cx, cy = centroids[idx]
        dx = (float(cx) - ref_x) / max(1.0, w)
        dy = (float(cy) - ref_y) / max(1.0, h)
        dist = float(np.sqrt(dx * dx + dy * dy))
        centrality = max(0.0, 1.0 - dist * 2.5)
        score = area * (0.35 + 0.65 * centrality)
        if score > best_score:
            best_score = score
            best_idx = idx

    body = np.zeros_like(seg_mask, dtype=np.uint8)
    body[labels == best_idx] = 255
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return body


def _foreground_body_from_image(person_rgb: np.ndarray, fallback_body: np.ndarray) -> np.ndarray:
    h, w = person_rgb.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    rect = (int(w * 0.15), int(h * 0.04), int(w * 0.70), int(h * 0.93))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(person_rgb, gc_mask, rect, bgd_model, fgd_model, 6, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        return fallback_body

    fg = _largest_component(fg)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    fallback_ratio = float(np.mean(fallback_body > 0))
    if fallback_ratio >= 0.03:
        prior = cv2.dilate(fallback_body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=1)
        constrained = cv2.bitwise_and(fg, prior)
        if float(np.mean(constrained > 0)) >= 0.03:
            fg = constrained

    if float(np.mean(fg > 0)) < 0.04:
        return fallback_body

    return fg


def _extract_source_garment(garment_rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = garment_rgba[..., :3].astype(np.uint8)
    alpha = garment_rgba[..., 3].astype(np.uint8)

    if float(np.mean(alpha < 250)) > 0.02:
        src_mask = (alpha > 20).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask_a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        gc_mask = np.full(gray.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        h, w = gray.shape[:2]
        rect = (int(w * 0.08), int(h * 0.05), int(w * 0.84), int(h * 0.9))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(rgb, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            gc_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            gc_fg = np.zeros_like(gray, dtype=np.uint8)

        # Choose the mask that is less likely to be full foreground.
        ratio_a = float(np.mean(mask_a > 0))
        ratio_b = float(np.mean(mask_b > 0))
        if abs(ratio_a - 0.35) <= abs(ratio_b - 0.35):
            src_mask = mask_a
        else:
            src_mask = mask_b

        if np.any(gc_fg > 0):
            src_mask = cv2.bitwise_and(src_mask, gc_fg)

    src_mask = _largest_component(src_mask)
    src_mask = cv2.morphologyEx(src_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    contours, _ = cv2.findContours(src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_mask = np.zeros_like(src_mask, dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        src_mask = cv2.bitwise_and(src_mask, hull_mask)
        src_mask = cv2.morphologyEx(src_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    src_mask = cv2.GaussianBlur(src_mask, (9, 9), 0)
    src_mask = (src_mask > 50).astype(np.uint8) * 255

    # When the source image is a full-body person photo, keep torso-like area to avoid ghost overlays.
    src_bbox = _bbox_from_mask(src_mask)
    if src_bbox is not None:
        x0, y0, x1, y1 = src_bbox
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        torso_window = np.zeros_like(src_mask, dtype=np.uint8)
        wx0 = int(x0 + bw * 0.14)
        wx1 = int(x1 - bw * 0.14)
        wy0 = int(y0 + bh * 0.12)
        wy1 = int(y0 + bh * 0.70)
        torso_window[max(0, wy0):min(src_mask.shape[0], wy1), max(0, wx0):min(src_mask.shape[1], wx1)] = 255
        torso_candidate = cv2.bitwise_and(src_mask, torso_window)
        if float(np.mean(torso_candidate > 0)) >= 0.015:
            src_mask = torso_candidate

    # Remove skin-like regions from source so face/arms are not projected as garment.
    hsv_src = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    src_skin_1 = cv2.inRange(hsv_src, np.array([0, 35, 60], dtype=np.uint8), np.array([25, 170, 255], dtype=np.uint8))
    src_skin_2 = cv2.inRange(hsv_src, np.array([160, 35, 60], dtype=np.uint8), np.array([180, 170, 255], dtype=np.uint8))
    src_skin = cv2.bitwise_or(src_skin_1, src_skin_2)
    src_skin = cv2.dilate(src_skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    src_mask = cv2.bitwise_and(src_mask, cv2.bitwise_not(src_skin))
    src_mask = _largest_component(src_mask)
    src_mask = cv2.morphologyEx(src_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)

    return rgb, src_mask


def _build_target_clothing_mask(person_rgb: np.ndarray, seg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, _ = seg_mask.shape[:2]
    body = _select_primary_body(seg_mask)

    # Keep upper body only for top-garment replacement.
    upper = np.zeros_like(body, dtype=np.uint8)
    upper[: int(h * 0.68), :] = body[: int(h * 0.68), :]

    hsv = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2HSV)
    skin_1 = cv2.inRange(hsv, np.array([0, 35, 60], dtype=np.uint8), np.array([25, 170, 255], dtype=np.uint8))
    skin_2 = cv2.inRange(hsv, np.array([160, 35, 60], dtype=np.uint8), np.array([180, 170, 255], dtype=np.uint8))
    skin = cv2.bitwise_or(skin_1, skin_2)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

    target = cv2.bitwise_and(upper, cv2.bitwise_not(skin))
    target = cv2.morphologyEx(target, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    target = _largest_component(target)

    # Relaxed torso fallback when skin detection removes too much signal.
    if float(np.mean(target > 0)) < 0.02:
        bbox = _bbox_from_mask(body)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            bw = max(1, x1 - x0 + 1)
            bh = max(1, y1 - y0 + 1)
            torso_x0 = int(x0 + bw * 0.12)
            torso_x1 = int(x1 - bw * 0.12)
            torso_y0 = int(y0 + bh * 0.20)
            torso_y1 = int(y0 + bh * 0.68)
            torso_mask = np.zeros_like(body, dtype=np.uint8)
            torso_mask[max(0, torso_y0):min(h, torso_y1), max(0, torso_x0):min(body.shape[1], torso_x1)] = 255
            target = cv2.bitwise_and(torso_mask, body)
            target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
            target = _largest_component(target)

    return target, body


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def _warp(mask_or_img: np.ndarray, matrix: np.ndarray, out_size: tuple[int, int], is_mask: bool) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border_mode = cv2.BORDER_CONSTANT
    border_value = 0 if is_mask else (0, 0, 0)
    return cv2.warpAffine(mask_or_img, matrix, out_size, flags=interpolation, borderMode=border_mode, borderValue=border_value)


def _score_fit(candidate_mask: np.ndarray, target_mask: np.ndarray, body_mask: np.ndarray) -> float:
    cand = candidate_mask > 0
    target = target_mask > 0
    body = body_mask > 0

    inter = np.logical_and(cand, target).sum()
    union = np.logical_or(cand, target).sum()
    iou = float(inter) / float(union + 1e-6)

    outside = np.logical_and(cand, np.logical_not(body)).sum()
    outside_ratio = float(outside) / float(cand.sum() + 1e-6)

    coverage = float(inter) / float(target.sum() + 1e-6)
    cand_area = float(cand.sum())
    target_area = float(target.sum())
    area_ratio = cand_area / float(target_area + 1e-6)

    size_penalty = abs(area_ratio - 1.0) * 0.35
    if area_ratio < 0.2:
        size_penalty += 0.8
    if area_ratio > 2.2:
        size_penalty += 0.35

    return iou * 0.55 + coverage * 0.55 - outside_ratio * 0.6 - size_penalty


def _orientation_penalty(angle_degrees: float, max_abs_angle: float) -> float:
    """Penalize large rotations so tops remain naturally upright."""
    ang = abs(float(angle_degrees))
    if ang <= 6.0:
        return 0.0
    norm = min(1.0, (ang - 6.0) / max(1e-6, (max_abs_angle - 6.0)))
    return norm * 0.18


def _compose_with_shading(person_rgb: np.ndarray, warped_rgb: np.ndarray, warped_mask: np.ndarray) -> np.ndarray:
    mask_f = warped_mask.astype(np.float32) / 255.0
    if np.max(mask_f) <= 0.0:
        return person_rgb

    # Feather mask to reduce hard edges.
    feather = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=3.0, sigmaY=3.0)
    feather = np.clip(feather * 1.15, 0.0, 1.0)

    # Add soft shading from the person luminance to preserve body depth cues.
    person_gray = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    smooth_gray = cv2.GaussianBlur(person_gray, (0, 0), sigmaX=9.0, sigmaY=9.0)
    shading = 0.55 + 0.65 * smooth_gray
    shaded_garment = np.clip(warped_rgb.astype(np.float32) * shading[..., None], 0, 255)

    out = person_rgb.astype(np.float32) * (1.0 - feather[..., None]) + shaded_garment * feather[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _compose_texture_fill(person_rgb: np.ndarray, texture_rgb: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    h, w = person_rgb.shape[:2]
    tex_h, tex_w = texture_rgb.shape[:2]
    if tex_h <= 0 or tex_w <= 0:
        return person_rgb

    tile_h = max(60, min(220, tex_h))
    tile_w = max(60, min(220, tex_w))
    tile = cv2.resize(texture_rgb, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
            y2 = min(y + tile_h, h)
            x2 = min(x + tile_w, w)
            canvas[y:y2, x:x2] = tile[: y2 - y, : x2 - x]

    return _compose_with_shading(person_rgb, canvas, target_mask)


def _is_texture_like_source(source_mask: np.ndarray) -> bool:
    h, w = source_mask.shape[:2]
    if h <= 0 or w <= 0:
        return True

    fg = source_mask > 0
    fill_ratio = float(np.mean(fg))
    if fill_ratio > 0.72:
        return True

    bbox = _bbox_from_mask(source_mask)
    if bbox is None:
        return True

    x0, y0, x1, y1 = bbox
    bw = float(x1 - x0 + 1) / float(w)
    bh = float(y1 - y0 + 1) / float(h)

    border_touch = 0
    border_touch += int(np.any(fg[0, :]))
    border_touch += int(np.any(fg[-1, :]))
    border_touch += int(np.any(fg[:, 0]))
    border_touch += int(np.any(fg[:, -1]))

    # Large border-touching masks are typically textures/screenshots, not isolated garments.
    if border_touch >= 2 and (bw > 0.85 or bh > 0.85):
        return True

    if border_touch >= 3:
        return True

    return False


def _ensure_fill_target(target_mask: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    if float(np.mean(target_mask > 0)) >= 0.01:
        return target_mask

    h, w = body_mask.shape[:2]
    bbox = _bbox_from_mask(body_mask)
    if bbox is None:
        return target_mask

    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)

    torso_y0 = int(y0 + bh * 0.18)
    torso_y1 = int(y0 + bh * 0.70)

    torso_mask = np.zeros((h, w), dtype=np.uint8)
    torso_mask[max(0, torso_y0):min(h, torso_y1), :] = body_mask[max(0, torso_y0):min(h, torso_y1), :]

    erode_w = max(9, int(bw * 0.08))
    if erode_w % 2 == 0:
        erode_w += 1
    torso_mask = cv2.erode(torso_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_w, 7)), iterations=1)
    torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    torso_mask = cv2.GaussianBlur(torso_mask, (9, 9), 0)
    torso_mask = (torso_mask > 60).astype(np.uint8) * 255
    return _largest_component(torso_mask)


def run_local_tryon_quality(
    person_image_path: str,
    garment_image_path: str,
    segmentation_mask_path: str,
    output_path: str,
    max_seconds: int = 180,
    force_quality: bool = False,
) -> QualityTryOnResult:
    start = time.perf_counter()

    person_rgb = np.array(Image.open(person_image_path).convert("RGB"), dtype=np.uint8)
    garment_rgba = np.array(Image.open(garment_image_path).convert("RGBA"), dtype=np.uint8)
    seg_mask = np.array(Image.open(segmentation_mask_path).convert("L"), dtype=np.uint8)

    target_mask, body_mask = _build_target_clothing_mask(person_rgb, seg_mask)
    if force_quality:
        body_mask = _foreground_body_from_image(person_rgb, body_mask)
        # In strict mode, prefer a stable torso target over noisy garment-detection masks.
        target_mask = _ensure_fill_target(np.zeros_like(target_mask, dtype=np.uint8), body_mask)

    if float(np.mean(target_mask > 0)) < 0.02 and not force_quality:
        return QualityTryOnResult(
            ok=False,
            mode="quality_search_fallback",
            status="fallback",
            reason="target_mask_too_small",
            score=None,
            iterations=0,
            elapsed_seconds=time.perf_counter() - start,
        )

    source_rgb, source_mask = _extract_source_garment(garment_rgba)
    if float(np.mean(source_mask > 0)) < 0.01 and not force_quality:
        return QualityTryOnResult(
            ok=False,
            mode="quality_search_fallback",
            status="fallback",
            reason="source_mask_too_small",
            score=None,
            iterations=0,
            elapsed_seconds=time.perf_counter() - start,
        )

    if float(np.mean(source_mask > 0)) < 0.01 and force_quality:
        fill_target = _ensure_fill_target(target_mask, body_mask)
        result_rgb = _compose_texture_fill(person_rgb, source_rgb, fill_target)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_rgb, mode="RGB").save(out_path)
        return QualityTryOnResult(
            ok=True,
            mode="quality_texture_fill_forced",
            status="ok",
            reason="source_mask_too_small_forced_fill",
            score=0.0,
            iterations=0,
            elapsed_seconds=time.perf_counter() - start,
        )

    if _is_texture_like_source(source_mask):
        fill_target = _ensure_fill_target(target_mask, body_mask)
        result_rgb = _compose_texture_fill(person_rgb, source_rgb, fill_target)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_rgb, mode="RGB").save(out_path)
        return QualityTryOnResult(
            ok=True,
            mode="quality_texture_fill",
            status="ok",
            reason="source_behaves_like_texture",
            score=0.0,
            iterations=0,
            elapsed_seconds=time.perf_counter() - start,
        )

    h, w = person_rgb.shape[:2]
    out_size = (w, h)

    target_bbox = _bbox_from_mask(target_mask)
    source_bbox = _bbox_from_mask(source_mask)
    if (target_bbox is None or source_bbox is None) and force_quality:
        fill_target = _ensure_fill_target(target_mask, body_mask)
        result_rgb = _compose_texture_fill(person_rgb, source_rgb, fill_target)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_rgb, mode="RGB").save(out_path)
        return QualityTryOnResult(
            ok=True,
            mode="quality_texture_fill_forced",
            status="ok",
            reason="missing_bbox_forced_fill",
            score=0.0,
            iterations=0,
            elapsed_seconds=time.perf_counter() - start,
        )

    if target_bbox is None or source_bbox is None:
        return QualityTryOnResult(
            ok=False,
            mode="quality_search_fallback",
            status="fallback",
            reason="missing_bbox",
            score=None,
            iterations=0,
            elapsed_seconds=time.perf_counter() - start,
        )

    tx0, ty0, tx1, ty1 = target_bbox
    sx0, sy0, sx1, sy1 = source_bbox
    target_w = max(1.0, float(tx1 - tx0 + 1))
    target_h = max(1.0, float(ty1 - ty0 + 1))
    source_w = max(1.0, float(sx1 - sx0 + 1))
    source_h = max(1.0, float(sy1 - sy0 + 1))

    target_center = np.array([(tx0 + tx1) * 0.5, (ty0 + ty1) * 0.5], dtype=np.float32)
    source_center = np.array([(sx0 + sx1) * 0.5, (sy0 + sy1) * 0.5], dtype=np.float32)

    # Base scale from bbox match, then search around it.
    base_scale = min(target_w / source_w, target_h / source_h)

    budget = float(max(20, min(1800, int(max_seconds))))
    deadline = start + budget

    best_score = -1e9
    best_matrix: np.ndarray | None = None
    iterations = 0
    max_abs_angle = 32.0

    # Coarse-to-fine parameter search.
    stages = [
        {
            "scale": np.linspace(base_scale * 0.72, base_scale * 1.48, 9),
            "rot": np.linspace(-22.0, 22.0, 9),
            "shift": np.linspace(-0.22, 0.22, 9),
        },
        {
            "scale": np.linspace(0.9, 1.1, 9),
            "rot": np.linspace(-8.0, 8.0, 9),
            "shift": np.linspace(-0.1, 0.1, 9),
        },
        {
            "scale": np.linspace(0.96, 1.04, 9),
            "rot": np.linspace(-3.0, 3.0, 7),
            "shift": np.linspace(-0.04, 0.04, 7),
        },
    ]

    stage_anchor_scale = 1.0
    stage_anchor_rot = 0.0
    stage_anchor_dx = 0.0
    stage_anchor_dy = 0.0

    for stage in stages:
        for scale_mul in stage["scale"]:
            if time.perf_counter() >= deadline:
                break
            for rot_delta in stage["rot"]:
                if time.perf_counter() >= deadline:
                    break
                for dx_mul in stage["shift"]:
                    if time.perf_counter() >= deadline:
                        break
                    for dy_mul in stage["shift"]:
                        if time.perf_counter() >= deadline:
                            break

                        scale = stage_anchor_scale * float(scale_mul)
                        angle = stage_anchor_rot + float(rot_delta)
                        if abs(angle) > max_abs_angle:
                            continue
                        dx = stage_anchor_dx + float(dx_mul) * target_w
                        dy = stage_anchor_dy + float(dy_mul) * target_h

                        matrix = cv2.getRotationMatrix2D((float(source_center[0]), float(source_center[1])), angle, scale)
                        matrix[0, 2] += float(target_center[0] - source_center[0] + dx)
                        matrix[1, 2] += float(target_center[1] - source_center[1] + dy)

                        warped_mask = _warp(source_mask, matrix, out_size, is_mask=True)
                        score = _score_fit(warped_mask, target_mask, body_mask) - _orientation_penalty(angle, max_abs_angle)
                        iterations += 1

                        if score > best_score:
                            best_score = score
                            best_matrix = matrix
                            stage_anchor_scale = scale
                            stage_anchor_rot = angle
                            stage_anchor_dx = dx
                            stage_anchor_dy = dy

    # Iterative refinement: spend the remaining budget exploring local neighborhood.
    if best_matrix is not None and time.perf_counter() < deadline:
        rng = np.random.default_rng(seed=42)
        current_scale = stage_anchor_scale
        current_angle = stage_anchor_rot
        current_dx = stage_anchor_dx
        current_dy = stage_anchor_dy
        step_scale = 0.03
        step_rot = 2.4
        step_shift_x = 0.03 * target_w
        step_shift_y = 0.03 * target_h

        while time.perf_counter() < deadline:
            proposed_scale = max(0.25, min(4.0, current_scale * (1.0 + rng.normal(0.0, step_scale))))
            proposed_angle = float(np.clip(current_angle + rng.normal(0.0, step_rot), -max_abs_angle, max_abs_angle))
            proposed_dx = current_dx + rng.normal(0.0, step_shift_x)
            proposed_dy = current_dy + rng.normal(0.0, step_shift_y)

            matrix = cv2.getRotationMatrix2D((float(source_center[0]), float(source_center[1])), proposed_angle, proposed_scale)
            matrix[0, 2] += float(target_center[0] - source_center[0] + proposed_dx)
            matrix[1, 2] += float(target_center[1] - source_center[1] + proposed_dy)

            warped_mask = _warp(source_mask, matrix, out_size, is_mask=True)
            score = _score_fit(warped_mask, target_mask, body_mask) - _orientation_penalty(proposed_angle, max_abs_angle)
            iterations += 1

            if score > best_score:
                best_score = score
                best_matrix = matrix
                current_scale = proposed_scale
                current_angle = proposed_angle
                current_dx = proposed_dx
                current_dy = proposed_dy
                step_scale = max(0.004, step_scale * 0.995)
                step_rot = max(0.35, step_rot * 0.995)
                step_shift_x = max(1.0, step_shift_x * 0.995)
                step_shift_y = max(1.0, step_shift_y * 0.995)
            else:
                step_scale = min(0.08, step_scale * 1.0008)
                step_rot = min(6.0, step_rot * 1.0008)
                step_shift_x = min(target_w * 0.1, step_shift_x * 1.0008)
                step_shift_y = min(target_h * 0.1, step_shift_y * 1.0008)

    if best_matrix is None and force_quality:
        fill_target = _ensure_fill_target(target_mask, body_mask)
        result_rgb = _compose_texture_fill(person_rgb, source_rgb, fill_target)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_rgb, mode="RGB").save(out_path)
        return QualityTryOnResult(
            ok=True,
            mode="quality_texture_fill_forced",
            status="ok",
            reason="no_candidate_forced_fill",
            score=0.0,
            iterations=iterations,
            elapsed_seconds=time.perf_counter() - start,
        )

    if best_matrix is None:
        return QualityTryOnResult(
            ok=False,
            mode="quality_search_fallback",
            status="fallback",
            reason="no_candidate",
            score=None,
            iterations=iterations,
            elapsed_seconds=time.perf_counter() - start,
        )

    warped_rgb = _warp(source_rgb, best_matrix, out_size, is_mask=False)
    warped_mask = _warp(source_mask, best_matrix, out_size, is_mask=True)

    # If the searched warp is effectively empty, force an explicit visible fill in strict mode.
    if force_quality and float(np.mean(warped_mask > 0)) < 0.01:
        fill_target = _ensure_fill_target(target_mask, body_mask)
        result_rgb = _compose_texture_fill(person_rgb, source_rgb, fill_target)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_rgb, mode="RGB").save(out_path)
        return QualityTryOnResult(
            ok=True,
            mode="quality_texture_fill_forced",
            status="ok",
            reason="warp_too_small_forced_fill",
            score=float(best_score),
            iterations=iterations,
            elapsed_seconds=time.perf_counter() - start,
        )

    result_rgb = _compose_with_shading(person_rgb, warped_rgb, warped_mask)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result_rgb, mode="RGB").save(out_path)

    elapsed = time.perf_counter() - start
    # Expose when optimization hits the configured time budget so UI can report it clearly.
    timed_out = elapsed >= (budget - 0.01)

    return QualityTryOnResult(
        ok=True,
        mode="quality_mask_search",
        status="ok",
        reason="time_budget_reached" if timed_out else None,
        score=float(best_score),
        iterations=iterations,
        elapsed_seconds=elapsed,
    )
