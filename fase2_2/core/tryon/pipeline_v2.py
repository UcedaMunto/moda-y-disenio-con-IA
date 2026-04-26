from __future__ import annotations

import math
import shutil
from pathlib import Path

import cv2
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


def _detect_clothing_region_all(image_rgb: np.ndarray, seg_mask: np.ndarray) -> np.ndarray | None:
    """Detecta TODA la región de ropa dentro del cuerpo usando análisis de piel vs no-piel.
    
    Estrategia:
    - Identifica píxeles de PIEL usando análisis HSV
    - Todo lo que NO es piel dentro del cuerpo = ROPA
    - Retorna máscara binaria de ropa (incluye torso, pantalones, etc.)
    """
    if image_rgb is None or seg_mask is None:
        return None
    
    img_h, img_w = image_rgb.shape[:2]
    
    # Convierte a HSV
    img_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # Rango de color de PIEL en HSV (CORREGIDO con valores reales)
    # Piel real: H entre 160-180 o 0-20, S entre 50-120 (saturación alta), V entre 90-240
    # Se usa dos rangos para capturar pieles claras y oscuras
    
    # Máscara de piel (tonos de carne/naranja)
    # Rango 1: H(0-20), S(50-120), V(90-240)
    lower_skin_1 = np.array([0, 50, 90], dtype=np.uint8)
    upper_skin_1 = np.array([20, 120, 240], dtype=np.uint8)
    
    # Rango 2: H(160-180), S(50-120), V(90-240)
    lower_skin_2 = np.array([160, 50, 90], dtype=np.uint8)
    upper_skin_2 = np.array([180, 120, 240], dtype=np.uint8)
    
    skin_mask_1 = cv2.inRange(img_hsv, lower_skin_1, upper_skin_1)
    skin_mask_2 = cv2.inRange(img_hsv, lower_skin_2, upper_skin_2)
    skin_mask = cv2.bitwise_or(skin_mask_1, skin_mask_2)
    
    # Suaviza máscara de piel
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    skin_mask = cv2.GaussianBlur(skin_mask, (11, 11), 0)
    skin_mask = (skin_mask > 127).astype(np.uint8) * 255
    
    # Ropa = dentro del cuerpo (seg_mask) PERO NO es piel
    clothing_mask = cv2.bitwise_and(seg_mask, cv2.bitwise_not(skin_mask))
    
    # Limpia ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Suaviza bordes
    clothing_mask = cv2.GaussianBlur(clothing_mask, (15, 15), 0)
    clothing_mask = (clothing_mask > 100).astype(np.uint8) * 255
    
    return clothing_mask if np.any(clothing_mask > 0) else None


def _detect_garment_region(image_rgb: np.ndarray, seg_mask: np.ndarray) -> np.ndarray | None:
    """Detecta la región de la prenda dentro del cuerpo usando análisis de color HSV.
    
    Estrategia:
    - Convierte RGB a HSV
    - Busca píxeles con color uniforme (varianza baja) en la región superior del cuerpo
    - Retorna máscara binaria de la región detectada
    """
    if image_rgb is None or seg_mask is None:
        return None
    
    img_h, img_w = image_rgb.shape[:2]
    
    # Limita a región superior del cuerpo donde típicamente está la prenda (0-60% de altura)
    upper_bound = int(img_h * 0.60)
    roi_mask = np.zeros_like(seg_mask, dtype=np.uint8)
    roi_mask[:upper_bound, :] = seg_mask[:upper_bound, :]
    
    # Convierte a HSV para análisis de color
    img_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # Busca región de color relativamente uniforme (baja saturación o alta saturación coherente)
    # Detecta donde hay menos variación en color dentro del cuerpo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Calcula varianza local de saturación
    s_float = s.astype(np.float32)
    s_mean = cv2.boxFilter(s_float, -1, (21, 21))
    s_var = cv2.boxFilter(s_float**2, -1, (21, 21)) - s_mean**2
    
    # Áreas donde la varianza es baja probablemente son prenda uniforme
    garment_candidate = (s_var < 800).astype(np.uint8)
    
    # Aplica cierre morfológico para conectar regiones
    garment_candidate = cv2.morphologyEx(garment_candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Intersecta con máscara de cuerpo
    garment_mask = cv2.bitwise_and(garment_candidate, roi_mask)
    
    # Encuentra el contorno más grande (probablemente la prenda)
    contours, _ = cv2.findContours(garment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Toma el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Debe ser área significativa (al menos 3% de la imagen)
    if area < (img_h * img_w * 0.03):
        return None
    
    # Dibuja el contorno en una máscara nueva
    final_mask = np.zeros_like(seg_mask, dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], 0, 255, -1)
    
    # Suaviza bordes
    final_mask = cv2.GaussianBlur(final_mask, (11, 11), 0)
    final_mask = (final_mask > 127).astype(np.uint8) * 255
    
    return final_mask if np.any(final_mask > 0) else None


def _extract_garment_silhouette(
    image_rgb: np.ndarray,
    seg_mask: np.ndarray,
    garment_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict] | None:
    """Extrae la silueta de la prenda con análisis de contornos.
    
    Retorna:
    - garment_silhouette: máscara binaria de la silueta
    - stats: diccionario con bounding box y área
    """
    if garment_mask is None:
        garment_mask = _detect_garment_region(image_rgb, seg_mask)
    
    if garment_mask is None:
        return None
    
    # Encuentra contornos de la prenda
    contours, _ = cv2.findContours(garment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Toma el contorno más grande
    largest = max(contours, key=cv2.contourArea)
    
    # Aproxima contorno con poligonal para suavizar
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    # Obtiene bounding box
    x, y, w, h = cv2.boundingRect(approx)
    
    # Crea silueta
    silhouette = np.zeros_like(seg_mask, dtype=np.uint8)
    cv2.drawContours(silhouette, [approx], 0, 255, -1)
    
    # Suaviza
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    silhouette = cv2.GaussianBlur(silhouette, (9, 9), 0)
    silhouette = (silhouette > 127).astype(np.uint8) * 255
    
    stats = {
        "bbox": (x, y, w, h),
        "area": cv2.contourArea(largest),
        "contour_points": len(approx),
    }
    
    return silhouette, stats


def _apply_texture_to_clothing_region(
    background: np.ndarray,
    clothing_mask: np.ndarray,
    texture_path: str,
    scale: float = 1.5,
) -> np.ndarray | None:
    """Aplica textura a toda la región de ropa detectada con patrón repetido.
    
    Args:
        background: imagen RGB del fondo (persona)
        clothing_mask: máscara binaria de toda la ropa
        texture_path: ruta de la textura
        scale: factor de escala para mosaico de textura
    
    Retorna:
        imagen con textura aplicada a ropa, o None si falla
    """
    if not Path(texture_path).exists():
        return None
    
    try:
        texture_pil = Image.open(texture_path).convert("RGB")
    except Exception:
        return None
    
    img_h, img_w = background.shape[:2]
    
    # Redimensiona textura para crear patrón repetible
    tile_w = max(50, int(img_w * scale / 3))
    tile_h = max(50, int(img_h * scale / 3))
    texture_tile = texture_pil.resize((tile_w, tile_h), Image.LANCZOS)
    texture_arr = np.array(texture_tile, dtype=np.uint8)
    
    # Crea lienzo de textura para toda la imagen usando mosaico
    texture_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for y in range(0, img_h, tile_h):
        for x in range(0, img_w, tile_w):
            y_end = min(y + tile_h, img_h)
            x_end = min(x + tile_w, img_w)
            h = y_end - y
            w = x_end - x
            texture_canvas[y:y_end, x:x_end] = texture_arr[:h, :w]
    
    # Normaliza máscara de ropa (suaviza bordes)
    clothing_mask_smooth = cv2.GaussianBlur(clothing_mask, (21, 21), 0)
    clothing_mask_norm = clothing_mask_smooth.astype(np.float32) / 255.0
    
    # Aplica blending: 100% textura donde está la ropa, suave en los bordes
    result = background.copy().astype(np.float32)
    result = result * (1.0 - clothing_mask_norm[..., np.newaxis]) + \
             texture_canvas.astype(np.float32) * clothing_mask_norm[..., np.newaxis]
    
    return result.astype(np.uint8)


def _replace_garment_with_texture(
    background: np.ndarray,
    garment_mask: np.ndarray,
    texture_path: str,
    scale: float = 1.0,
) -> np.ndarray | None:
    """Reemplaza la región de la prenda con la textura proporcionada.
    
    Args:
        background: imagen RGB del fondo (persona)
        garment_mask: máscara binaria de la prenda a reemplazar
        texture_path: ruta de la textura
        scale: factor de escala para la textura
    
    Retorna:
        imagen con textura aplicada, o None si falla
    """
    if not Path(texture_path).exists():
        return None
    
    try:
        texture_pil = Image.open(texture_path).convert("RGB")
    except Exception:
        return None
    
    # Obtiene bounding box de la prenda
    y_coords, x_coords = np.where(garment_mask > 0)
    if len(y_coords) == 0:
        return None
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    garment_w = x_max - x_min + 1
    garment_h = y_max - y_min + 1
    
    # Redimensiona texture al tamaño de la prenda detectada
    target_w = int(garment_w * scale)
    target_h = int(garment_h * scale)
    
    texture_resized = texture_pil.resize((target_w, target_h), Image.LANCZOS)
    texture_arr = np.array(texture_resized, dtype=np.uint8)
    
    # Calcula posición para centrar la textura en la prenda
    offset_x = (garment_w - target_w) // 2
    offset_y = (garment_h - target_h) // 2
    
    # Crea resultado
    result = background.copy()
    
    # Aplica textura donde está la máscara
    local_mask = garment_mask[y_min:y_max+1, x_min:x_max+1]
    
    # Crea canvas para la textura posicionada
    canvas = np.zeros((garment_h, garment_w, 3), dtype=np.uint8)
    
    # Coloca la textura en el canvas
    if offset_y >= 0 and offset_x >= 0:
        tex_y_end = min(target_h, canvas.shape[0] - offset_y)
        tex_x_end = min(target_w, canvas.shape[1] - offset_x)
        canvas[offset_y:offset_y+tex_y_end, offset_x:offset_x+tex_x_end] = texture_arr[:tex_y_end, :tex_x_end]
    else:
        canvas = np.tile(texture_arr[0:1, 0:1], (canvas.shape[0], canvas.shape[1], 1))
    
    # Aplica blending entre la prenda original y la textura
    local_mask_norm = (local_mask.astype(np.float32) / 255.0)
    blended = (
        result[y_min:y_max+1, x_min:x_max+1].astype(np.float32) * (1 - local_mask_norm[..., np.newaxis] * 0.85) +
        canvas.astype(np.float32) * (local_mask_norm[..., np.newaxis] * 0.85)
    ).astype(np.uint8)
    
    result[y_min:y_max+1, x_min:x_max+1] = blended
    
    return result


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


def _render_clothing_removal_and_replacement(
    image_path: str,
    garment_path: str,
    output_path: str,
    segmentation_mask_path: str | None,
    scale: float = 1.5,
) -> dict:
    """NUEVO MODO: Detecta TODA la ropa en la imagen y la reemplaza con textura.
    
    Pasos:
    1. Reconoce la figura humana (MediaPipe + segmentación)
    2. Dentro del cuerpo, hace TRANSPARENTE toda la ropa (distinguiendo de piel)
    3. Coloca la textura de tela seleccionada en esa región
    
    Este modo es alternativo y proporciona:
    - Reemplazo completo de toda la ropa (no solo una prenda)
    - Preservación de piel expuesta (brazos, cuello, etc.)
    - Textura uniforme en toda la región de ropa
    """
    try:
        background = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        return {
            "mode": "clothing_replacement_error",
            "reason": f"cannot_load_image: {e}",
            "status": "error",
        }
    
    if not segmentation_mask_path or not Path(segmentation_mask_path).exists():
        # Fallback
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(background, mode="RGB").save(out)
        return {
            "mode": "clothing_replacement_fallback",
            "reason": "no_segmentation_mask",
            "status": "fallback",
        }
    
    try:
        seg_mask = np.array(Image.open(segmentation_mask_path).convert("L"), dtype=np.uint8)
    except Exception as e:
        return {
            "mode": "clothing_replacement_error",
            "reason": f"cannot_load_mask: {e}",
            "status": "error",
        }
    
    # Paso 1: Detecta TODA la región de ropa dentro del cuerpo
    # (usa análisis de color piel vs no-piel)
    try:
        clothing_mask = _detect_clothing_region_all(background, seg_mask)
    except Exception as e:
        return {
            "mode": "clothing_replacement_error",
            "reason": f"detection_error: {e}",
            "status": "error",
        }
    
    if clothing_mask is None:
        # Fallback
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(background, mode="RGB").save(out)
        return {
            "mode": "clothing_replacement_fallback",
            "reason": "no_clothing_detected",
            "status": "fallback",
        }
    
    # Paso 2: Aplica la textura de tela a toda la región de ropa
    try:
        result = _apply_texture_to_clothing_region(background, clothing_mask, garment_path, scale=scale)
    except Exception as e:
        return {
            "mode": "clothing_replacement_error",
            "reason": f"texture_error: {e}",
            "status": "error",
        }
    
    if result is None:
        # Fallback
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(background, mode="RGB").save(out)
        return {
            "mode": "clothing_replacement_fallback",
            "reason": "texture_application_failed",
            "status": "fallback",
        }
    
    # Guarda resultado
    try:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.astype(np.uint8), mode="RGB").save(out)
    except Exception as e:
        return {
            "mode": "clothing_replacement_error",
            "reason": f"cannot_save_output: {e}",
            "status": "error",
        }
    
    return {
        "mode": "clothing_removal_and_replacement",
        "status": "ok",
        "description": "Ropa detectada y reemplazada con textura",
        "scale_applied": float(scale),
    }


def _render_garment_detection_and_replacement(
    image_path: str,
    garment_path: str,
    output_path: str,
    segmentation_mask_path: str | None,
    scale: float = 1.0,
) -> dict:
    """Detecta la prenda actual en la imagen y la reemplaza con la textura.
    
    Este modo es alternativo a overlay y funciona mejor cuando:
    - La prenda tiene color uniforme/coherente
    - Se quiere reemplazar completamente la prenda existente
    - La textura se ajusta a la forma de la prenda detectada
    """
    background = np.array(Image.open(image_path).convert("RGB"))
    
    if not segmentation_mask_path or not Path(segmentation_mask_path).exists():
        # Fallback: copia la imagen si no hay máscara
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(background, mode="RGB").save(out)
        return {
            "mode": "garment_detection_fallback",
            "reason": "no_segmentation_mask",
            "status": "fallback",
        }
    
    seg_mask = np.array(Image.open(segmentation_mask_path).convert("L"), dtype=np.uint8)
    
    # Paso 1: Detecta la región de la prenda
    garment_mask = _detect_garment_region(background, seg_mask)
    if garment_mask is None:
        # Fallback: Si no detecta prenda, usa overlay tradicional
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(background, mode="RGB").save(out)
        return {
            "mode": "garment_detection_fallback",
            "reason": "no_garment_detected",
            "status": "fallback",
        }
    
    # Paso 2: Extrae silueta de la prenda
    silhouette_result = _extract_garment_silhouette(background, seg_mask, garment_mask)
    if silhouette_result is None:
        # Fallback
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(background, mode="RGB").save(out)
        return {
            "mode": "garment_detection_fallback",
            "reason": "silhouette_extraction_failed",
            "status": "fallback",
        }
    
    silhouette, stats = silhouette_result
    
    # Paso 3: Reemplaza la prenda con la textura
    result = _replace_garment_with_texture(background, silhouette, garment_path, scale=scale)
    if result is None:
        # Fallback
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(background, mode="RGB").save(out)
        return {
            "mode": "garment_detection_fallback",
            "reason": "texture_replacement_failed",
            "status": "fallback",
        }
    
    # Guarda resultado
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(out)
    
    return {
        "mode": "garment_detection_and_replacement",
        "status": "ok",
        "garment_stats": stats,
        "scale_applied": float(scale),
    }


def _render_layered_overlay_with_offset(
    image_path: str,
    garment_path: str,
    output_path: str,
    scale: float,
    offset_x: float,
    offset_y: float,
    rotation_deg: float,
    segmentation_mask_path: str | None,
    use_occlusion: bool,
    shape_guide_keypoints: dict | None,
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

    # Evita que la prenda quede casi fuera de cuadro por offsets extremos.
    min_visible_ratio = 0.65
    min_x = int(-actual_w * (1.0 - min_visible_ratio))
    max_x = int(bg_w - actual_w * min_visible_ratio)
    min_y = int(-actual_h * (1.0 - min_visible_ratio))
    max_y = int(bg_h - actual_h * min_visible_ratio)
    x = int(max(min_x, min(max_x, x)))
    y = int(max(min_y, min(max_y, y)))

    seg_mask = None
    occlusion_mask = None
    if segmentation_mask_path and Path(segmentation_mask_path).exists():
        seg_mask = np.array(Image.open(segmentation_mask_path).convert("L"), dtype=np.uint8)
        if use_occlusion:
            occlusion_mask = build_upper_occlusion_mask(seg_mask, upper_ratio=0.45)

    # Si la textura viene como cuadrado opaco, recortarla al cuerpo evita el "bloque" sobre la modelo.
    if seg_mask is not None and resized.ndim == 3 and resized.shape[2] == 4:
        bg_h_mask, bg_w_mask = seg_mask.shape[:2]
        body_local = np.zeros((actual_h, actual_w), dtype=np.uint8)

        src_x0 = max(0, x)
        src_y0 = max(0, y)
        src_x1 = min(bg_w_mask, x + actual_w)
        src_y1 = min(bg_h_mask, y + actual_h)
        if src_x1 > src_x0 and src_y1 > src_y0:
            dst_x0 = src_x0 - x
            dst_y0 = src_y0 - y
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            body_local[dst_y0:dst_y1, dst_x0:dst_x1] = np.where(
                seg_mask[src_y0:src_y1, src_x0:src_x1] > 0, 255, 0
            ).astype(np.uint8)

        if isinstance(shape_guide_keypoints, dict) and shape_guide_keypoints:
            torso_local = np.zeros((actual_h, actual_w), dtype=np.uint8)

            def _pt(name: str):
                p = shape_guide_keypoints.get(name)
                if not isinstance(p, dict):
                    return None
                try:
                    px = int(float(p.get("x")) * bg_w)
                    py = int(float(p.get("y")) * bg_h)
                    return px - x, py - y
                except Exception:
                    return None

            ls = _pt("left_shoulder")
            rs = _pt("right_shoulder")
            rh = _pt("right_hip")
            lh = _pt("left_hip")
            le = _pt("left_elbow")
            re = _pt("right_elbow")
            neck = _pt("neck_base")

            if all(v is not None for v in (ls, rs, rh, lh)):
                from PIL import ImageDraw, ImageFilter

                shoulder_w = max(8.0, float(math.hypot(rs[0] - ls[0], rs[1] - ls[1])))
                shoulder_y = 0.5 * (ls[1] + rs[1])
                hip_y = 0.5 * (lh[1] + rh[1])
                torso_h = max(24.0, abs(hip_y - shoulder_y))

                sleeve_out = 0.15 * shoulder_w
                sleeve_drop = 0.14 * torso_h
                armpit_drop = 0.30 * torso_h
                waist_inset = 0.14 * shoulder_w
                hem_drop = 0.14 * torso_h

                # Silueta tipo camiseta (no rectangular) guiada por hombros/cadera.
                l_outer = (int(ls[0] - sleeve_out), int(ls[1] + sleeve_drop))
                r_outer = (int(rs[0] + sleeve_out), int(rs[1] + sleeve_drop))
                l_armpit = (int(ls[0] + 0.10 * shoulder_w), int(ls[1] + armpit_drop))
                r_armpit = (int(rs[0] - 0.10 * shoulder_w), int(rs[1] + armpit_drop))
                lh_waist = (int(lh[0] + waist_inset), int(lh[1] + hem_drop))
                rh_waist = (int(rh[0] - waist_inset), int(rh[1] + hem_drop))

                shirt_poly = [l_outer, ls, rs, r_outer, r_armpit, rh_waist, lh_waist, l_armpit]

                m = Image.fromarray(torso_local, mode="L")
                draw = ImageDraw.Draw(m)
                draw.polygon(shirt_poly, fill=255)

                neck_center = neck if neck is not None else (
                    int((ls[0] + rs[0]) * 0.5),
                    int(min(ls[1], rs[1]) + 0.11 * torso_h),
                )
                nw = max(8, int(0.20 * shoulder_w))
                nh = max(6, int(0.14 * torso_h))
                draw.ellipse(
                    [
                        (neck_center[0] - nw, neck_center[1] - nh),
                        (neck_center[0] + nw, neck_center[1] + nh),
                    ],
                    fill=0,
                )

                m = m.filter(ImageFilter.MaxFilter(size=5))
                torso_local = np.array(m, dtype=np.uint8)
                # Interseccion con mascara de persona para mantener prenda sobre el cuerpo.
                body_local = np.where((torso_local > 0) & (body_local > 0), 255, 0).astype(np.uint8)

        alpha = resized[:, :, 3].astype(np.uint8)
        alpha_nonzero_ratio = float(np.count_nonzero(alpha)) / float(max(1, alpha.size))
        body_nonzero_ratio = float(np.count_nonzero(body_local)) / float(max(1, body_local.size))
        if body_nonzero_ratio > 0.02 and alpha_nonzero_ratio > 0.80:
            resized[:, :, 3] = np.minimum(alpha, body_local)

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


def _compute_pose_guides_from_shape_keypoints(keypoints: dict | None) -> dict | None:
    if not isinstance(keypoints, dict) or not keypoints:
        return None

    def _xy(name: str) -> tuple[float, float] | None:
        p = keypoints.get(name)
        if not isinstance(p, dict):
            return None
        try:
            return float(p.get("x")), float(p.get("y"))
        except Exception:
            return None

    ls = _xy("left_shoulder")
    rs = _xy("right_shoulder")
    lh = _xy("left_hip")
    rh = _xy("right_hip")
    neck = _xy("neck_base")
    hip_center = _xy("hip_center")

    if not (ls and rs and lh and rh):
        return None

    shoulder_dx = rs[0] - ls[0]
    shoulder_dy = rs[1] - ls[1]
    shoulder_width = float(math.hypot(shoulder_dx, shoulder_dy))
    shoulder_angle_deg = float(math.degrees(math.atan2(shoulder_dy, shoulder_dx)))
    hip_width = float(math.hypot(rh[0] - lh[0], rh[1] - lh[1]))

    shoulder_mid = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
    torso_anchor = neck or shoulder_mid
    torso_anchor_y = float(min(torso_anchor[1], shoulder_mid[1]))

    if hip_center is None:
        hip_center = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)
    torso_height = float(max(1e-4, abs(hip_center[1] - shoulder_mid[1])))

    return {
        "shoulder_width_norm": shoulder_width,
        "hip_width_norm": hip_width,
        "torso_height_norm": torso_height,
        "shoulder_angle_deg": shoulder_angle_deg,
        "anchor_x_norm": float(torso_anchor[0]),
        "anchor_y_norm": torso_anchor_y,
    }


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
    external_pose_guides = _compute_pose_guides_from_shape_keypoints(request.shape_guide_keypoints)
    if external_pose_guides:
        pose_guides = external_pose_guides

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
    if external_pose_guides and bool(request.apply_pose_guides):
        # Usa ancla del torso de Human Shape Lab para centrar la prenda en el cuerpo.
        anchor_x = float(external_pose_guides.get("anchor_x_norm", 0.5))
        anchor_y = float(external_pose_guides.get("anchor_y_norm", 0.22))
        guide_off_x = (anchor_x - 0.5) * 1.15
        guide_off_y = (anchor_y - 0.22) * 1.10
        pred_transform["offset_x"] = float(
            max(-0.2, min(0.2, 0.25 * pred_transform.get("offset_x", 0.0) + 0.75 * guide_off_x))
        )
        pred_transform["offset_y"] = float(
            max(-0.22, min(0.22, 0.25 * pred_transform.get("offset_y", 0.0) + 0.75 * guide_off_y))
        )
    if bool(request.apply_pose_guides):
        # Suaviza offsets para evitar desplazamientos excesivos del modelo en modo guiado.
        pred_transform["offset_x"] = float(max(-0.15, min(0.15, pred_transform.get("offset_x", 0.0) * 0.55)))
        pred_transform["offset_y"] = float(max(-0.18, min(0.18, pred_transform.get("offset_y", 0.0) * 0.60)))
    pred_transform["scale"] = float(
        max(0.5, min(2.5, pred_transform["scale"] * float(request.size_multiplier)))
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
            # Elige modo de renderizado basado en flags de detección (prioridad)
            if bool(request.remove_clothing_and_apply_texture):
                # NUEVO MODO: Remover TODA la ropa y aplicar textura
                render_meta = _render_clothing_removal_and_replacement(
                    image_path=request.image_path,
                    garment_path=request.garment_path,
                    output_path=request.output_path,
                    segmentation_mask_path=seg.mask_path,
                    scale=pred_transform["scale"],
                )
                baseline = False
                note = "clothing_removal_and_replacement output"
            elif bool(request.detect_and_replace_garment):
                render_meta = _render_garment_detection_and_replacement(
                    image_path=request.image_path,
                    garment_path=request.garment_path,
                    output_path=request.output_path,
                    segmentation_mask_path=seg.mask_path,
                    scale=pred_transform["scale"],
                )
                baseline = False
                note = "garment detection and replacement output"
            else:
                render_meta = _render_layered_overlay_with_offset(
                    image_path=request.image_path,
                    garment_path=request.garment_path,
                    output_path=request.output_path,
                    scale=pred_transform["scale"],
                    offset_x=pred_transform["offset_x"],
                    offset_y=pred_transform["offset_y"],
                    rotation_deg=pred_transform.get("rotation_deg", 0.0),
                    segmentation_mask_path=seg.mask_path,
                    use_occlusion=not bool(request.garment_in_front),
                    shape_guide_keypoints=request.shape_guide_keypoints,
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
            "size_multiplier": float(request.size_multiplier),
            "garment_in_front": bool(request.garment_in_front),
            "pose_guides": pose_guides,
            "external_pose_guides_used": bool(external_pose_guides),
            "segmentation_backend": seg.backend,
            "segmentation_note": seg.note,
            "segmentation_mask_path": seg.mask_path,
            "segmentation_mask_coverage": seg.mask_coverage,
            "render": render_meta,
        },
    )
