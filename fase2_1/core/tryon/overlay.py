import numpy as np


def alpha_overlay(background, overlay_rgba, x: int, y: int):
    """Composicion RGBA simple con alpha para MVP 2.1."""
    h, w = overlay_rgba.shape[:2]
    y_end = min(y + h, background.shape[0])
    x_end = min(x + w, background.shape[1])
    if y >= y_end or x >= x_end:
        return background

    crop_h = y_end - y
    crop_w = x_end - x

    roi = background[y:y_end, x:x_end]
    rgba = overlay_rgba[:crop_h, :crop_w]

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0
    blended = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    background[y:y_end, x:x_end] = blended
    return background


def alpha_overlay_with_occlusion(background, overlay_rgba, occlusion_mask, x: int, y: int):
    """Composicion RGBA respetando máscara de oclusión.

    Convención de `occlusion_mask` (uint8):
    - 0: zona libre, se puede renderizar prenda.
    - 255: zona ocluyente (brazo/cabello), se conserva fondo.
    """
    h, w = overlay_rgba.shape[:2]
    y_end = min(y + h, background.shape[0])
    x_end = min(x + w, background.shape[1])
    if y >= y_end or x >= x_end:
        return background

    crop_h = y_end - y
    crop_w = x_end - x

    roi = background[y:y_end, x:x_end]
    rgba = overlay_rgba[:crop_h, :crop_w]
    occ = occlusion_mask[y:y_end, x:x_end]

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0

    # Where occlusion=255 keep original background.
    occ_factor = (occ.astype(np.float32) / 255.0)[:, :, None]
    effective_alpha = alpha * (1.0 - occ_factor)

    blended = (effective_alpha * rgb + (1.0 - effective_alpha) * roi).astype(np.uint8)
    background[y:y_end, x:x_end] = blended
    return background


def build_upper_occlusion_mask(person_mask: np.ndarray, upper_ratio: float = 0.45) -> np.ndarray:
    """Construye mascara de oclusion priorizando capa frontal superior.

    La segmentacion completa de persona no puede usarse directamente como
    oclusion porque ocultaria toda la prenda. Este helper limita la oclusion
    a la zona superior donde suelen aparecer brazos/cabello en primer plano.
    """
    if person_mask.ndim != 2:
        raise ValueError("person_mask debe ser matriz 2D")

    h = person_mask.shape[0]
    split = max(1, min(h, int(h * float(upper_ratio))))

    out = np.zeros_like(person_mask, dtype=np.uint8)
    out[:split, :] = np.where(person_mask[:split, :] > 0, 255, 0).astype(np.uint8)
    return out


def compose_tryon_layers(
    background_rgb: np.ndarray,
    garment_rgba: np.ndarray,
    x: int,
    y: int,
    occlusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Composicion por orden de capas: fondo -> prenda -> oclusores."""
    canvas = background_rgb.copy()
    if occlusion_mask is None:
        return alpha_overlay(canvas, garment_rgba, x=x, y=y)
    return alpha_overlay_with_occlusion(canvas, garment_rgba, occlusion_mask=occlusion_mask, x=x, y=y)
