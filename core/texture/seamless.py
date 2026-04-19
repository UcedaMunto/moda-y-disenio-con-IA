from pathlib import Path

from PIL import Image, ImageChops


def _blend_center_cross(image: Image.Image, band: int = 64) -> Image.Image:
    w, h = image.size
    cx = w // 2
    cy = h // 2

    out = image.copy()

    # Vertical seam blend band around center X
    left = image.crop((max(0, cx - band), 0, cx, h))
    right = image.crop((cx, 0, min(w, cx + band), h))
    if left.size == right.size and left.size[0] > 0:
        blended = Image.blend(left, right, alpha=0.5)
        out.paste(blended, (cx - left.size[0], 0))
        out.paste(blended, (cx, 0))

    # Horizontal seam blend band around center Y
    top = image.crop((0, max(0, cy - band), w, cy))
    bottom = image.crop((0, cy, w, min(h, cy + band)))
    if top.size == bottom.size and top.size[1] > 0:
        blended = Image.blend(top, bottom, alpha=0.5)
        out.paste(blended, (0, cy - top.size[1]))
        out.paste(blended, (0, cy))

    return out


def make_image_seamless(image: Image.Image) -> Image.Image:
    w, h = image.size
    shifted = ImageChops.offset(image, w // 2, h // 2)
    softened = _blend_center_cross(shifted, band=max(16, min(w, h) // 16))
    restored = ImageChops.offset(softened, -(w // 2), -(h // 2))
    return restored


def make_file_seamless(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Texture file not found: {path}")

    with Image.open(file_path) as image:
        seamless = make_image_seamless(image.convert("RGB"))
        seamless.save(file_path, format="PNG")

    return str(file_path)


def apply_seamless_to_files(paths: list[str]) -> list[str]:
    outputs: list[str] = []
    for item in paths:
        outputs.append(make_file_seamless(item))
    return outputs
