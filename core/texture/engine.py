from pathlib import Path
import os
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

from core.texture.diffusion_engine import try_generate_with_diffusion
from core.texture.seamless import apply_seamless_to_files


def generate_texture_candidates(
    image_paths: list[str],
    project_id: str,
    n_candidates: int,
) -> list[str]:
    """Generates concrete candidate textures from reference images.

    This is a deterministic baseline generator for early phases while LoRA is integrated.
    """
    output_dir = Path("data/processed") / project_id / "textures"
    output_dir.mkdir(parents=True, exist_ok=True)

    diffusion_outputs = try_generate_with_diffusion(
        image_paths=image_paths,
        output_dir=output_dir,
        n_candidates=n_candidates,
    )
    if diffusion_outputs is not None:
        if os.getenv("TEXTURE_SEAMLESS_ENABLED", "1") == "1":
            return apply_seamless_to_files(diffusion_outputs)
        return diffusion_outputs

    if not image_paths:
        raise ValueError("At least one image path is required")

    sources: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            sources.append(image.convert("RGB"))

    def transform_variant(base: Image.Image, variant_index: int) -> Image.Image:
        mode = variant_index % 8
        candidate = base.copy()

        if mode == 0:
            candidate = ImageOps.mirror(candidate)
        elif mode == 1:
            candidate = candidate.rotate(90, expand=False)
        elif mode == 2:
            candidate = ImageEnhance.Color(candidate).enhance(1.2)
        elif mode == 3:
            candidate = ImageEnhance.Contrast(candidate).enhance(1.25)
        elif mode == 4:
            candidate = candidate.filter(ImageFilter.SMOOTH_MORE)
        elif mode == 5:
            candidate = ImageChops.offset(candidate, 128, 128)
        elif mode == 6:
            candidate = ImageEnhance.Brightness(candidate).enhance(1.15)
        else:
            candidate = ImageOps.flip(candidate)

        return candidate

    candidates: list[str] = []
    for idx in range(1, n_candidates + 1):
        source_image = sources[(idx - 1) % len(sources)]
        candidate_image = transform_variant(source_image, idx - 1)
        candidate_path = output_dir / f"candidate_{idx:02d}.png"
        candidate_image.save(candidate_path, format="PNG")
        candidates.append(str(candidate_path))

    if os.getenv("TEXTURE_SEAMLESS_ENABLED", "1") == "1":
        return apply_seamless_to_files(candidates)

    return candidates
