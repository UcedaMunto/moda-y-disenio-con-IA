from pathlib import Path
from PIL import Image, ImageOps


def preprocess_images(image_paths: list[str], project_id: str) -> list[str]:
    """Validates and normalizes input images for downstream generation."""
    valid_paths: list[str] = []
    missing: list[str] = []

    for path in image_paths:
        p = Path(path)
        if p.exists() and p.is_file():
            valid_paths.append(str(p))
        else:
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            f"Project '{project_id}' has missing input files: {missing}"
        )

    output_dir = Path("data/processed") / project_id / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_paths: list[str] = []
    for idx, path in enumerate(valid_paths, start=1):
        with Image.open(path) as image:
            # Normalize orientation and resize to a stable working resolution.
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            fitted = ImageOps.fit(normalized, (1024, 1024), method=Image.Resampling.LANCZOS)
            output_path = output_dir / f"ref_{idx:02d}.png"
            fitted.save(output_path, format="PNG")
            normalized_paths.append(str(output_path))

    return normalized_paths
