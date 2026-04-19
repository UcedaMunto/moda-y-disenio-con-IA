from math import ceil, sqrt
from pathlib import Path

from PIL import Image, ImageDraw


def build_preview_sheet(project_id: str, candidate_paths: list[str]) -> str:
    if not candidate_paths:
        raise ValueError("No candidate paths were provided")

    images: list[Image.Image] = []
    for path in candidate_paths:
        p = Path(path)
        if not p.exists() or not p.is_file():
            continue
        with Image.open(p) as img:
            images.append(img.convert("RGB").resize((512, 512)))

    if not images:
        raise FileNotFoundError("No readable candidate images were found")

    cols = max(1, ceil(sqrt(len(images))))
    rows = ceil(len(images) / cols)

    sheet = Image.new("RGB", (cols * 512, rows * 512), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * 512
        y = row * 512
        sheet.paste(image, (x, y))
        draw.rectangle((x + 6, y + 6, x + 90, y + 36), fill=(0, 0, 0))
        draw.text((x + 12, y + 12), f"#{idx + 1}", fill=(255, 255, 255))

    output_dir = Path("data/processed") / project_id / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candidates_sheet.png"
    sheet.save(output_path, format="PNG")
    return str(output_path)
