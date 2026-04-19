from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageOps


def generate_3d_preview(
    model_path: str,
    texture_path: str,
    output_path: str,
) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    blender_cmd = shutil.which("blender")
    script_path = Path("scripts") / "blender_render_preview.py"

    if blender_cmd and script_path.exists():
        command = [
            blender_cmd,
            "-b",
            "-P",
            str(script_path),
            "--",
            model_path,
            texture_path,
            str(out),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and out.exists():
            return str(out)

    # Fallback: provide texture-based preview if Blender is unavailable.
    texture = Path(texture_path)
    if not texture.exists():
        raise FileNotFoundError(f"Texture file not found: {texture_path}")

    with Image.open(texture) as image:
        preview = ImageOps.fit(image.convert("RGB"), (1024, 1024))
        preview.save(out, format="PNG")

    return str(out)
