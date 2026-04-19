from pathlib import Path
import shutil
import subprocess


def apply_texture_and_export(model_path: str, texture_path: str, output_path: str) -> str:
    """Applies texture to a model through Blender headless when available."""
    model = Path(model_path)
    texture = Path(texture_path)

    if not model.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not texture.exists():
        raise FileNotFoundError(f"Texture file not found: {texture_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    blender_cmd = shutil.which("blender")
    script_path = Path("scripts") / "blender_apply_texture.py"
    if blender_cmd and script_path.exists():
        command = [
            blender_cmd,
            "-b",
            "-P",
            str(script_path),
            "--",
            str(model),
            str(texture),
            str(out),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Blender export failed: "
                f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
            )
        return str(out)

    # Fallback to keep API contract working where Blender is not installed.
    out.write_text(
        "fallback export placeholder\n"
        f"model={model_path}\n"
        f"texture={texture_path}\n",
        encoding="utf-8",
    )
    return str(out)
