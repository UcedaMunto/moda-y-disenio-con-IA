from pathlib import Path


def list_project_candidates(project_id: str) -> list[str]:
    texture_dir = Path("data/processed") / project_id / "textures"
    if not texture_dir.exists():
        return []

    files = sorted(
        [item for item in texture_dir.glob("*.png") if item.is_file()]
    )
    return [str(item) for item in files]
