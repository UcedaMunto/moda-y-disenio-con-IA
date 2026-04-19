from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_EXT = {".obj", ".fbx", ".glb", ".gltf"}
ARCHIVE_EXT = {".zip", ".rar"}


def _unique_path(target_dir: Path, file_name: str) -> Path:
    candidate = target_dir / file_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    idx = 1
    while True:
        next_candidate = target_dir / f"{stem}_{idx}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        idx += 1


def _copy_supported_files(source_dir: Path, telas_dir: Path, models_dir: Path) -> dict:
    copied_images: list[str] = []
    copied_models: list[str] = []

    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in IMAGE_EXT:
            out = _unique_path(telas_dir, path.name)
            shutil.copy2(path, out)
            copied_images.append(str(out))
        elif ext in MODEL_EXT:
            out = _unique_path(models_dir, path.name)
            shutil.copy2(path, out)
            copied_models.append(str(out))

    return {
        "copied_images": copied_images,
        "copied_models": copied_models,
    }


def _extract_zip(archive_path: Path, telas_dir: Path, models_dir: Path) -> dict:
    extracted_images: list[str] = []
    extracted_models: list[str] = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for member in members:
            ext = Path(member.filename).suffix.lower()
            base_name = Path(member.filename).name
            if ext in IMAGE_EXT:
                out = _unique_path(telas_dir, base_name)
                with zf.open(member, "r") as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_images.append(str(out))
            elif ext in MODEL_EXT:
                out = _unique_path(models_dir, base_name)
                with zf.open(member, "r") as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_models.append(str(out))

    return {
        "extracted_images": extracted_images,
        "extracted_models": extracted_models,
    }


def _extract_rar(archive_path: Path, telas_dir: Path, models_dir: Path) -> dict:
    unrar_bin = shutil.which("unrar")
    if not unrar_bin:
        return {
            "extracted_images": [],
            "extracted_models": [],
            "skipped": f"unrar not installed for {archive_path.name}",
        }

    extracted_images: list[str] = []
    extracted_models: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        command = [unrar_bin, "x", "-o+", str(archive_path), str(tmp_dir)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "extracted_images": [],
                "extracted_models": [],
                "skipped": f"unrar failed for {archive_path.name}",
            }

        for path in tmp_dir.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in IMAGE_EXT:
                out = _unique_path(telas_dir, path.name)
                shutil.copy2(path, out)
                extracted_images.append(str(out))
            elif ext in MODEL_EXT:
                out = _unique_path(models_dir, path.name)
                shutil.copy2(path, out)
                extracted_models.append(str(out))

    return {
        "extracted_images": extracted_images,
        "extracted_models": extracted_models,
        "skipped": None,
    }


def import_assets_from_directory(source_dir: str, data_dir: str = "data") -> dict:
    source = Path(source_dir)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    root = Path(data_dir)
    telas_dir = root / "raw" / "telas"
    models_dir = root / "raw" / "models"
    telas_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_supported_files(source, telas_dir, models_dir)

    archive_report = {
        "zip": [],
        "rar": [],
        "skipped": [],
    }

    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in ARCHIVE_EXT:
            continue

        if ext == ".zip":
            result = _extract_zip(path, telas_dir, models_dir)
            archive_report["zip"].append(
                {
                    "archive": str(path),
                    "extracted_images": result["extracted_images"],
                    "extracted_models": result["extracted_models"],
                }
            )
        elif ext == ".rar":
            result = _extract_rar(path, telas_dir, models_dir)
            archive_report["rar"].append(
                {
                    "archive": str(path),
                    "extracted_images": result["extracted_images"],
                    "extracted_models": result["extracted_models"],
                }
            )
            if result.get("skipped"):
                archive_report["skipped"].append(result["skipped"])

    return {
        "source": str(source),
        "copied_images": copied["copied_images"],
        "copied_models": copied["copied_models"],
        "archives": archive_report,
    }
