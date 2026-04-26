from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from fase2_3.config.settings import Fase23Settings


def build_bootstrap_plan(settings: Fase23Settings, compatibility_report: dict) -> dict:
    verdict = compatibility_report["verdict"]
    next_steps = [
        f"git clone --depth 1 --branch {settings.opentryon_ref} {settings.opentryon_repo_url} {settings.opentryon_root}",
        f"cd {settings.opentryon_root}",
    ]

    if verdict["recommended_mode"] == "local_gpu_or_api":
        next_steps.extend(
            [
                "pip install -r requirements.txt",
                "pip install -e .",
                "python api_server.py",
            ]
        )
    else:
        next_steps.extend(
            [
                "pip install -r requirements.txt",
                "pip install -e .",
                "Configurar solo proveedores API o rutas CPU; evitar modelos locales CUDA-only.",
                "python api_server.py",
            ]
        )

    return {
        "opentryon_root": str(settings.opentryon_root),
        "clone_required": not settings.opentryon_root.exists(),
        "recommended_mode": verdict["recommended_mode"],
        "reason": verdict["reason"],
        "next_steps": next_steps,
    }


def clone_or_update_repo(settings: Fase23Settings, update: bool = False) -> dict:
    git_path = shutil.which("git")
    if not git_path:
        raise RuntimeError("git no esta disponible en el host")

    settings.vendor_root.mkdir(parents=True, exist_ok=True)

    if settings.opentryon_root.exists():
        if not update:
            return {
                "status": "exists",
                "path": str(settings.opentryon_root),
                "message": "OpenTryOn ya esta clonado. Usa --update para actualizar.",
            }

        subprocess.run(
            [git_path, "-C", str(settings.opentryon_root), "pull", "--ff-only"],
            check=True,
        )
        return {
            "status": "updated",
            "path": str(settings.opentryon_root),
            "message": "OpenTryOn actualizado.",
        }

    subprocess.run(
        [
            git_path,
            "clone",
            "--depth",
            "1",
            "--branch",
            settings.opentryon_ref,
            settings.opentryon_repo_url,
            str(settings.opentryon_root),
        ],
        check=True,
    )
    return {
        "status": "cloned",
        "path": str(settings.opentryon_root),
        "message": "OpenTryOn clonado en fase2_3/vendor/opentryon.",
    }