#!/usr/bin/env python3
"""Smoke runner para SAM 3D Body dentro de este repo.

Uso recomendado (con entorno sam_3d_body activo):
python scripts/run_sam3d_body_smoke.py \
  --checkpoint-path sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr-path sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM 3D Body demo on local photos")
    parser.add_argument("--repo-root", default=".", help="Root of this repository")
    parser.add_argument("--sam3d-root", default="sam-3d-body", help="Path to sam-3d-body clone")
    parser.add_argument("--image-folder", default="fotos_personas", help="Folder with input images")
    parser.add_argument(
        "--output-folder",
        default="data/processed/fase2_3/sam3d_body_smoke",
        help="Folder for output visualizations",
    )
    parser.add_argument("--checkpoint-path", required=True, help="Path to SAM 3D Body model.ckpt")
    parser.add_argument("--mhr-path", required=True, help="Path to MHR model asset")
    parser.add_argument(
        "--detector-name",
        default="",
        help="Detector name for demo.py. Use empty string to avoid detectron2 dependency.",
    )
    parser.add_argument(
        "--python-bin",
        default="python",
        help="Python binary from sam_3d_body environment",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sam3d_root = _resolve(repo_root, args.sam3d_root)
    image_folder = _resolve(repo_root, args.image_folder)
    output_folder = _resolve(repo_root, args.output_folder)
    checkpoint_path = _resolve(repo_root, args.checkpoint_path)
    mhr_path = _resolve(repo_root, args.mhr_path)

    if not sam3d_root.exists():
        raise FileNotFoundError(f"sam-3d-body root not found: {sam3d_root}")
    if not image_folder.exists():
        raise FileNotFoundError(f"image folder not found: {image_folder}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not mhr_path.exists():
        raise FileNotFoundError(f"mhr path not found: {mhr_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "demo.py",
        "--image_folder",
        str(image_folder),
        "--output_folder",
        str(output_folder),
        "--checkpoint_path",
        str(checkpoint_path),
        "--mhr_path",
        str(mhr_path),
        "--detector_name",
        args.detector_name,
    ]

    print("Running:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(sam3d_root), check=False)

    if result.returncode != 0:
        print(f"SAM 3D Body demo failed with exit code {result.returncode}")
        return result.returncode

    print(f"OK: outputs generated in {output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
