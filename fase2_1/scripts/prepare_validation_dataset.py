from __future__ import annotations

import argparse
import json
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"input_dir invalido: {input_dir}")

    images: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        images.append(path)
    return images


def build_manifest(
    image_paths: list[Path],
    input_dir: str,
    min_images: int = 20,
    max_images: int = 50,
) -> dict:
    root = Path(input_dir)
    rows = [
        {
            "id": f"val_{idx:04d}",
            "image_path": str(path),
            "relative_path": str(path.relative_to(root)),
        }
        for idx, path in enumerate(image_paths, start=1)
    ]

    total = len(rows)
    in_target_range = min_images <= total <= max_images

    return {
        "status": "ok",
        "dataset": {
            "input_dir": str(root),
            "total_images": total,
            "min_target": min_images,
            "max_target": max_images,
            "in_target_range": in_target_range,
        },
        "images": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara y valida set interno de validacion para Fase 2.1")
    parser.add_argument("--input-dir", required=True, help="Directorio con imagenes de validacion")
    parser.add_argument(
        "--manifest-path",
        default="data/processed/fase2_1_eval/validation_manifest.json",
        help="Ruta de salida del manifest JSON",
    )
    parser.add_argument("--min-images", type=int, default=20, help="Minimo esperado de imagenes")
    parser.add_argument("--max-images", type=int, default=50, help="Maximo esperado de imagenes")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Si se define, recorta el set a N imagenes (orden alfabetico)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Retorna error si el total queda fuera del rango [min_images, max_images]",
    )
    args = parser.parse_args()

    images = collect_images(args.input_dir)
    if args.sample_size is not None and args.sample_size > 0:
        images = images[: args.sample_size]

    payload = build_manifest(
        image_paths=images,
        input_dir=args.input_dir,
        min_images=args.min_images,
        max_images=args.max_images,
    )

    output = Path(args.manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["dataset"], indent=2))

    if args.strict and not payload["dataset"]["in_target_range"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
