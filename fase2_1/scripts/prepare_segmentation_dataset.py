from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_files(input_dir: str, valid_extensions: set[str]) -> list[Path]:
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directorio invalido: {input_dir}")

    rows: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            rows.append(path)
    return rows


def _resolve_mask_for_image(image_path: Path, images_root: Path, masks_root: Path) -> Path | None:
    rel = image_path.relative_to(images_root)
    base = rel.with_suffix("")

    # Prefer same relative name with png mask.
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = masks_root / f"{base}{ext}"
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in VALID_MASK_EXTENSIONS:
            return candidate

    # Fallback: same folder, any file sharing stem.
    folder = masks_root / rel.parent
    if not folder.exists() or not folder.is_dir():
        return None

    for candidate in sorted(folder.glob(f"{base.name}.*")):
        if candidate.is_file() and candidate.suffix.lower() in VALID_MASK_EXTENSIONS:
            return candidate

    return None


def build_pair_index(images_dir: str, masks_dir: str) -> tuple[list[dict], list[str]]:
    images_root = Path(images_dir)
    masks_root = Path(masks_dir)
    images = collect_files(images_dir, VALID_IMAGE_EXTENSIONS)

    pairs: list[dict] = []
    missing_masks: list[str] = []

    for idx, image_path in enumerate(images, start=1):
        mask_path = _resolve_mask_for_image(image_path, images_root=images_root, masks_root=masks_root)
        if mask_path is None:
            missing_masks.append(str(image_path))
            continue

        rel_image = image_path.relative_to(images_root)
        rel_mask = mask_path.relative_to(masks_root)
        pairs.append(
            {
                "id": f"seg_{idx:05d}",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_rel": str(rel_image),
                "mask_rel": str(rel_mask),
            }
        )

    return pairs, missing_masks


def split_entries(
    entries: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    if not entries:
        return {"train": [], "val": [], "test": []}

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("La suma de ratios debe ser mayor a cero")

    # Normalize ratios for resilient usage.
    train_r = train_ratio / ratio_sum
    val_r = val_ratio / ratio_sum

    shuffled = list(entries)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    n_test = n - n_train - n_val

    if n > 0 and n_train == 0:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val : n_train + n_val + n_test]

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_segmentation_dataset_manifest(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    pairs, missing_masks = build_pair_index(images_dir=images_dir, masks_dir=masks_dir)
    splits = split_entries(
        pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_jsonl(out / "train.jsonl", splits["train"])
    _write_jsonl(out / "val.jsonl", splits["val"])
    _write_jsonl(out / "test.jsonl", splits["test"])

    payload = {
        "status": "ok",
        "dataset": {
            "images_dir": str(Path(images_dir)),
            "masks_dir": str(Path(masks_dir)),
            "output_dir": str(out),
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_images": len(collect_files(images_dir, VALID_IMAGE_EXTENSIONS)),
            "paired_samples": len(pairs),
            "missing_masks": len(missing_masks),
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
        },
        "missing_mask_images": missing_masks,
        "manifests": {
            "train": str(out / "train.jsonl"),
            "val": str(out / "val.jsonl"),
            "test": str(out / "test.jsonl"),
        },
    }

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["manifest_path"] = str(manifest_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara dataset de segmentacion (pairs + split) para Fase 2.1")
    parser.add_argument("--images-dir", required=True, help="Directorio con imagenes de persona")
    parser.add_argument("--masks-dir", required=True, help="Directorio con mascaras por imagen")
    parser.add_argument(
        "--output-dir",
        default="data/processed/fase2_1_train/segmentation",
        help="Directorio de salida para manifest y splits",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Retorna error si hay imagenes sin mascara o no hay pares validos",
    )
    args = parser.parse_args()

    payload = build_segmentation_dataset_manifest(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(json.dumps(payload["dataset"], indent=2, ensure_ascii=False))

    if args.strict and (payload["dataset"]["paired_samples"] == 0 or payload["dataset"]["missing_masks"] > 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
