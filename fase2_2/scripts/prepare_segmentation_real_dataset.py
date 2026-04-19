from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from fase2_1.core.tryon.parsing import segment_person

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_files(input_dir: str, valid_extensions: set[str]) -> list[Path]:
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directorio invalido: {input_dir}")

    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            out.append(path)
    return out


def _resolve_mask_for_image(image_path: Path, images_root: Path, masks_root: Path) -> Path | None:
    rel = image_path.relative_to(images_root)
    base = rel.with_suffix("")

    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = masks_root / f"{base}{ext}"
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in VALID_MASK_EXTENSIONS:
            return candidate

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
    missing: list[str] = []

    for idx, image_path in enumerate(images, start=1):
        mask_path = _resolve_mask_for_image(image_path, images_root, masks_root)
        if mask_path is None:
            missing.append(str(image_path))
            continue

        rel_image = image_path.relative_to(images_root)
        rel_mask = mask_path.relative_to(masks_root)
        pairs.append(
            {
                "id": f"real_seg_{idx:06d}",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_rel": str(rel_image),
                "mask_rel": str(rel_mask),
            }
        )

    return pairs, missing


def _read_binary_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr >= 127).astype(np.uint8)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError("Mascaras con dimensiones distintas")

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


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

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_real_segmentation_manifest(
    source_images_dir: str,
    source_masks_dir: str,
    output_dir: str,
    max_samples: int | None = 200,
    min_iou: float = 0.25,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    pairs, missing_masks = build_pair_index(source_images_dir, source_masks_dir)

    rng = random.Random(seed)
    selected = list(pairs)
    rng.shuffle(selected)
    if max_samples is not None and max_samples > 0:
        selected = selected[:max_samples]

    out = Path(output_dir)
    images_out = out / "images"
    masks_out = out / "masks"
    auto_out = out / "auto_masks"
    out.mkdir(parents=True, exist_ok=True)

    qualified: list[dict] = []
    low_iou: list[dict] = []

    for item in selected:
        image_src = Path(item["image_path"])
        mask_src = Path(item["mask_path"])

        image_rel = Path(item["image_rel"])
        mask_rel = Path(item["mask_rel"])

        image_dst = images_out / image_rel
        mask_dst = masks_out / mask_rel
        auto_rel = image_rel.with_suffix(".png")
        auto_dst = auto_out / auto_rel

        image_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        auto_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_src, image_dst)
        shutil.copy2(mask_src, mask_dst)

        auto_result = segment_person(str(image_dst), output_mask_path=str(auto_dst))
        auto_mask = _read_binary_mask(str(auto_dst))
        gt_mask = _read_binary_mask(str(mask_dst))
        iou = round(compute_iou(gt_mask, auto_mask), 6)

        sample = {
            "id": item["id"],
            "image_path": str(image_dst),
            "mask_path": str(mask_dst),
            "auto_mask_path": str(auto_dst),
            "image_rel": str(image_rel),
            "mask_rel": str(mask_rel),
            "iou_auto": iou,
            "auto_backend": auto_result.backend,
            "auto_mask_coverage": auto_result.mask_coverage,
            "quality_pass": bool(iou >= min_iou),
        }

        if sample["quality_pass"]:
            qualified.append(sample)
        else:
            low_iou.append(sample)

    splits = split_entries(
        qualified,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    _write_jsonl(out / "train.jsonl", splits["train"])
    _write_jsonl(out / "val.jsonl", splits["val"])
    _write_jsonl(out / "test.jsonl", splits["test"])
    _write_jsonl(out / "low_iou.jsonl", low_iou)

    payload = {
        "status": "ok",
        "dataset": {
            "source_images_dir": str(Path(source_images_dir)),
            "source_masks_dir": str(Path(source_masks_dir)),
            "output_dir": str(out),
            "seed": seed,
            "max_samples": max_samples,
            "min_iou": min_iou,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_images": len(collect_files(source_images_dir, VALID_IMAGE_EXTENSIONS)),
            "paired_samples": len(pairs),
            "selected_samples": len(selected),
            "qualified_samples": len(qualified),
            "low_iou_samples": len(low_iou),
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
            "low_iou": str(out / "low_iou.jsonl"),
        },
    }

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["manifest_path"] = str(manifest_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara subset local de dataset publico de segmentacion con validacion IoU"
    )
    parser.add_argument("--source-images-dir", required=True, help="Directorio de imagenes fuente")
    parser.add_argument("--source-masks-dir", required=True, help="Directorio de mascaras fuente")
    parser.add_argument(
        "--output-dir",
        default="data/processed/fase2_2_train/segmentation_real",
        help="Directorio de salida",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--min-iou", type=float, default=0.25)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = build_real_segmentation_manifest(
        source_images_dir=args.source_images_dir,
        source_masks_dir=args.source_masks_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        min_iou=args.min_iou,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(json.dumps(payload["dataset"], indent=2, ensure_ascii=False))

    if args.strict:
        data = payload["dataset"]
        if data["qualified_samples"] == 0 or data["missing_masks"] > 0:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
