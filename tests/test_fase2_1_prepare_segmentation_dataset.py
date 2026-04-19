import json
from pathlib import Path

from fase2_1.scripts.prepare_segmentation_dataset import (
    build_pair_index,
    build_segmentation_dataset_manifest,
    split_entries,
)


def test_build_pair_index_collects_pairs_and_missing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    (images / "a.jpg").write_bytes(b"img")
    (images / "b.jpg").write_bytes(b"img")
    (masks / "a.png").write_bytes(b"mask")

    pairs, missing = build_pair_index(str(images), str(masks))

    assert len(pairs) == 1
    assert pairs[0]["image_rel"] == "a.jpg"
    assert pairs[0]["mask_rel"] == "a.png"
    assert len(missing) == 1
    assert missing[0].endswith("b.jpg")


def test_split_entries_is_reproducible() -> None:
    entries = [{"id": f"seg_{i}"} for i in range(10)]

    first = split_entries(entries, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=7)
    second = split_entries(entries, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=7)

    assert first == second
    assert len(first["train"]) == 6
    assert len(first["val"]) == 2
    assert len(first["test"]) == 2


def test_build_segmentation_dataset_manifest_writes_outputs(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    out = tmp_path / "out"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    (images / "nested").mkdir(parents=True, exist_ok=True)
    (masks / "nested").mkdir(parents=True, exist_ok=True)

    (images / "nested" / "p1.jpg").write_bytes(b"img")
    (images / "nested" / "p2.jpg").write_bytes(b"img")
    (masks / "nested" / "p1.png").write_bytes(b"mask")
    (masks / "nested" / "p2.png").write_bytes(b"mask")

    payload = build_segmentation_dataset_manifest(
        images_dir=str(images),
        masks_dir=str(masks),
        output_dir=str(out),
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
    )

    assert payload["status"] == "ok"
    assert payload["dataset"]["paired_samples"] == 2
    assert payload["dataset"]["missing_masks"] == 0

    manifest_path = out / "manifest.json"
    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"
    test_path = out / "test.jsonl"

    assert manifest_path.exists()
    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["dataset"]["paired_samples"] == 2
