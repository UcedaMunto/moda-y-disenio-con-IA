import json
from pathlib import Path

import numpy as np
from PIL import Image

from fase2_2.scripts.prepare_segmentation_real_dataset import (
    build_pair_index,
    build_real_segmentation_manifest,
    compute_iou,
    split_entries,
)


def _write_rgb(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: Path, filled: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((16, 16), dtype=np.uint8)
    if filled:
        arr[4:12, 4:12] = 255
    Image.fromarray(arr, mode="L").save(path)


def test_compute_iou_identity() -> None:
    a = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    b = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    assert compute_iou(a, b) == 1.0


def test_build_pair_index_collects_pairs_and_missing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    _write_rgb(images / "a.jpg")
    _write_rgb(images / "b.jpg")
    _write_mask(masks / "a.png")

    pairs, missing = build_pair_index(str(images), str(masks))

    assert len(pairs) == 1
    assert pairs[0]["image_rel"] == "a.jpg"
    assert pairs[0]["mask_rel"] == "a.png"
    assert len(missing) == 1


def test_split_entries_is_reproducible() -> None:
    entries = [{"id": f"x_{i}"} for i in range(10)]
    first = split_entries(entries, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=9)
    second = split_entries(entries, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=9)

    assert first == second
    assert len(first["train"]) == 6
    assert len(first["val"]) == 2
    assert len(first["test"]) == 2


def test_build_real_segmentation_manifest_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    images = tmp_path / "source_images"
    masks = tmp_path / "source_masks"
    out = tmp_path / "out"

    _write_rgb(images / "nested" / "p1.jpg", value=80)
    _write_rgb(images / "nested" / "p2.jpg", value=110)
    _write_mask(masks / "nested" / "p1.png", filled=True)
    _write_mask(masks / "nested" / "p2.png", filled=True)

    class _MockSegResult:
        def __init__(self, backend: str = "mock", mask_coverage: float = 25.0):
            self.backend = backend
            self.mask_coverage = mask_coverage

    def _mock_segment_person(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35):
        _write_mask(Path(output_mask_path), filled=True)
        return _MockSegResult()

    monkeypatch.setattr(
        "fase2_2.scripts.prepare_segmentation_real_dataset.segment_person",
        _mock_segment_person,
    )

    payload = build_real_segmentation_manifest(
        source_images_dir=str(images),
        source_masks_dir=str(masks),
        output_dir=str(out),
        max_samples=10,
        min_iou=0.5,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
    )

    assert payload["status"] == "ok"
    assert payload["dataset"]["paired_samples"] == 2
    assert payload["dataset"]["qualified_samples"] == 2
    assert payload["dataset"]["low_iou_samples"] == 0

    manifest_path = out / "manifest.json"
    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"
    test_path = out / "test.jsonl"
    low_iou_path = out / "low_iou.jsonl"

    assert manifest_path.exists()
    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()
    assert low_iou_path.exists()

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["dataset"]["qualified_samples"] == 2
