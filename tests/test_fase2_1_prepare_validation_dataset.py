from pathlib import Path

import pytest

from fase2_1.scripts.prepare_validation_dataset import build_manifest, collect_images


def test_collect_images_filters_supported_extensions(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir(parents=True, exist_ok=True)
    (source / "a.jpg").write_bytes(b"x")
    (source / "b.PNG").write_bytes(b"x")
    (source / "c.txt").write_text("x", encoding="utf-8")

    images = collect_images(str(source))
    names = [p.name for p in images]

    assert names == ["a.jpg", "b.PNG"]


def test_collect_images_raises_for_missing_input(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        collect_images(str(missing))


def test_build_manifest_marks_range_and_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir(parents=True, exist_ok=True)
    first = source / "person_001.jpg"
    second = source / "nested" / "person_002.png"
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"x")
    second.write_bytes(b"x")

    payload = build_manifest(
        image_paths=[first, second],
        input_dir=str(source),
        min_images=2,
        max_images=5,
    )

    assert payload["status"] == "ok"
    assert payload["dataset"]["total_images"] == 2
    assert payload["dataset"]["in_target_range"] is True
    assert payload["images"][0]["id"] == "val_0001"
    assert payload["images"][1]["relative_path"] == "nested/person_002.png"


def test_build_manifest_out_of_range(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir(parents=True, exist_ok=True)
    image = source / "person_001.jpg"
    image.write_bytes(b"x")

    payload = build_manifest(
        image_paths=[image],
        input_dir=str(source),
        min_images=20,
        max_images=50,
    )

    assert payload["dataset"]["total_images"] == 1
    assert payload["dataset"]["in_target_range"] is False
