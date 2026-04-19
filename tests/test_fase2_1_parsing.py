from pathlib import Path

import numpy as np
from PIL import Image

from fase2_1.core.tryon import parsing


def test_segment_person_uses_mediapipe_backend_when_available(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "person.jpg"
    mask_path = tmp_path / "person_mask.png"
    Image.new("RGB", (8, 6), color=(120, 90, 70)).save(image_path)

    fake_mask = np.zeros((6, 8), dtype=np.uint8)
    fake_mask[:, :4] = 255

    def fake_segment(image_rgb: np.ndarray, threshold: float) -> np.ndarray:
        assert image_rgb.shape == (6, 8, 3)
        assert threshold == 0.5
        return fake_mask

    monkeypatch.setattr(parsing, "_segment_with_mediapipe", fake_segment)

    result = parsing.segment_person(str(image_path), output_mask_path=str(mask_path), threshold=0.5)

    assert result.backend == "mediapipe_selfie_segmentation"
    assert result.mask_path == str(mask_path)
    assert result.mask_coverage == 50.0
    assert mask_path.exists()


def test_segment_person_fallback_when_backend_fails(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "person.jpg"
    mask_path = tmp_path / "person_mask.png"
    Image.new("RGB", (10, 10), color=(20, 30, 40)).save(image_path)

    def fake_fail(_image_rgb: np.ndarray, _threshold: float) -> np.ndarray:
        raise RuntimeError("no backend")

    monkeypatch.setattr(parsing, "_segment_with_mediapipe", fake_fail)

    result = parsing.segment_person(str(image_path), output_mask_path=str(mask_path))

    assert result.backend == "fallback_ellipse"
    assert "Fallback activado" in result.note
    assert result.mask_path == str(mask_path)
    assert 0.0 < result.mask_coverage <= 100.0
    assert mask_path.exists()


def test_segment_person_missing_image_raises() -> None:
    try:
        parsing.segment_person("/tmp/does-not-exist.jpg")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True
