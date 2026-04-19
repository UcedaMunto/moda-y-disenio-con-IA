import json
from pathlib import Path

import pytest

from fase2_2.core.tryon.parsing_adapter import (
    _load_finetuned_segmentation_config,
    segment_person_v2,
)


class _MockSegmentationResult:
    def __init__(self):
        self.mask_path = "/tmp/mask.png"
        self.backend = "mediapipe_selfie_segmentation"
        self.note = "ok"
        self.mask_coverage = 42.0


def test_load_finetuned_config_empty() -> None:
    assert _load_finetuned_segmentation_config(None) == {}


def test_load_finetuned_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        _load_finetuned_segmentation_config("/tmp/no_existe_seg_cfg.json")


def test_load_finetuned_config_reads_json(tmp_path: Path) -> None:
    cfg = tmp_path / "seg_config.json"
    cfg.write_text(json.dumps({"threshold": 0.4, "model_name": "seg-ft-v1"}), encoding="utf-8")
    loaded = _load_finetuned_segmentation_config(str(cfg))
    assert loaded["threshold"] == 0.4
    assert loaded["model_name"] == "seg-ft-v1"


def test_segment_person_v2_uses_threshold_from_config(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "seg_config.json"
    cfg.write_text(
        json.dumps(
            {
                "threshold": 0.47,
                "model_name": "seg-ft-v1",
                "backend": "mediapipe_selfie_segmentation",
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def _mock_segment_person(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35):
        captured["threshold"] = threshold
        return _MockSegmentationResult()

    monkeypatch.setattr(
        "fase2_2.core.tryon.parsing_adapter.segment_person",
        _mock_segment_person,
    )

    res = segment_person_v2("img.jpg", model_config_path=str(cfg))

    assert captured["threshold"] == pytest.approx(0.47)
    assert "seg_config.json" in res.note
    assert "model_name=seg-ft-v1" in res.note


def test_segment_person_v2_uses_default_threshold_without_config(monkeypatch) -> None:
    captured = {}

    def _mock_segment_person(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35):
        captured["threshold"] = threshold
        return _MockSegmentationResult()

    monkeypatch.setattr(
        "fase2_2.core.tryon.parsing_adapter.segment_person",
        _mock_segment_person,
    )

    res = segment_person_v2("img.jpg", threshold=0.33)

    assert captured["threshold"] == pytest.approx(0.33)
    assert res.note == "ok"
