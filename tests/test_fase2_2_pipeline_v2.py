from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fase2_1.core.tryon.schemas import TryOnRequest
from fase2_2.core.tryon.pipeline_v2 import _predict_transform_or_default, run_tryon_v2


class _Landmark:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _PoseLandmarks:
    def __init__(self):
        self.landmark = [_Landmark(0.0, 0.0) for _ in range(33)]
        self.landmark[11] = _Landmark(0.3, 0.2)
        self.landmark[12] = _Landmark(0.7, 0.2)
        self.landmark[23] = _Landmark(0.35, 0.7)
        self.landmark[24] = _Landmark(0.65, 0.7)


class _MockSegResult:
    def __init__(self, mask_path: str):
        self.mask_path = mask_path
        self.backend = "mock"
        self.note = "ok"
        self.mask_coverage = 40.0


def _write_rgb(path: Path, value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((128, 96, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_rgba(path: Path, value: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((64, 48, 4), value, dtype=np.uint8)
    arr[..., 3] = 200
    Image.fromarray(arr, mode="RGBA").save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((128, 96), dtype=np.uint8)
    arr[20:120, 20:76] = 255
    Image.fromarray(arr, mode="L").save(path)


def test_predict_transform_or_default_no_model() -> None:
    landmarks_contract = {
        "points": {
            "left_shoulder": {"x": 0.3, "y": 0.2, "visibility": 1.0},
            "right_shoulder": {"x": 0.7, "y": 0.2, "visibility": 1.0},
            "left_hip": {"x": 0.35, "y": 0.7, "visibility": 1.0},
            "right_hip": {"x": 0.65, "y": 0.7, "visibility": 1.0},
        }
    }
    pred, source = _predict_transform_or_default(
        landmarks_contract=landmarks_contract,
        garment_type="shirt",
        offset_model_path=None,
        fallback_scale=1.2,
    )
    assert pred["scale"] == 1.2
    assert pred["offset_x"] == 0.0
    assert pred["offset_y"] == 0.0
    assert source["mode"] == "heuristic"


def test_run_tryon_v2_applies_model_offset(monkeypatch, tmp_path: Path) -> None:
    person = tmp_path / "person.jpg"
    garment = tmp_path / "garment.png"
    out = tmp_path / "out.png"
    mask = tmp_path / "mask.png"

    _write_rgb(person)
    _write_rgba(garment)
    _write_mask(mask)

    monkeypatch.setattr("fase2_2.core.tryon.pipeline_v2.detect_pose", lambda _: _PoseLandmarks())

    def _mock_segment_person_v2(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35, model_config_path: str | None = None):
        _write_mask(Path(output_mask_path))
        return _MockSegResult(mask_path=str(output_mask_path))

    monkeypatch.setattr("fase2_2.core.tryon.pipeline_v2.segment_person_v2", _mock_segment_person_v2)
    monkeypatch.setattr(
        "fase2_2.core.tryon.pipeline_v2._predict_transform_or_default",
        lambda landmarks_contract, garment_type, offset_model_path, fallback_scale: (
            {"scale": 1.1, "offset_x": 0.1, "offset_y": -0.05},
            {"mode": "model", "model_path": "mock_model.json"},
        ),
    )

    req = TryOnRequest(
        image_path=str(person),
        garment_path=str(garment),
        output_path=str(out),
        garment_type="shirt",
    )

    res = run_tryon_v2(req, offset_model_path="mock_model.json")

    assert res.status == "ok"
    assert Path(res.output_path).exists()
    assert res.meta["transform_source"]["mode"] == "model"
    assert res.meta["transform_contract"]["offset_x"] == 0.1
    assert res.meta["transform_contract"]["offset_y"] == -0.05
    assert res.meta["render"]["mode"] == "layered_overlay_v2"
