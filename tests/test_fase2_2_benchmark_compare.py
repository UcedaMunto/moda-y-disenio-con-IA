from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from fase2_2.scripts.benchmark_compare_v21_v22 import benchmark_compare, collect_images, percentile


def _write_rgb(path: Path, value: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((24, 24, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_collect_images_limit(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    _write_rgb(img_dir / "a.jpg")
    _write_rgb(img_dir / "b.png")
    _write_rgb(img_dir / "c.webp")

    rows = collect_images(str(img_dir), limit=2)
    assert len(rows) == 2


def test_percentile_basic() -> None:
    data = [10.0, 20.0, 30.0, 40.0]
    assert percentile(data, 50) == 25.0
    assert percentile(data, 0) == 10.0
    assert percentile(data, 100) == 40.0


def test_benchmark_compare_with_mocks(monkeypatch, tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    garment = tmp_path / "garment.png"

    _write_rgb(img_dir / "p1.jpg", value=100)
    _write_rgb(img_dir / "p2.jpg", value=140)
    _write_rgb(garment, value=200)

    class _MockRes:
        def __init__(self, scale: float, offset_x: float = 0.0, offset_y: float = 0.0):
            self._payload = {
                "status": "ok",
                "meta": {
                    "segmentation_backend": "mock",
                    "segmentation_mask_coverage": 35.0,
                    "transform_contract": {
                        "scale": scale,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                    },
                },
            }

        def model_dump(self):
            return self._payload

    def _mock_run_tryon(req):
        return _MockRes(scale=1.1)

    def _mock_run_tryon_v2(req, offset_model_path=None, segmentation_model_config_path=None):
        return _MockRes(scale=1.2, offset_x=0.08, offset_y=-0.03)

    monkeypatch.setattr("fase2_2.scripts.benchmark_compare_v21_v22.run_tryon", _mock_run_tryon)
    monkeypatch.setattr("fase2_2.scripts.benchmark_compare_v21_v22.run_tryon_v2", _mock_run_tryon_v2)

    payload = benchmark_compare(
        input_dir=str(img_dir),
        garment_path=str(garment),
        output_dir=str(out_dir),
        garment_type="shirt",
        limit=None,
        offset_model_path="model.json",
        segmentation_model_config_path="seg_cfg.json",
    )

    assert payload["summary"]["samples"] == 2
    assert payload["summary"]["v21"]["ok"] == 2
    assert payload["summary"]["v22"]["ok"] == 2

    paired = payload["results"]["paired"]
    assert len(paired) == 2
    assert paired[0]["v22_offset_x"] == 0.08
    assert paired[0]["v22_offset_y"] == -0.03
