import json
from pathlib import Path

import numpy as np
from PIL import Image

from fase2_2.scripts.train_segmentation_lite import (
    evaluate_threshold,
    train_segmentation_lite,
    tune_threshold,
)


def _write_rgb(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((12, 12, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((12, 12), dtype=np.uint8)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = 255
    Image.fromarray(arr, mode="L").save(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")


def test_tune_threshold_selects_best(monkeypatch, tmp_path: Path) -> None:
    rows = [{"image_path": "a", "mask_path": "b"} for _ in range(3)]

    def _mock_eval(samples, threshold, model_config_path=None):
        return {"mean_iou": 1.0 - abs(threshold - 0.4), "n_samples": len(samples), "backend_count": {"mock": 3}}

    monkeypatch.setattr("fase2_2.scripts.train_segmentation_lite.evaluate_threshold", _mock_eval)

    tuned = tune_threshold(rows, [0.2, 0.4, 0.6])
    assert tuned["best_threshold"] == 0.4
    assert tuned["best_train_iou"] == 1.0


def test_evaluate_threshold_runs_with_mock_segment(monkeypatch, tmp_path: Path) -> None:
    img = tmp_path / "i.jpg"
    gt = tmp_path / "m.png"
    _write_rgb(img)
    _write_mask(gt, (3, 3, 9, 9))

    samples = [{"image_path": str(img), "mask_path": str(gt)}]

    class _MockRes:
        def __init__(self):
            self.backend = "mock"
            self.note = "ok"
            self.mask_coverage = 30.0

    def _mock_segment_person_v2(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35, model_config_path: str | None = None):
        _write_mask(Path(output_mask_path), (3, 3, 9, 9))
        return _MockRes()

    monkeypatch.setattr(
        "fase2_2.scripts.train_segmentation_lite.segment_person_v2",
        _mock_segment_person_v2,
    )

    out = evaluate_threshold(samples=samples, threshold=0.4)
    assert out["n_samples"] == 1
    assert out["mean_iou"] == 1.0
    assert out["backend_count"]["mock"] == 1


def test_train_segmentation_lite_writes_config(monkeypatch, tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    out_cfg = tmp_path / "model" / "seg_cfg.json"

    train_rows = [{"image_path": "a.jpg", "mask_path": "a.png"}, {"image_path": "b.jpg", "mask_path": "b.png"}]
    val_rows = [{"image_path": "c.jpg", "mask_path": "c.png"}]
    _write_jsonl(train_jsonl, train_rows)
    _write_jsonl(val_jsonl, val_rows)

    def _mock_tune_threshold(train_samples, candidate_thresholds, model_config_path=None):
        assert len(train_samples) == 2
        return {
            "best_threshold": 0.42,
            "best_train_iou": 0.77,
            "trials": [
                {"threshold": 0.35, "mean_iou": 0.70, "n_samples": 2},
                {"threshold": 0.42, "mean_iou": 0.77, "n_samples": 2},
            ],
        }

    def _mock_eval(samples, threshold, model_config_path=None):
        assert threshold == 0.42
        return {"mean_iou": 0.73, "n_samples": len(samples), "backend_count": {"mock": len(samples)}}

    monkeypatch.setattr("fase2_2.scripts.train_segmentation_lite.tune_threshold", _mock_tune_threshold)
    monkeypatch.setattr("fase2_2.scripts.train_segmentation_lite.evaluate_threshold", _mock_eval)

    result = train_segmentation_lite(
        train_jsonl=str(train_jsonl),
        val_jsonl=str(val_jsonl),
        output_config_path=str(out_cfg),
        candidate_thresholds=[0.35, 0.42],
    )

    assert result["status"] == "ok"
    assert result["best_threshold"] == 0.42
    assert result["train_best_iou"] == 0.77
    assert result["val_iou"] == 0.73
    assert out_cfg.exists()

    saved = json.loads(out_cfg.read_text(encoding="utf-8"))
    assert saved["threshold"] == 0.42
    assert saved["metrics"]["train_best_iou"] == 0.77
