from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from apps.api import main as api_main
from apps.api.main import app

client = TestClient(app)


def _write_test_image(path: Path, size: tuple[int, int] = (128, 192)) -> None:
    img = Image.new("RGB", size, (40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.ellipse((42, 18, 86, 62), fill=(220, 210, 200))
    draw.rectangle((44, 62, 84, 144), fill=(90, 120, 200))
    img.save(path)


def _mask_data_url(size: tuple[int, int] = (128, 192)) -> str:
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 20, 98, 180), fill=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _patch_lab_dirs(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    photos_dir = tmp_path / "fotos_personas"
    data_dir = tmp_path / "data"
    photos_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "BASE_DIR", tmp_path)
    monkeypatch.setattr(api_main, "FOTOS_PERSONAS_DIR", photos_dir)
    monkeypatch.setattr(api_main, "DATA_DIR", data_dir)
    monkeypatch.setattr(api_main, "HUMAN_SHAPE_LAB_DIR", data_dir / "processed" / "human_shape_lab")
    return photos_dir, data_dir


def test_human_shape_lab_bootstrap_and_status(tmp_path: Path, monkeypatch) -> None:
    photos_dir, _ = _patch_lab_dirs(tmp_path, monkeypatch)
    photo_path = photos_dir / "persona_01.jpg"
    _write_test_image(photo_path)

    import fase2_2.core.tryon.parsing_adapter as parsing_adapter

    def fake_segment(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35, model_config_path: str | None = None):
        if output_mask_path:
            mask = Image.new("L", (128, 192), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle((36, 18, 92, 185), fill=255)
            Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
            mask.save(output_mask_path)
        return SimpleNamespace(backend="mock_seg", note="mock note", mask_coverage=44.2)

    monkeypatch.setattr(parsing_adapter, "segment_person_v2", fake_segment)

    response = client.post("/human-shape-lab/bootstrap", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["imported_total"] == 1
    assert data["imported"][0]["sample_id"]
    assert data["imported"][0]["current_mask_url"]

    status = client.get("/human-shape-lab/status")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["lab"]["samples_total"] == 1
    assert status_payload["samples"][0]["image_name"] == "persona_01.jpg"


def test_human_shape_lab_save_mask_marks_sample_corrected(tmp_path: Path, monkeypatch) -> None:
    photos_dir, _ = _patch_lab_dirs(tmp_path, monkeypatch)
    photo_path = photos_dir / "persona_02.jpg"
    _write_test_image(photo_path)

    import fase2_2.core.tryon.parsing_adapter as parsing_adapter

    def fake_segment(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35, model_config_path: str | None = None):
        if output_mask_path:
            mask = Image.new("L", (128, 192), 0)
            ImageDraw.Draw(mask).rectangle((40, 24, 90, 184), fill=255)
            Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
            mask.save(output_mask_path)
        return SimpleNamespace(backend="mock_seg", note="mock note", mask_coverage=41.0)

    monkeypatch.setattr(parsing_adapter, "segment_person_v2", fake_segment)

    bootstrap = client.post("/human-shape-lab/bootstrap", json={})
    sample = bootstrap.json()["imported"][0]

    response = client.post(
        "/human-shape-lab/save-mask",
        json={
            "sample_id": sample["sample_id"],
            "mask_data_url": _mask_data_url(),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["sample"]["is_corrected"] is True


def test_human_shape_lab_train_endpoint(tmp_path: Path, monkeypatch) -> None:
    photos_dir, data_dir = _patch_lab_dirs(tmp_path, monkeypatch)
    photo_path = photos_dir / "persona_03.jpg"
    _write_test_image(photo_path)

    import fase2_2.core.tryon.parsing_adapter as parsing_adapter
    import fase2_2.scripts.prepare_segmentation_real_dataset as prep
    import fase2_2.scripts.train_segmentation_lite as train_mod

    def fake_segment(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35, model_config_path: str | None = None):
        if output_mask_path:
            mask = Image.new("L", (128, 192), 0)
            ImageDraw.Draw(mask).rectangle((38, 20, 88, 186), fill=255)
            Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
            mask.save(output_mask_path)
        return SimpleNamespace(backend="mock_seg", note="mock note", mask_coverage=46.0)

    def fake_build_manifest(source_images_dir: str, source_masks_dir: str, output_dir: str, max_samples=None, min_iou=0.05, **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_jsonl = out / "train.jsonl"
        val_jsonl = out / "val.jsonl"
        row = {"image_path": source_images_dir + "/persona_03.jpg", "mask_path": source_masks_dir + "/persona_03.png"}
        train_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
        val_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {
            "dataset": {"qualified_samples": 1, "train_count": 1, "val_count": 1, "output_dir": str(out)},
            "manifests": {"train": str(train_jsonl), "val": str(val_jsonl), "test": str(out / "test.jsonl"), "low_iou": str(out / "low_iou.jsonl")},
        }

    def fake_train(train_jsonl: str, val_jsonl: str, output_config_path: str, candidate_thresholds=None, base_model_config_path=None):
        out = Path(output_config_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model_name": "segmentation_lite_ft_v1", "threshold": 0.4}), encoding="utf-8")
        return {
            "status": "ok",
            "output_config_path": str(out),
            "best_threshold": 0.4,
            "train_best_iou": 0.72,
            "val_iou": 0.66,
            "n_train": 1,
            "n_val": 1,
        }

    monkeypatch.setattr(parsing_adapter, "segment_person_v2", fake_segment)
    monkeypatch.setattr(prep, "build_real_segmentation_manifest", fake_build_manifest)
    monkeypatch.setattr(train_mod, "train_segmentation_lite", fake_train)

    bootstrap = client.post("/human-shape-lab/bootstrap", json={})
    assert bootstrap.status_code == 200

    response = client.post("/human-shape-lab/train", json={"min_iou": 0.05})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["training"]["best_threshold"] == 0.4
    assert payload["lab"]["model_config_path"] == str((data_dir / "processed" / "human_shape_lab" / "models" / "segmentation_model_config.json").relative_to(data_dir.parent)).replace("\\", "/")
