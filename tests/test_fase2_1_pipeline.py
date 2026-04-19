from pathlib import Path

from fase2_1.core.tryon import pipeline as pipeline_module
from fase2_1.core.tryon.schemas import TryOnRequest


class _FakeSegmentation:
    def __init__(self, mask_path: str, backend: str, note: str, mask_coverage: float):
        self.mask_path = mask_path
        self.backend = backend
        self.note = note
        self.mask_coverage = mask_coverage


def test_run_tryon_propagates_segmentation_metadata(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "person.jpg"
    out = tmp_path / "outputs" / "result.jpg"
    src.write_bytes(b"fake-image")

    monkeypatch.setattr(pipeline_module, "detect_pose", lambda _image_path: object())
    monkeypatch.setattr(pipeline_module, "compute_scale", lambda _landmarks, garment_type="other": 1.15)
    monkeypatch.setattr(
        pipeline_module,
        "build_landmarks_contract",
        lambda _landmarks: {"version": "1.0", "points": {}},
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_transform_contract",
        lambda scale, garment_type="other": {"version": "1.0", "scale": scale, "garment_type": garment_type},
    )

    def fake_segment(image_path: str, output_mask_path: str | None = None, threshold: float = 0.35):
        assert image_path == str(src)
        assert output_mask_path == str(out.with_suffix(".person_mask.png"))
        assert threshold == 0.35
        Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_mask_path).write_bytes(b"mask")
        return _FakeSegmentation(
            mask_path=output_mask_path,
            backend="mediapipe_selfie_segmentation",
            note="ok",
            mask_coverage=42.0,
        )

    monkeypatch.setattr(pipeline_module, "segment_person", fake_segment)

    req = TryOnRequest(
        image_path=str(src),
        garment_path="data/raw/models/TShirts.obj",
        output_path=str(out),
        garment_type="shirt",
    )
    result = pipeline_module.run_tryon(req)

    assert result.status == "ok"
    assert result.scale == 1.15
    assert Path(result.output_path).exists()
    assert result.meta["segmentation_backend"] == "mediapipe_selfie_segmentation"
    assert result.meta["segmentation_mask_path"] == str(out.with_suffix(".person_mask.png"))
    assert result.meta["segmentation_mask_coverage"] == 42.0
