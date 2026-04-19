from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import main as api_main
from apps.api.main import app


client = TestClient(app)


def test_fase2_1_tryon_run_ok(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "person.jpg"
    out = tmp_path / "out" / "result.jpg"
    src.write_bytes(b"fake-image")

    def fake_run(request):
        assert request.garment_type == "shirt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-output")
        return {
            "status": "ok",
            "output_path": str(out),
            "scale": 1.23,
            "meta": {
                "baseline": True,
                "garment_type": request.garment_type,
                "landmarks_contract": {"version": "1.0", "points": {}},
                "transform_contract": {"version": "1.0", "scale": 1.23},
            },
        }

    monkeypatch.setattr(api_main, "_run_fase21_tryon", fake_run)

    response = client.post(
        "/fase2_1/tryon/run",
        json={
            "image_path": str(src),
            "garment_path": "data/raw/models/TShirts.obj",
            "output_path": str(out),
            "garment_type": "shirt",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["output_path"] == str(out)
    assert data["meta"]["garment_type"] == "shirt"
    assert data["meta"]["landmarks_contract"]["version"] == "1.0"
    assert data["meta"]["transform_contract"]["version"] == "1.0"
    assert out.exists()


def test_fase2_1_tryon_run_error(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "missing.jpg"
    out = tmp_path / "out" / "result.jpg"

    def fake_fail(_request):
        raise RuntimeError("boom")

    monkeypatch.setattr(api_main, "_run_fase21_tryon", fake_fail)

    response = client.post(
        "/fase2_1/tryon/run",
        json={
            "image_path": str(src),
            "garment_path": "data/raw/models/TShirts.obj",
            "output_path": str(out),
        },
    )

    assert response.status_code == 400
    assert "boom" in response.json()["detail"]
