from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import main as api_main
from apps.api.main import app


client = TestClient(app)


def test_fase2_1_tryon_batch_ok(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "report.json"

    def fake_batch(_request):
        report.write_text("{}", encoding="utf-8")
        return {
            "status": "ok",
            "summary": {
                "total": 2,
                "ok": 2,
                "errors": 0,
                "success_rate": 100.0,
                "avg_elapsed_ms": 12.3,
            },
            "report_path": str(report),
            "results": [
                {"image": "a.jpg", "status": "ok", "output_path": "out/a.jpg", "garment_type": "pants", "elapsed_ms": 10.1},
                {"image": "b.jpg", "status": "ok", "output_path": "out/b.jpg", "garment_type": "pants", "elapsed_ms": 14.5},
            ],
        }

    monkeypatch.setattr(api_main, "_run_fase21_tryon_batch", fake_batch)

    response = client.post(
        "/fase2_1/tryon/batch",
        json={
            "input_dir": str(tmp_path),
            "garment_path": "data/raw/models/TShirts.obj",
            "garment_type": "pants",
            "output_dir": str(tmp_path / "outs"),
            "report_path": str(report),
            "limit": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["summary"]["total"] == 2
    assert payload["report_path"] == str(report)
    assert payload["results"][0]["garment_type"] == "pants"


def test_fase2_1_tryon_batch_error(tmp_path: Path, monkeypatch) -> None:
    def fake_fail(_request):
        raise RuntimeError("batch failed")

    monkeypatch.setattr(api_main, "_run_fase21_tryon_batch", fake_fail)

    response = client.post(
        "/fase2_1/tryon/batch",
        json={
            "input_dir": str(tmp_path),
            "garment_path": "data/raw/models/TShirts.obj",
            "garment_type": "other",
            "output_dir": str(tmp_path / "outs"),
            "report_path": str(tmp_path / "report.json"),
        },
    )

    assert response.status_code == 400
    assert "batch failed" in response.json()["detail"]
