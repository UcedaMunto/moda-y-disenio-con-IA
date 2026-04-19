from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_requires_existing_files(tmp_path: Path) -> None:
    fake_a = tmp_path / "a.jpg"
    fake_b = tmp_path / "b.jpg"
    fake_a.write_bytes(b"not-an-image")
    fake_b.write_bytes(b"not-an-image")

    payload = {
        "project_id": "test-project",
        "image_paths": [str(fake_a), str(fake_b)],
        "n_candidates": 2,
    }
    response = client.post("/projects/generate", json=payload)

    assert response.status_code == 400
