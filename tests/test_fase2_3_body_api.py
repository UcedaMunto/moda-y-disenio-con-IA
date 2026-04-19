from pathlib import Path
import json

from fastapi.testclient import TestClient

from apps.api import main as api_main
from apps.api.main import app


client = TestClient(app)


def test_fase2_3_body_reconstruct_ok(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "person.jpg"
    src.write_bytes(b"fake-image")

    def fake_run(request):
        assert request.image_path == str(src)
        return {
            "status": "ok",
            "mode": "mock",
            "case_id": "case-001",
            "input_image": str(src),
            "output_dir": "data/processed/fase2_3/case-001",
            "mesh_paths": [],
            "preview_path": "data/processed/fase2_3/case-001/person_mock_preview.jpg",
            "preview_url": "/artifacts/processed/fase2_3/case-001/person_mock_preview.jpg",
            "meta_path": "data/processed/fase2_3/case-001/run_meta.json",
        }

    monkeypatch.setattr(api_main, "_run_fase23_body_reconstruct", fake_run)

    response = client.post(
        "/fase2_3/body/reconstruct",
        json={
            "image_path": str(src),
            "use_mock": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] == "mock"
    assert data["case_id"] == "case-001"
    assert data["preview_url"].startswith("/artifacts/")


def test_fase2_3_body_reconstruct_error(monkeypatch) -> None:
    def fake_fail(_request):
        raise RuntimeError("sam3d failed")

    monkeypatch.setattr(api_main, "_run_fase23_body_reconstruct", fake_fail)

    response = client.post(
        "/fase2_3/body/reconstruct",
        json={
            "image_path": "fotos_personas/missing.jpg",
            "use_mock": False,
        },
    )

    assert response.status_code == 400
    assert "sam3d failed" in response.json()["detail"]


def test_fase2_3_body_reconstruct_missing_checkpoints_falls_back_to_mock(tmp_path: Path) -> None:
    src = tmp_path / "person.jpg"
    src.write_bytes(b"fake-image")

    response = client.post(
        "/fase2_3/body/reconstruct",
        json={
            "image_path": str(src),
            "use_mock": False,
            "checkpoint_path": None,
            "mhr_path": None,
            "case_id": "missing-ckpt-fallback",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "mock_fallback"
    assert "warning" in payload


def test_fase2_3_body_preflight_ok(tmp_path: Path, monkeypatch) -> None:
    sam3d_root = tmp_path / "sam3d"
    sam3d_root.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("x", encoding="utf-8")
    mhr = tmp_path / "mhr_model.pt"
    mhr.write_text("x", encoding="utf-8")

    monkeypatch.setenv("SAM3D_ROOT", str(sam3d_root))
    monkeypatch.setenv("SAM3D_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("SAM3D_MHR_PATH", str(mhr))

    response = client.get("/fase2_3/body/preflight")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["capabilities"]["real_ready"] is True
    assert data["capabilities"]["mock_ready"] is True
    assert data["checks"]["sam3d_root_exists"] is True
    assert data["checks"]["checkpoint_exists"] is True
    assert data["checks"]["mhr_exists"] is True


def test_fase2_3_body_reconstruct_uses_env_paths(tmp_path: Path, monkeypatch) -> None:
    sam3d_root = tmp_path / "sam3d"
    sam3d_root.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("x", encoding="utf-8")
    mhr = tmp_path / "mhr_model.pt"
    mhr.write_text("x", encoding="utf-8")
    image_path = tmp_path / "person.jpg"
    image_path.write_bytes(b"fake-image")

    monkeypatch.setenv("SAM3D_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("SAM3D_MHR_PATH", str(mhr))
    monkeypatch.setattr(api_main, "DATA_DIR", tmp_path / "data")
    api_main.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fake_subprocess_run(cmd, cwd, capture_output, text, check):
        assert "--checkpoint-path" in cmd
        assert cmd[cmd.index("--checkpoint-path") + 1] == str(checkpoint)
        assert "--mhr-path" in cmd
        assert cmd[cmd.index("--mhr-path") + 1] == str(mhr)

        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        preview = out_dir / "person_overlay.jpg"
        preview.write_bytes(b"fake-jpg")
        mesh = out_dir / "person_mesh_000.ply"
        mesh.write_text("ply", encoding="utf-8")
        glb = out_dir / "person_mesh_000.glb"
        glb.write_text("glb", encoding="utf-8")
        result = {
            "status": "ok",
            "mesh_paths": [str(mesh)],
            "glb_paths": [str(glb)],
            "overlay_path": str(preview),
            "n_people": 1,
            "device": "cpu",
        }
        (out_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

        class ProcResult:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return ProcResult()

    monkeypatch.setattr(api_main.subprocess, "run", fake_subprocess_run)

    request = api_main.Fase23BodyReconstructRequest(
        image_path=str(image_path),
        case_id="case-env-001",
        sam3d_root=str(sam3d_root),
        checkpoint_path=None,
        mhr_path=None,
        use_mock=False,
    )
    payload = api_main._run_fase23_body_reconstruct(request)

    assert payload["status"] == "ok"
    assert payload["mode"] == "sam3d_body"
    assert payload["mesh_paths"]
    assert payload["glb_paths"]
    assert payload.get("avatar_glb_path")
    assert payload["runtime"]["checkpoint_path"] == str(checkpoint)
    assert payload["runtime"]["mhr_path"] == str(mhr)
