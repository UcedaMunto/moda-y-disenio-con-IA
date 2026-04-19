from pathlib import Path

from fastapi.testclient import TestClient

from apps.api import main as api_main
from apps.api.main import app


client = TestClient(app)


def test_fase2_1_tryon_manual_eval_ok(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "manual_reviews.jsonl"

    def fake_eval(_request):
        report.write_text('{"status": "ok"}\n', encoding="utf-8")
        return {
            "status": "ok",
            "saved_path": str(report),
            "score": 87.5,
            "record": {
                "image": "a.jpg",
                "checklist_version": "2.1.0",
            },
        }

    monkeypatch.setattr(api_main, "_run_fase21_manual_eval", fake_eval)

    response = client.post(
        "/fase2_1/tryon/evaluate",
        json={
            "image": "data/processed/fase2_1_eval/outputs/a.jpg",
            "output_path": "data/processed/fase2_1_eval/outputs/a.jpg",
            "criteria_scores": {
                "alignment_shoulders": 1.0,
                "torso_scale": 0.5,
            },
            "reviewer": "qa",
            "comment": "ok",
            "report_path": str(report),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["saved_path"] == str(report)
    assert payload["score"] == 87.5


def test_fase2_1_tryon_manual_eval_error(monkeypatch) -> None:
    def fake_fail(_request):
        raise RuntimeError("manual eval failed")

    monkeypatch.setattr(api_main, "_run_fase21_manual_eval", fake_fail)

    response = client.post(
        "/fase2_1/tryon/evaluate",
        json={
            "image": "x.jpg",
            "criteria_scores": {},
        },
    )

    assert response.status_code == 400
    assert "manual eval failed" in response.json()["detail"]


def test_fase2_1_tryon_manual_eval_summary_ok(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "manual_reviews.jsonl"

    def fake_summary(
        report_path: str,
        project_id: str | None = None,
        valid_score_threshold: float = 85.0,
    ):
        assert report_path == str(report)
        assert project_id == "fase2_1"
        assert valid_score_threshold == 88.0
        return {
            "status": "ok",
            "report_path": str(report),
            "project_id": "fase2_1",
            "total": 3,
            "avg_score": 88.2,
            "min_score": 70.0,
            "max_score": 95.0,
            "latest_ts": 1,
            "by_reviewer": {"qa": 3},
            "checklist_versions": {"2.1.0": 3},
            "valid_score_threshold": 88.0,
            "visually_valid_count": 2,
            "visually_valid_rate": 66.67,
        }

    monkeypatch.setattr(api_main, "_run_fase21_manual_eval_summary", fake_summary)

    response = client.get(
        "/fase2_1/tryon/evaluate-summary",
        params={"report_path": str(report), "project_id": "fase2_1", "valid_score_threshold": 88.0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["total"] == 3
    assert payload["avg_score"] == 88.2
    assert payload["visually_valid_rate"] == 66.67


def test_fase2_1_tryon_manual_eval_summary_error(monkeypatch) -> None:
    def fake_fail(
        report_path: str,
        project_id: str | None = None,
        valid_score_threshold: float = 85.0,
    ):
        assert isinstance(report_path, str)
        assert project_id is None or isinstance(project_id, str)
        assert isinstance(valid_score_threshold, float)
        raise RuntimeError("summary failed")

    monkeypatch.setattr(api_main, "_run_fase21_manual_eval_summary", fake_fail)

    response = client.get("/fase2_1/tryon/evaluate-summary")

    assert response.status_code == 400
    assert "summary failed" in response.json()["detail"]


def test_fase2_1_tryon_manual_eval_consolidated_ok(tmp_path: Path, monkeypatch) -> None:
    batch_report = tmp_path / "report.json"
    manual_report = tmp_path / "manual_reviews.jsonl"

    def fake_consolidated(
        batch_report_path: str,
        manual_report_path: str,
        project_id: str | None = None,
        consolidated_path: str | None = None,
        valid_score_threshold: float = 85.0,
    ):
        assert batch_report_path == str(batch_report)
        assert manual_report_path == str(manual_report)
        assert project_id == "fase2_1"
        assert consolidated_path is None
        assert valid_score_threshold == 87.0
        return {
            "status": "ok",
            "summary": {
                "total_images": 2,
                "batch_success_rate": 100.0,
                "manual_total": 2,
                "manual_avg_score": 88.5,
                "manual_visually_valid_rate": 100.0,
            },
        }

    monkeypatch.setattr(api_main, "_run_fase21_manual_eval_consolidated", fake_consolidated)

    response = client.get(
        "/fase2_1/tryon/evaluate-consolidated",
        params={
            "batch_report_path": str(batch_report),
            "manual_report_path": str(manual_report),
            "project_id": "fase2_1",
            "valid_score_threshold": 87.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["summary"]["manual_avg_score"] == 88.5
    assert payload["summary"]["manual_visually_valid_rate"] == 100.0


def test_fase2_1_tryon_manual_eval_consolidated_error(monkeypatch) -> None:
    def fake_fail(
        batch_report_path: str,
        manual_report_path: str,
        project_id: str | None = None,
        consolidated_path: str | None = None,
        valid_score_threshold: float = 85.0,
    ):
        assert isinstance(batch_report_path, str)
        assert isinstance(manual_report_path, str)
        assert project_id is None or isinstance(project_id, str)
        assert consolidated_path is None or isinstance(consolidated_path, str)
        assert isinstance(valid_score_threshold, float)
        raise RuntimeError("consolidated failed")

    monkeypatch.setattr(api_main, "_run_fase21_manual_eval_consolidated", fake_fail)

    response = client.get("/fase2_1/tryon/evaluate-consolidated")

    assert response.status_code == 400
    assert "consolidated failed" in response.json()["detail"]
