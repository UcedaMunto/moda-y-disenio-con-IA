from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import DATA_DIR, app


client = TestClient(app)


def test_deepfashion_coverage_endpoint_refresh(monkeypatch, tmp_path: Path) -> None:
    def _mock_run_deepfashion_coverage(include_samples: bool = True, sample_limit: int = 25) -> dict:
        return {
            "status": "ok",
            "counts": {
                "old": {"union": 10},
                "new": {"union": 12},
                "shared_union_ids": 10,
                "old_only_union_ids": 0,
                "new_only_union_ids": 2,
            },
            "coverage": {
                "new_over_old_union_pct": 100.0,
                "old_over_new_union_pct": 83.333,
            },
            "samples": {
                "shared_ids": ["1", "2"],
                "old_only_ids": [],
                "new_only_ids": ["11", "12"],
            },
        }

    monkeypatch.setattr("apps.api.main._run_deepfashion_coverage", _mock_run_deepfashion_coverage)

    response = client.get("/assets/deepfashion-coverage", params={"refresh": "true", "sample_limit": 10})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["cached"] is False
    assert payload["counts"]["shared_union_ids"] == 10
    assert payload["coverage"]["new_over_old_union_pct"] == 100.0


def test_deepfashion_coverage_endpoint_cached(tmp_path: Path) -> None:
    report_path = DATA_DIR / "processed" / "deepfashion" / "coverage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        """{
  "status": "ok",
  "counts": {
    "old": {"union": 1},
    "new": {"union": 1},
    "shared_union_ids": 1,
    "old_only_union_ids": 0,
    "new_only_union_ids": 0
  },
  "coverage": {
    "new_over_old_union_pct": 100.0,
    "old_over_new_union_pct": 100.0
  }
}
""",
        encoding="utf-8",
    )

    response = client.get("/assets/deepfashion-coverage")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["cached"] is True
    assert payload["counts"]["shared_union_ids"] == 1
