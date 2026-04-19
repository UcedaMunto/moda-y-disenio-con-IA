import json
from pathlib import Path

from fase2_1.scripts.evaluate_batch import build_consolidated_report


def test_build_consolidated_report(tmp_path: Path) -> None:
    batch_payload = {
        "report_path": "data/processed/fase2_1_eval/report.json",
        "summary": {
            "total": 2,
            "success_rate": 100.0,
            "avg_elapsed_ms": 12.4,
            "avg_auto_quality_score": 70.0,
            "checklist_version": "2.1.0",
        },
        "results": [],
    }

    manual_summary = {
        "status": "ok",
        "report_path": "data/processed/fase2_1_eval/manual_reviews.jsonl",
        "project_id": "fase2_1",
        "total": 2,
        "avg_score": 85.5,
        "min_score": 80.0,
        "max_score": 91.0,
        "latest_ts": 1,
        "by_reviewer": {"qa": 2},
        "checklist_versions": {"2.1.0": 2},
        "valid_score_threshold": 85.0,
        "visually_valid_count": 1,
        "visually_valid_rate": 50.0,
    }

    out = tmp_path / "consolidated.json"
    result = build_consolidated_report(batch_payload, manual_summary, str(out))

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["total_images"] == 2
    assert payload["summary"]["manual_avg_score"] == 85.5
    assert payload["summary"]["manual_visually_valid_rate"] == 50.0
    assert payload["summary"]["manual_acceptance_passed"] is False
    assert result["consolidated_report_path"] == str(out)
