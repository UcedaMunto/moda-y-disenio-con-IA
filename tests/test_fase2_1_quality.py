import json
from pathlib import Path

from fase2_1.core.tryon.quality import load_quality_checklist, manual_quality_score
from fase2_1.core.tryon.review import append_manual_review, build_consolidated_evaluation_report, summarize_manual_reviews


def test_manual_quality_score_weighted() -> None:
    checklist = {
        "version": "x",
        "criteria": [
            {"id": "a", "weight": 0.25},
            {"id": "b", "weight": 0.75},
        ],
    }
    score = manual_quality_score({"a": 1.0, "b": 0.5}, checklist)
    assert score == 62.5


def test_append_manual_review_writes_jsonl(tmp_path: Path) -> None:
    checklist_path = tmp_path / "checklist.json"
    checklist_path.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "criteria": [
                    {"id": "alignment_shoulders", "weight": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    report_path = tmp_path / "manual_reviews.jsonl"
    result = append_manual_review(
        image="img.jpg",
        output_path="out.jpg",
        criteria_scores={"alignment_shoulders": 1.0},
        reviewer="qa",
        comment="ok",
        report_path=str(report_path),
        checklist_path=str(checklist_path),
    )

    assert result["status"] == "ok"
    assert result["score"] == 100.0
    assert report_path.exists()


def test_load_quality_checklist_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    checklist = load_quality_checklist(str(missing))
    assert checklist["version"] == "unknown"
    assert checklist["criteria"] == []


def test_summarize_manual_reviews_filters_project(tmp_path: Path) -> None:
    report_path = tmp_path / "manual_reviews.jsonl"
    report_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 10,
                        "project_id": "fase2_1",
                        "score": 80.0,
                        "reviewer": "qa",
                        "checklist_version": "2.1.0",
                    }
                ),
                json.dumps(
                    {
                        "ts": 11,
                        "project_id": "fase2_1",
                        "score": 90.0,
                        "reviewer": "qa",
                        "checklist_version": "2.1.0",
                    }
                ),
                json.dumps(
                    {
                        "ts": 12,
                        "project_id": "otro",
                        "score": 10.0,
                        "reviewer": "x",
                        "checklist_version": "1.0",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_manual_reviews(str(report_path), project_id="fase2_1")

    assert summary["total"] == 2
    assert summary["avg_score"] == 85.0
    assert summary["min_score"] == 80.0
    assert summary["max_score"] == 90.0
    assert summary["latest_ts"] == 11
    assert summary["by_reviewer"]["qa"] == 2


def test_build_consolidated_evaluation_report(tmp_path: Path) -> None:
    batch_path = tmp_path / "report.json"
    manual_path = tmp_path / "manual_reviews.jsonl"
    out_path = tmp_path / "consolidated.json"

    batch_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "report_path": str(batch_path),
                "summary": {
                    "total": 3,
                    "success_rate": 66.67,
                    "avg_elapsed_ms": 11.2,
                    "avg_auto_quality_score": 70.0,
                    "checklist_version": "2.1.0",
                },
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    manual_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1, "project_id": "fase2_1", "score": 80.0, "reviewer": "qa", "checklist_version": "2.1.0"}),
                json.dumps({"ts": 2, "project_id": "fase2_1", "score": 90.0, "reviewer": "qa", "checklist_version": "2.1.0"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_consolidated_evaluation_report(
        batch_report_path=str(batch_path),
        manual_report_path=str(manual_path),
        project_id="fase2_1",
        consolidated_path=str(out_path),
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["total_images"] == 3
    assert payload["summary"]["manual_avg_score"] == 85.0
    assert payload["consolidated_report_path"] == str(out_path)
    assert out_path.exists()
