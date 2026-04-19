from core.telemetry.pipeline_stats import summarize_generation_stats


def test_summarize_generation_stats_empty_project() -> None:
    summary = summarize_generation_stats("project-without-stats")
    assert summary["runs"] == 0
    assert summary["reprocess_count"] == 0
    assert summary["avg_generation_seconds"] is None
