from core.projection.metrics import project_metrics


def test_project_metrics_without_data() -> None:
    metrics = project_metrics("project-without-data")
    assert metrics["project_id"] == "project-without-data"
    assert metrics["total_feedback"] == 0
    assert metrics["approve_count"] == 0
    assert metrics["reject_count"] == 0
