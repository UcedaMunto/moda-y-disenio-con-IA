from core.projection.feedback_store import ranking_feedback


def test_ranking_feedback_empty_project() -> None:
    result = ranking_feedback(project_id="project-without-data", limit=5)
    assert result["project_id"] == "project-without-data"
    assert result["limit"] == 5
    assert isinstance(result["items"], list)
