from core.projection.feedback_store import ranking_feedback, summarize_feedback
from core.telemetry.pipeline_stats import summarize_generation_stats
from core.texture.catalog import list_project_candidates


def project_metrics(project_id: str) -> dict:
    summary = summarize_feedback(project_id)
    generation = summarize_generation_stats(project_id)
    candidates = list_project_candidates(project_id)
    ranking = ranking_feedback(project_id, limit=1)

    total_feedback = int(summary.get("total") or 0)
    labels = summary.get("labels") or {}
    approve_count = int(labels.get("approve") or 0)
    reject_count = int(labels.get("reject") or 0)

    approval_rate = (approve_count / total_feedback) if total_feedback else None
    rejection_rate = (reject_count / total_feedback) if total_feedback else None

    top_candidate = None
    items = ranking.get("items") or []
    if items:
        top_candidate = items[0]

    return {
        "project_id": project_id,
        "total_candidates": len(candidates),
        "generation_runs": generation.get("runs"),
        "avg_generation_seconds": generation.get("avg_generation_seconds"),
        "last_generation_seconds": generation.get("last_generation_seconds"),
        "reprocess_count": generation.get("reprocess_count"),
        "total_feedback": total_feedback,
        "approve_count": approve_count,
        "reject_count": reject_count,
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate,
        "avg_score": summary.get("avg_score"),
        "top_candidate": top_candidate,
        "feedback_storage": summary.get("storage"),
    }
