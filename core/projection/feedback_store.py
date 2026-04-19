import json
from collections import Counter
from pathlib import Path

from core.db.postgres import candidate_ranking as postgres_candidate_ranking
from core.db.postgres import feedback_summary as postgres_feedback_summary
from core.db.postgres import insert_feedback


def _feedback_file(project_id: str) -> Path:
    path = Path("data/processed") / project_id / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_feedback(
    project_id: str,
    candidate_path: str,
    label: str,
    score: int | None = None,
    comment: str | None = None,
) -> None:
    payload = {
        "project_id": project_id,
        "candidate_path": candidate_path,
        "label": label,
        "score": score,
        "comment": comment,
    }
    try:
        insert_feedback(
            project_id=project_id,
            candidate_path=candidate_path,
            label=label,
            score=score,
            comment=comment,
        )
    except Exception:
        # Fallback keeps local development usable if DB is temporarily unavailable.
        file_path = _feedback_file(project_id)
        with file_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + "\n")


def summarize_feedback(project_id: str) -> dict:
    try:
        return postgres_feedback_summary(project_id)
    except Exception:
        pass

    file_path = _feedback_file(project_id)
    if not file_path.exists():
        return {
            "project_id": project_id,
            "total": 0,
            "labels": {},
            "avg_score": None,
            "storage": "jsonl",
        }

    labels: list[str] = []
    scores: list[int] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            labels.append(row.get("label", "unknown"))
            score = row.get("score")
            if isinstance(score, int):
                scores.append(score)

    counts = Counter(labels)
    avg_score = (sum(scores) / len(scores)) if scores else None

    return {
        "project_id": project_id,
        "total": len(labels),
        "labels": dict(counts),
        "avg_score": avg_score,
        "storage": "jsonl",
    }


def ranking_feedback(project_id: str, limit: int = 20) -> dict:
    try:
        return postgres_candidate_ranking(project_id=project_id, limit=limit)
    except Exception:
        pass

    file_path = _feedback_file(project_id)
    if not file_path.exists():
        return {
            "project_id": project_id,
            "limit": limit,
            "items": [],
            "storage": "jsonl",
        }

    per_candidate: dict[str, dict] = {}
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            path = row.get("candidate_path")
            if not path:
                continue
            data = per_candidate.setdefault(
                path,
                {
                    "candidate_path": path,
                    "total_feedback": 0,
                    "approve_count": 0,
                    "reject_count": 0,
                    "scores": [],
                },
            )
            data["total_feedback"] += 1
            if row.get("label") == "approve":
                data["approve_count"] += 1
            elif row.get("label") == "reject":
                data["reject_count"] += 1
            score = row.get("score")
            if isinstance(score, int):
                data["scores"].append(score)

    ranked = []
    for item in per_candidate.values():
        total = item["total_feedback"]
        avg_score = (
            sum(item["scores"]) / len(item["scores"])
            if item["scores"]
            else None
        )
        approval_ratio = (item["approve_count"] / total) if total else 0.0
        ranked.append(
            {
                "candidate_path": item["candidate_path"],
                "total_feedback": total,
                "approve_count": item["approve_count"],
                "reject_count": item["reject_count"],
                "avg_score": avg_score,
                "approval_ratio": approval_ratio,
            }
        )

    ranked.sort(
        key=lambda row: (
            row["approval_ratio"],
            row["avg_score"] if row["avg_score"] is not None else -1,
            row["total_feedback"],
        ),
        reverse=True,
    )

    safe_limit = max(1, min(limit, 200))
    return {
        "project_id": project_id,
        "limit": safe_limit,
        "items": ranked[:safe_limit],
        "storage": "jsonl",
    }
