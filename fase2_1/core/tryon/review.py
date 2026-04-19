from __future__ import annotations

import json
import time
from pathlib import Path

from .quality import load_quality_checklist, manual_quality_score


def append_manual_review(
    image: str,
    output_path: str | None,
    criteria_scores: dict[str, float],
    reviewer: str | None,
    comment: str | None,
    report_path: str,
    project_id: str = "fase2_1",
    checklist_path: str | None = None,
) -> dict:
    checklist = load_quality_checklist(checklist_path)
    score = manual_quality_score(criteria_scores, checklist)

    record = {
        "ts": int(time.time()),
        "project_id": project_id,
        "image": image,
        "output_path": output_path,
        "criteria_scores": criteria_scores,
        "score": score,
        "reviewer": reviewer,
        "comment": comment,
        "checklist_version": checklist.get("version", "unknown"),
    }

    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "status": "ok",
        "saved_path": str(out),
        "score": score,
        "record": record,
    }


def summarize_manual_reviews(report_path: str, project_id: str | None = None) -> dict:
    src = Path(report_path)
    if not src.exists():
        return {
            "status": "ok",
            "report_path": str(src),
            "project_id": project_id,
            "total": 0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "latest_ts": None,
            "by_reviewer": {},
            "checklist_versions": {},
        }

    rows: list[dict] = []
    for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if project_id and row.get("project_id") != project_id:
            continue
        rows.append(row)

    if not rows:
        return {
            "status": "ok",
            "report_path": str(src),
            "project_id": project_id,
            "total": 0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "latest_ts": None,
            "by_reviewer": {},
            "checklist_versions": {},
        }

    scores = [float(r.get("score", 0.0) or 0.0) for r in rows]
    latest_ts = max(int(r.get("ts", 0) or 0) for r in rows)

    by_reviewer: dict[str, int] = {}
    checklist_versions: dict[str, int] = {}
    for row in rows:
        reviewer = str(row.get("reviewer") or "unknown")
        by_reviewer[reviewer] = by_reviewer.get(reviewer, 0) + 1

        version = str(row.get("checklist_version") or "unknown")
        checklist_versions[version] = checklist_versions.get(version, 0) + 1

    return {
        "status": "ok",
        "report_path": str(src),
        "project_id": project_id,
        "total": len(rows),
        "avg_score": round(sum(scores) / len(scores), 2),
        "min_score": round(min(scores), 2),
        "max_score": round(max(scores), 2),
        "latest_ts": latest_ts,
        "by_reviewer": by_reviewer,
        "checklist_versions": checklist_versions,
    }


def build_consolidated_evaluation_report(
    batch_report_path: str,
    manual_report_path: str,
    project_id: str | None = None,
    consolidated_path: str | None = None,
) -> dict:
    batch_path = Path(batch_report_path)
    if batch_path.exists():
        try:
            batch_payload = json.loads(batch_path.read_text(encoding="utf-8"))
        except Exception:
            batch_payload = {"status": "error", "summary": {}, "results": [], "report_path": str(batch_path)}
    else:
        batch_payload = {"status": "missing", "summary": {}, "results": [], "report_path": str(batch_path)}

    manual_summary = summarize_manual_reviews(report_path=manual_report_path, project_id=project_id)
    auto_summary = batch_payload.get("summary", {})

    merged = {
        "status": "ok",
        "batch_report_path": str(batch_path),
        "manual_report_path": manual_summary.get("report_path"),
        "summary": {
            "total_images": auto_summary.get("total", 0),
            "batch_success_rate": auto_summary.get("success_rate", 0.0),
            "avg_elapsed_ms": auto_summary.get("avg_elapsed_ms", 0.0),
            "avg_auto_quality_score": auto_summary.get("avg_auto_quality_score", 0.0),
            "manual_total": manual_summary.get("total", 0),
            "manual_avg_score": manual_summary.get("avg_score", 0.0),
            "manual_min_score": manual_summary.get("min_score", 0.0),
            "manual_max_score": manual_summary.get("max_score", 0.0),
            "checklist_version": auto_summary.get("checklist_version", "unknown"),
            "project_id": project_id,
        },
        "manual": manual_summary,
        "batch": batch_payload,
    }

    if consolidated_path:
        out = Path(consolidated_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        merged["consolidated_report_path"] = str(out)

    return merged
