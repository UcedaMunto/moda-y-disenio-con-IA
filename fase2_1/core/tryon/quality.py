from __future__ import annotations

import json
from pathlib import Path


def default_checklist_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "config" / "quality_checklist.json")


def load_quality_checklist(checklist_path: str | None = None) -> dict:
    path = Path(checklist_path or default_checklist_path())
    if not path.exists():
        return {"version": "unknown", "criteria": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": "invalid", "criteria": []}


def auto_quality_score(status: str, checklist: dict) -> tuple[float, bool]:
    if status != "ok":
        return 0.0, True
    criteria = checklist.get("criteria", [])
    if not criteria:
        return 70.0, True
    # Baseline score for successful run; manual review is still required.
    return 70.0, True


def manual_quality_score(criteria_scores: dict[str, float], checklist: dict) -> float:
    criteria = checklist.get("criteria", [])
    if not criteria:
        return 0.0

    weighted = 0.0
    weights = 0.0
    for c in criteria:
        cid = str(c.get("id", "")).strip()
        w = float(c.get("weight", 0.0) or 0.0)
        if not cid or w <= 0:
            continue
        raw = float(criteria_scores.get(cid, 0.0) or 0.0)
        value = min(1.0, max(0.0, raw))
        weighted += value * w
        weights += w

    if weights <= 0:
        return 0.0
    return round((weighted / weights) * 100.0, 2)
