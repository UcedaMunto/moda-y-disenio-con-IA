import json
from pathlib import Path


def _stats_file(project_id: str) -> Path:
    file_path = Path("data/processed") / project_id / "generation_stats.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def record_generation_stats(
    project_id: str,
    n_candidates: int,
    n_references: int,
    duration_seconds: float,
) -> None:
    payload = {
        "project_id": project_id,
        "n_candidates": n_candidates,
        "n_references": n_references,
        "duration_seconds": duration_seconds,
    }
    file_path = _stats_file(project_id)
    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True) + "\n")


def summarize_generation_stats(project_id: str) -> dict:
    file_path = _stats_file(project_id)
    if not file_path.exists():
        return {
            "runs": 0,
            "avg_generation_seconds": None,
            "last_generation_seconds": None,
            "reprocess_count": 0,
        }

    durations: list[float] = []
    lines = []
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            lines.append(row)
            value = row.get("duration_seconds")
            if isinstance(value, (int, float)):
                durations.append(float(value))

    runs = len(lines)
    avg_duration = (sum(durations) / len(durations)) if durations else None
    last_duration = durations[-1] if durations else None

    return {
        "runs": runs,
        "avg_generation_seconds": avg_duration,
        "last_generation_seconds": last_duration,
        "reprocess_count": max(0, runs - 1),
    }
