from __future__ import annotations

import json
import time
from pathlib import Path

from .pipeline import run_tryon
from .quality import auto_quality_score, load_quality_checklist
from .schemas import TryOnBatchResult, TryOnRequest


def collect_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def run_tryon_batch(
    input_dir: str,
    garment_path: str,
    output_dir: str,
    report_path: str,
    checklist_path: str | None = None,
    limit: int | None = None,
) -> TryOnBatchResult:
    base_input = Path(input_dir)
    if not base_input.exists() or not base_input.is_dir():
        raise FileNotFoundError(f"Input dir no existe: {input_dir}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)

    images = collect_images(base_input)
    if limit is not None:
        images = images[:limit]

    checklist = load_quality_checklist(checklist_path)
    results: list[dict] = []
    ok = 0

    for image in images:
        out_img = out_dir / image.name
        req = TryOnRequest(
            image_path=str(image),
            garment_path=garment_path,
            output_path=str(out_img),
        )
        t0 = time.perf_counter()
        try:
            res = run_tryon(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ok += 1
            score, needs_manual = auto_quality_score("ok", checklist)
            results.append(
                {
                    "image": str(image),
                    "status": res.status,
                    "output_path": res.output_path,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "auto_quality_score": score,
                    "manual_review_required": needs_manual,
                }
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            score, needs_manual = auto_quality_score("error", checklist)
            results.append(
                {
                    "image": str(image),
                    "status": "error",
                    "error": str(exc),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "auto_quality_score": score,
                    "manual_review_required": needs_manual,
                }
            )

    avg_ms = round(sum(r["elapsed_ms"] for r in results) / len(results), 3) if results else 0.0
    avg_auto = round(sum(float(r.get("auto_quality_score", 0.0)) for r in results) / len(results), 2) if results else 0.0
    summary = {
        "total": len(results),
        "ok": ok,
        "errors": len(results) - ok,
        "success_rate": round((ok / len(results)) * 100.0, 2) if results else 0.0,
        "avg_elapsed_ms": avg_ms,
        "avg_auto_quality_score": avg_auto,
        "checklist_version": checklist.get("version", "unknown"),
    }

    payload = {
        "status": "ok",
        "summary": summary,
        "report_path": str(report),
        "results": results,
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return TryOnBatchResult(**payload)
