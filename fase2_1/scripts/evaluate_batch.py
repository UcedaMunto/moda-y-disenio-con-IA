from __future__ import annotations

import argparse
import json
from pathlib import Path

from fase2_1.core.tryon.batch import run_tryon_batch
from fase2_1.core.tryon.review import build_consolidated_evaluation_report, summarize_manual_reviews


def build_consolidated_report(batch_payload: dict, manual_summary: dict, consolidated_path: str) -> dict:
    # Backward-compatible helper used by tests and quick scripting.
    auto_summary = batch_payload.get("summary", {})
    merged = {
        "status": "ok",
        "batch_report_path": batch_payload.get("report_path"),
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
        },
        "manual": manual_summary,
        "batch": batch_payload,
    }

    out = Path(consolidated_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    merged["consolidated_report_path"] = str(out)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Fase 2.1 batch try-on latency and success")
    parser.add_argument("--input-dir", required=True, help="Directorio de imagenes de persona")
    parser.add_argument("--garment-path", required=True, help="Ruta de prenda (placeholder en baseline)")
    parser.add_argument("--output-dir", default="data/processed/fase2_1_eval", help="Directorio salida")
    parser.add_argument("--report", default="data/processed/fase2_1_eval/report.json", help="Reporte JSON")
    parser.add_argument(
        "--manual-report",
        default="data/processed/fase2_1_eval/manual_reviews.jsonl",
        help="Reporte JSONL de evaluacion manual",
    )
    parser.add_argument(
        "--consolidated-report",
        default="data/processed/fase2_1_eval/consolidated_report.json",
        help="Reporte consolidado batch + manual",
    )
    parser.add_argument("--project-id", default=None, help="Filtra evaluaciones manuales por project_id")
    parser.add_argument("--checklist-path", default=None, help="Ruta opcional del checklist de calidad")
    args = parser.parse_args()

    batch = run_tryon_batch(
        input_dir=args.input_dir,
        garment_path=args.garment_path,
        output_dir=args.output_dir,
        report_path=args.report,
        checklist_path=args.checklist_path,
        limit=None,
    )

    batch_payload = batch.model_dump()
    manual_summary = summarize_manual_reviews(report_path=args.manual_report, project_id=args.project_id)
    consolidated = build_consolidated_evaluation_report(
        batch_report_path=batch_payload.get("report_path", args.report),
        manual_report_path=args.manual_report,
        project_id=args.project_id,
        consolidated_path=args.consolidated_report,
    )
    print(json.dumps(consolidated, indent=2))


if __name__ == "__main__":
    main()
