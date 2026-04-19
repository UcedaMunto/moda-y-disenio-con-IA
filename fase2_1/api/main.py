from fastapi import FastAPI, HTTPException

from fase2_1.core.tryon.batch import run_tryon_batch
from fase2_1.core.tryon.pipeline import run_tryon
from fase2_1.core.tryon.review import (
    append_manual_review,
    build_consolidated_evaluation_report,
    summarize_manual_reviews,
)
from fase2_1.core.tryon.schemas import TryOnBatchRequest, TryOnManualEvalRequest, TryOnRequest

app = FastAPI(title="Fase 2.1 TryOn API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "module": "fase2_1"}


@app.post("/tryon/run")
def tryon_run(request: TryOnRequest) -> dict:
    try:
        result = run_tryon(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.post("/tryon/batch")
def tryon_batch(request: TryOnBatchRequest) -> dict:
    try:
        result = run_tryon_batch(
            input_dir=request.input_dir,
            garment_path=request.garment_path,
            output_dir=request.output_dir,
            report_path=request.report_path,
            checklist_path=request.checklist_path,
            limit=request.limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.post("/tryon/evaluate")
def tryon_evaluate(request: TryOnManualEvalRequest) -> dict:
    try:
        return append_manual_review(
            image=request.image,
            output_path=request.output_path,
            criteria_scores=request.criteria_scores,
            reviewer=request.reviewer,
            comment=request.comment,
            report_path=request.report_path,
            project_id=request.project_id,
            checklist_path=request.checklist_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tryon/evaluate-summary")
def tryon_evaluate_summary(
    report_path: str = "data/processed/fase2_1_eval/manual_reviews.jsonl",
    project_id: str | None = None,
) -> dict:
    try:
        return summarize_manual_reviews(report_path=report_path, project_id=project_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tryon/evaluate-consolidated")
def tryon_evaluate_consolidated(
    batch_report_path: str = "data/processed/fase2_1_eval/report.json",
    manual_report_path: str = "data/processed/fase2_1_eval/manual_reviews.jsonl",
    project_id: str | None = None,
    consolidated_path: str | None = None,
) -> dict:
    try:
        return build_consolidated_evaluation_report(
            batch_report_path=batch_report_path,
            manual_report_path=manual_report_path,
            project_id=project_id,
            consolidated_path=consolidated_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
