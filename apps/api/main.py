from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import HTTPException
import base64
import binascii
import os
import re
import time
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.db.postgres import init_database
from core.assets.importer import import_assets_from_directory
from core.preprocessing.pipeline import preprocess_images
from core.texture.catalog import list_project_candidates
from core.texture.engine import generate_texture_candidates
from core.texture.preview import build_preview_sheet
from core.telemetry.pipeline_stats import record_generation_stats
from core.projection.selector import filter_candidates
from core.projection.metrics import project_metrics
from core.projection.feedback_store import append_feedback, ranking_feedback, summarize_feedback
from core.rendering.blender_runner import apply_texture_and_export
from core.rendering.model_contract import validate_model_contract
from core.rendering.preview_3d import generate_3d_preview
from core.rendering.pointcloud_proxy import build_point_cloud_proxy_mesh, is_point_cloud_model_path
from core.tracking.mlflow_tracker import track_event
from fase2_1.core.tryon.schemas import (
    TryOnBatchRequest as Fase21TryOnBatchRequest,
    TryOnManualEvalRequest as Fase21TryOnManualEvalRequest,
    TryOnRequest as Fase21TryOnRequest,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Ensure feedback table exists before first write.
    try:
        init_database()
    except Exception:
        # Keep API running with file fallback when DB is down.
        pass
    yield


app = FastAPI(title="Fabric2Mesh API", version="0.1.0", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = (BASE_DIR / "data").resolve()
UI_DIR = (BASE_DIR / "apps" / "api" / "ui").resolve()
DEEPFASHION_DIR = (BASE_DIR / "data_deepfashon").resolve()
DEEPFASHION_ANTIGUO_DIR = (BASE_DIR / "data_deepfasho_antiguo").resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/artifacts", StaticFiles(directory=str(DATA_DIR)), name="artifacts")
app.mount("/ui-static", StaticFiles(directory=str(UI_DIR)), name="ui-static")
if DEEPFASHION_DIR.exists():
    app.mount("/deepfashion-artifacts", StaticFiles(directory=str(DEEPFASHION_DIR)), name="deepfashion-artifacts")
if DEEPFASHION_ANTIGUO_DIR.exists():
    app.mount("/deepfashion-antiguo-artifacts", StaticFiles(directory=str(DEEPFASHION_ANTIGUO_DIR)), name="deepfashion-antiguo-artifacts")


MODEL_EXT = {".obj", ".glb", ".gltf", ".fbx", ".ply"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_MODEL_INDEX_CACHE: list[dict] | None = None
MAX_INITIAL_FRONT_MODELS = 100
MAX_SEARCH_RESULTS = 20
MAX_MODEL_FILE_BYTES = 100 * 1024 * 1024  # 100MB hard cap to protect browser/GPU

_DF3D_UPPER = {"long_sleeve_upper", "short_sleeve_upper", "no_sleeve_upper"}
_DF3D_DRESSES = {"long_sleeve_dress", "short_sleeve_dress", "no_sleeve_dress", "dress"}
_DF3D_PANTS = {"long_pants", "short_pants"}


def infer_garment_type(file_name: str) -> str:
    name = file_name.lower()
    shirt_keys = {"shirt", "tshirt", "t-shirt", "tee", "raglan", "top", "camisa"}
    skirt_keys = {"skirt", "falda"}
    pants_keys = {"pants", "pant", "trouser", "jean", "pantalon", "pantalones"}

    if any(key in name for key in shirt_keys):
        return "shirt"
    if any(key in name for key in skirt_keys):
        return "skirt"
    if any(key in name for key in pants_keys):
        return "pants"
    return "other"


def _model_item(logical_path: str, size_bytes: int, garment_type_override: str | None = None, source_override: str | None = None) -> dict:
    suffix = Path(logical_path).suffix.lower()
    name = Path(logical_path).name
    garment_type = garment_type_override or infer_garment_type(Path(logical_path).name)
    is_point_cloud = "/point_cloud/" in logical_path or "/pointcloud/" in logical_path or logical_path.endswith(".ply")
    
    if source_override:
        source = source_override
    elif logical_path.startswith("data/raw/models/"):
        source = "local"
    elif logical_path.startswith("data_deepfashon/"):
        source = "deepfashion"
    elif logical_path.startswith("data_deepfasho_antiguo/"):
        source = "deepfashion_antiguo"
    else:
        source = "unknown"
    
    source_rank = {"local": 0, "deepfashion": 1, "deepfashion_antiguo": 2, "unknown": 3}.get(source, 3)
    
    return {
        "path": logical_path,
        "name": name,
        "size_bytes": size_bytes,
        "format": suffix.replace(".", ""),
        "source": source,
        "is_point_cloud": is_point_cloud,
        "garment_type": garment_type,
        "is_target_garment": garment_type in {"shirt", "skirt", "pants", "dress"},
        "_search_text": f"{logical_path.lower()} {name.lower()} {garment_type.lower()}",
        "_source_rank": source_rank,
    }


def _public_model_item(item: dict) -> dict:
    return {
        "path": item.get("path"),
        "name": item.get("name"),
        "size_bytes": item.get("size_bytes"),
        "format": item.get("format"),
        "source": item.get("source"),
        "is_point_cloud": item.get("is_point_cloud"),
        "garment_type": item.get("garment_type"),
        "is_target_garment": item.get("is_target_garment"),
    }


def _scan_models_from(base_dir: Path, logical_prefix: str) -> list[dict]:
    out: list[dict] = []
    if not base_dir.exists():
        return out
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MODEL_EXT:
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        if size_bytes > MAX_MODEL_FILE_BYTES:
            # Avoid exposing huge raw meshes that can crash browser/GPU on preview.
            continue
        rel = path.relative_to(base_dir).as_posix()
        logical_path = f"{logical_prefix}/{rel}"
        out.append(_model_item(logical_path, size_bytes=size_bytes))
    return out


def _read_deepfashion_cloth_type_map(base_dir: Path) -> dict[str, str]:
    """Map garment numeric id -> coarse type from cloth_type_list.txt."""
    cloth_file = base_dir / "cloth_type_list.txt"
    mapping: dict[str, str] = {}
    if not cloth_file.exists():
        return mapping

    for raw_line in cloth_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        label = parts[0].strip().lower()
        if label in _DF3D_UPPER:
            gtype = "shirt"
        elif label in _DF3D_DRESSES:
            gtype = "dress"
        elif label in _DF3D_PANTS:
            gtype = "pants"
        else:
            # Unknown/other class: skip for now to keep only clothes classes we know.
            continue

        for token in parts[1:]:
            token = token.strip()
            if token.isdigit():
                mapping[token] = gtype
    return mapping


def _scan_deepfashion_models(base_dir: Path, cloth_map: dict[str, str]) -> list[dict]:
    """Index only garment point clouds listed in cloth_type_list.txt.

    This intentionally ignores annotation folders like DF3D_Featurelines.
    """
    out: list[dict] = []
    point_cloud_dir = base_dir / "point_cloud"
    if not point_cloud_dir.exists():
        return out

    for path in sorted(point_cloud_dir.rglob("*.ply")):
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        if size_bytes > MAX_MODEL_FILE_BYTES:
            continue

        rel = path.relative_to(base_dir).as_posix()  # point_cloud/<id>/<id-pose>.ply
        top_id = rel.split("/", 2)[1] if rel.startswith("point_cloud/") and "/" in rel else ""
        garment_type = cloth_map.get(top_id)
        if not garment_type:
            # Keep only IDs explicitly defined as clothes in cloth_type_list.
            continue

        logical_path = f"data_deepfashon/{rel}"
        out.append(_model_item(logical_path, size_bytes=size_bytes, garment_type_override=garment_type, source_override="deepfashion"))
    return out


def _scan_deepfashion_antiguo_models(base_dir: Path, cloth_map: dict[str, str]) -> list[dict]:
    """Index garment point clouds from the older DeepFashion dataset.
    
    Uses the same cloth_type_list.txt for garment classification.
    The directory structure is: pointcloud/<id>/<id-pose>.ply
    """
    out: list[dict] = []
    point_cloud_dir = base_dir / "pointcloud"
    if not point_cloud_dir.exists():
        return out

    for path in sorted(point_cloud_dir.rglob("*.ply")):
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        if size_bytes > MAX_MODEL_FILE_BYTES:
            continue

        rel = path.relative_to(base_dir).as_posix()  # pointcloud/<id>/<id-pose>.ply
        top_id = rel.split("/", 2)[1] if rel.startswith("pointcloud/") and "/" in rel else ""
        garment_type = cloth_map.get(top_id)
        if not garment_type:
            # Keep only IDs explicitly defined as clothes in cloth_type_list.
            continue

        logical_path = f"data_deepfasho_antiguo/{rel}"
        out.append(_model_item(logical_path, size_bytes=size_bytes, garment_type_override=garment_type, source_override="deepfashion_antiguo"))
    return out


def get_model_index(force_refresh: bool = False) -> list[dict]:
    global _MODEL_INDEX_CACHE
    if _MODEL_INDEX_CACHE is not None and not force_refresh:
        return _MODEL_INDEX_CACHE

    # Read cloth type map once to share across all DeepFashion scans
    cloth_map = _read_deepfashion_cloth_type_map(DEEPFASHION_DIR)

    local_models = _scan_models_from(DATA_DIR / "raw" / "models", "data/raw/models")
    deepfashion_models = _scan_deepfashion_models(DEEPFASHION_DIR, cloth_map)
    deepfashion_antiguo_models = _scan_deepfashion_antiguo_models(DEEPFASHION_ANTIGUO_DIR, cloth_map)
    
    _MODEL_INDEX_CACHE = sorted(
        local_models + deepfashion_models + deepfashion_antiguo_models,
        key=lambda m: (m.get("_source_rank", 1), str(m.get("name", "")).lower()),
    )
    return _MODEL_INDEX_CACHE


@app.get("/")
def ui_index() -> FileResponse:
    index_file = UI_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(index_file))


@app.get("/assets/catalog")
def assets_catalog() -> dict:
    telas_dir = DATA_DIR / "raw" / "telas"

    images: list[str] = []
    all_models = get_model_index()
    # Keep catalog payload light; full search is exposed by /assets/model-search.
    local_models = [m for m in all_models if str(m.get("path", "")).startswith("data/raw/models/")]
    models = local_models[:300]

    if telas_dir.exists():
        for path in sorted(telas_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                images.append(str(Path("data/raw/telas") / path.name))

    target_models = [item for item in models if item["is_target_garment"]]

    return {
        "images": images,
        "models": models,
        "target_models": target_models,
        "indexed_models_total": len(all_models),
        "max_model_file_bytes": MAX_MODEL_FILE_BYTES,
        "deepfashion_models_total": len([m for m in all_models if m.get("source") == "deepfashion"]),
        "deepfashion_antiguo_models_total": len([m for m in all_models if m.get("source") == "deepfashion_antiguo"]),
        "local_models_total": len([m for m in all_models if m.get("source") == "local"]),
    }


@app.get("/assets/model-search")
def model_search(q: str = "", limit: int = 6) -> dict:
    all_models = get_model_index()
    query = q.strip().lower()

    # No query: only send a bounded initial set to avoid overwhelming UI/browser.
    if not query:
        safe_limit = max(1, min(limit, MAX_INITIAL_FRONT_MODELS))
        items = [_public_model_item(m) for m in all_models[:safe_limit]]
        return {
            "query": q,
            "total": len(all_models),
            "limit": safe_limit,
            "items": items,
        }

    # Query active: keep payload very small and stop scanning once enough hits are found.
    safe_limit = max(1, min(limit, MAX_SEARCH_RESULTS))
    matched: list[dict] = []
    total_matches = 0
    for item in all_models:
        if query not in str(item.get("_search_text", "")):
            continue
        total_matches += 1
        if len(matched) < safe_limit:
            matched.append(_public_model_item(item))

    return {
        "query": q,
        "total": total_matches,
        "limit": safe_limit,
        "items": matched,
    }


class GenerateRequest(BaseModel):
    project_id: str = Field(..., description="Unique project identifier")
    image_paths: list[str] = Field(..., min_length=1, max_length=10)
    n_candidates: int = Field(default=8, ge=1, le=32)


class SelectionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_id: str
    model_path: str
    selected_texture_path: str
    output_path: str = "data/exports/output.glb"
    convert_point_cloud: bool = True


class Preview3DRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_id: str
    model_path: str
    texture_path: str
    output_path: str = "data/processed/preview/preview_3d.png"
    convert_point_cloud: bool = True


class FeedbackRequest(BaseModel):
    project_id: str
    candidate_path: str
    label: str = Field(..., pattern="^(approve|reject)$")
    score: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = None


class ImportAssetsRequest(BaseModel):
    source_dir: str | None = None


class UploadFabricRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    data_url: str = Field(..., min_length=32)


def _run_fase21_tryon(request: Fase21TryOnRequest) -> dict:
    # Lazy import so apps/api can boot even when fase2_1 optional deps are missing.
    from fase2_1.core.tryon.pipeline import run_tryon

    result = run_tryon(request)
    return result.model_dump()


def _run_fase21_tryon_batch(request: Fase21TryOnBatchRequest) -> dict:
    from fase2_1.core.tryon.batch import run_tryon_batch

    result = run_tryon_batch(
        input_dir=request.input_dir,
        garment_path=request.garment_path,
        garment_type=request.garment_type,
        output_dir=request.output_dir,
        report_path=request.report_path,
        checklist_path=request.checklist_path,
        limit=request.limit,
    )
    return result.model_dump()


def _run_fase21_manual_eval(request: Fase21TryOnManualEvalRequest) -> dict:
    from fase2_1.core.tryon.review import append_manual_review

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


def _run_fase21_manual_eval_summary(
    report_path: str,
    project_id: str | None = None,
    valid_score_threshold: float = 85.0,
) -> dict:
    from fase2_1.core.tryon.review import summarize_manual_reviews

    return summarize_manual_reviews(
        report_path=report_path,
        project_id=project_id,
        valid_score_threshold=valid_score_threshold,
    )


def _run_fase21_manual_eval_consolidated(
    batch_report_path: str,
    manual_report_path: str,
    project_id: str | None = None,
    consolidated_path: str | None = None,
    valid_score_threshold: float = 85.0,
) -> dict:
    from fase2_1.core.tryon.review import build_consolidated_evaluation_report

    return build_consolidated_evaluation_report(
        batch_report_path=batch_report_path,
        manual_report_path=manual_report_path,
        project_id=project_id,
        consolidated_path=consolidated_path,
        valid_score_threshold=valid_score_threshold,
    )


def _sanitize_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip())
    clean = clean.strip(".-_")
    return clean or "fabric"


def _resolve_model_for_rendering(model_path: str, convert_point_cloud: bool = True) -> tuple[str, dict | None]:
    if not convert_point_cloud or not is_point_cloud_model_path(model_path):
        return model_path, None
    resolved_path, conversion_report = build_point_cloud_proxy_mesh(model_path)
    return resolved_path, conversion_report


def _enrich_model_report_with_conversion(model_report: dict, conversion_report: dict | None) -> dict:
    if not conversion_report:
        return model_report
    enriched = dict(model_report)
    enriched["point_cloud_conversion"] = conversion_report
    if conversion_report.get("converted"):
        enriched["uv_message"] = (
            "Modelo point cloud convertido a malla proxy. "
            "Si UV aparece missing, Blender intentara generar UV automatico al aplicar textura."
        )
    return enriched


@app.get("/health")
def health() -> dict[str, str]:
    try:
        init_database()
        db_status = "ok"
    except Exception:
        db_status = "unavailable"
    return {"status": "ok", "database": db_status}


@app.get("/projects/validate-model")
def validate_model_endpoint(model_path: str) -> dict:
    """Lightweight model validation endpoint for immediate UI feedback."""
    try:
        render_model_path, conversion_report = _resolve_model_for_rendering(model_path, convert_point_cloud=True)
        model_report = validate_model_contract(render_model_path)
        model_report = _enrich_model_report_with_conversion(model_report, conversion_report)
        model_report["input_model_path"] = model_path
        model_report["render_model_path"] = render_model_path
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return model_report


@app.post("/assets/import-from-downloads")
def import_from_downloads(request: ImportAssetsRequest) -> dict:
    source_dir = request.source_dir or os.getenv("DOWNLOADS_IMPORT_DIR", "/mnt/downloads")
    try:
        report = import_assets_from_directory(source_dir=source_dir, data_dir=str(DATA_DIR))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "report": report,
    }


@app.post("/assets/upload-fabric")
def upload_fabric(request: UploadFabricRequest) -> dict:
    m = re.match(r"^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$", request.data_url)
    if not m:
        raise HTTPException(status_code=400, detail="Formato data_url invalido")

    mime = m.group(1).lower()
    payload_b64 = m.group(2)
    ext_map = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    ext = ext_map.get(mime)
    if not ext:
        raise HTTPException(status_code=400, detail=f"Tipo de imagen no soportado: {mime}")

    try:
        raw = base64.b64decode(payload_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Base64 invalido") from exc

    telas_dir = DATA_DIR / "raw" / "telas"
    telas_dir.mkdir(parents=True, exist_ok=True)

    base_name = _sanitize_name(request.name)
    out_name = f"{base_name}{ext}"
    out_path = telas_dir / out_name
    if out_path.exists():
        out_name = f"{base_name}-{int(time.time())}{ext}"
        out_path = telas_dir / out_name

    out_path.write_bytes(raw)

    return {
        "status": "ok",
        "saved_path": str(Path("data/raw/telas") / out_name),
        "size_bytes": len(raw),
    }


@app.post("/fase2_1/tryon/run")
def run_tryon_fase21(request: Fase21TryOnRequest) -> dict:
    """Endpoint puente para ejecutar pipeline separado de Fase 2.1."""
    try:
        payload = _run_fase21_tryon(request)
        track_event(
            event_name="fase2_1_tryon_run",
            payload={
                "status": payload.get("status", "unknown"),
                "image_path": request.image_path,
                "garment_path": request.garment_path,
                "output_path": payload.get("output_path", request.output_path),
                "scale": payload.get("scale", -1),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.post("/fase2_1/tryon/batch")
def run_tryon_fase21_batch(request: Fase21TryOnBatchRequest) -> dict:
    """Ejecuta evaluación batch de Fase 2.1 con reporte JSON estable."""
    try:
        payload = _run_fase21_tryon_batch(request)
        summary = payload.get("summary", {})
        track_event(
            event_name="fase2_1_tryon_batch",
            payload={
                "input_dir": request.input_dir,
                "garment_path": request.garment_path,
                "garment_type": request.garment_type,
                "output_dir": request.output_dir,
                "report_path": payload.get("report_path", request.report_path),
                "total": summary.get("total", 0),
                "ok": summary.get("ok", 0),
                "errors": summary.get("errors", 0),
                "success_rate": summary.get("success_rate", 0),
                "avg_elapsed_ms": summary.get("avg_elapsed_ms", 0),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.post("/fase2_1/tryon/evaluate")
def evaluate_tryon_fase21(request: Fase21TryOnManualEvalRequest) -> dict:
    """Guarda evaluación manual por imagen y calcula score ponderado por checklist."""
    try:
        payload = _run_fase21_manual_eval(request)
        track_event(
            event_name="fase2_1_tryon_manual_eval",
            payload={
                "project_id": request.project_id,
                "image": request.image,
                "output_path": request.output_path or "",
                "score": payload.get("score", 0),
                "saved_path": payload.get("saved_path", request.report_path),
                "criteria_count": len(request.criteria_scores),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.get("/fase2_1/tryon/evaluate-summary")
def evaluate_summary_tryon_fase21(
    report_path: str = "data/processed/fase2_1_eval/manual_reviews.jsonl",
    project_id: str | None = None,
    valid_score_threshold: float = 85.0,
) -> dict:
    """Resume evaluaciones manuales para un proyecto de Fase 2.1."""
    try:
        payload = _run_fase21_manual_eval_summary(
            report_path=report_path,
            project_id=project_id,
            valid_score_threshold=valid_score_threshold,
        )
        track_event(
            event_name="fase2_1_tryon_manual_eval_summary",
            payload={
                "project_id": project_id or "all",
                "report_path": report_path,
                "valid_score_threshold": valid_score_threshold,
                "total": payload.get("total", 0),
                "avg_score": payload.get("avg_score", 0),
                "visually_valid_rate": payload.get("visually_valid_rate", 0),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.get("/fase2_1/tryon/evaluate-consolidated")
def evaluate_consolidated_tryon_fase21(
    batch_report_path: str = "data/processed/fase2_1_eval/report.json",
    manual_report_path: str = "data/processed/fase2_1_eval/manual_reviews.jsonl",
    project_id: str | None = None,
    consolidated_path: str | None = None,
    valid_score_threshold: float = 85.0,
) -> dict:
    """Construye resumen consolidado (batch + evaluacion manual) de Fase 2.1."""
    try:
        payload = _run_fase21_manual_eval_consolidated(
            batch_report_path=batch_report_path,
            manual_report_path=manual_report_path,
            project_id=project_id,
            consolidated_path=consolidated_path,
            valid_score_threshold=valid_score_threshold,
        )
        summary = payload.get("summary", {})
        track_event(
            event_name="fase2_1_tryon_manual_eval_consolidated",
            payload={
                "project_id": project_id or "all",
                "batch_report_path": batch_report_path,
                "manual_report_path": manual_report_path,
                "valid_score_threshold": valid_score_threshold,
                "manual_total": summary.get("manual_total", 0),
                "manual_avg_score": summary.get("manual_avg_score", 0),
                "manual_visually_valid_rate": summary.get("manual_visually_valid_rate", 0),
                "batch_success_rate": summary.get("batch_success_rate", 0),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return payload


@app.post("/projects/generate")
def generate(request: GenerateRequest) -> dict:
    started_at = time.perf_counter()
    try:
        preprocessed = preprocess_images(request.image_paths, request.project_id)
        candidates = generate_texture_candidates(
            image_paths=preprocessed,
            project_id=request.project_id,
            n_candidates=request.n_candidates,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration = time.perf_counter() - started_at
    record_generation_stats(
        project_id=request.project_id,
        n_candidates=len(candidates),
        n_references=len(preprocessed),
        duration_seconds=duration,
    )

    track_event(
        event_name="generate_candidates",
        payload={
            "project_id": request.project_id,
            "n_candidates": len(candidates),
            "n_references": len(preprocessed),
            "duration_seconds": duration,
        },
    )

    return {
        "project_id": request.project_id,
        "engine_mode": os.getenv("TEXTURE_ENGINE", "baseline"),
        "references": preprocessed,
        "total_candidates": len(candidates),
        "candidates": candidates,
    }



@app.post("/projects/select")
def select_texture(request: SelectionRequest) -> dict:
    approved = filter_candidates([request.selected_texture_path], mode="approve")
    if not approved:
        return {
            "project_id": request.project_id,
            "status": "no-approved-texture",
        }

    try:
        render_model_path, conversion_report = _resolve_model_for_rendering(
            request.model_path,
            convert_point_cloud=request.convert_point_cloud,
        )
        model_report = validate_model_contract(render_model_path)
        model_report = _enrich_model_report_with_conversion(model_report, conversion_report)
        model_report["input_model_path"] = request.model_path
        model_report["render_model_path"] = render_model_path
        export_path = apply_texture_and_export(
            model_path=render_model_path,
            texture_path=approved[0],
            output_path=request.output_path,
        )
        track_event(
            event_name="select_texture",
            payload={
                "project_id": request.project_id,
                "status": "exported",
                "model_format": model_report.get("format"),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "project_id": request.project_id,
        "status": "exported",
        "model_report": model_report,
        "output_path": export_path,
    }


@app.post("/projects/feedback")
def save_feedback(request: FeedbackRequest) -> dict:
    append_feedback(
        project_id=request.project_id,
        candidate_path=request.candidate_path,
        label=request.label,
        score=request.score,
        comment=request.comment,
    )
    track_event(
        event_name="feedback",
        payload={
            "project_id": request.project_id,
            "label": request.label,
            "score": request.score if request.score is not None else -1,
        },
    )
    return {
        "project_id": request.project_id,
        "status": "saved",
    }


@app.get("/projects/{project_id}/feedback-summary")
def feedback_summary(project_id: str) -> dict:
    return summarize_feedback(project_id)


@app.get("/projects/{project_id}/ranking")
def feedback_ranking(project_id: str, limit: int = 20) -> dict:
    return ranking_feedback(project_id=project_id, limit=limit)


@app.get("/projects/{project_id}/metrics")
def metrics(project_id: str) -> dict:
    return project_metrics(project_id)


@app.get("/projects/{project_id}/candidates")
def project_candidates(project_id: str) -> dict:
    return {
        "project_id": project_id,
        "candidates": list_project_candidates(project_id),
    }


@app.post("/projects/{project_id}/preview-sheet")
def project_preview_sheet(project_id: str) -> dict:
    try:
        candidates = list_project_candidates(project_id)
        preview_path = build_preview_sheet(project_id, candidates)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "project_id": project_id,
        "preview_path": preview_path,
        "total_candidates": len(candidates),
    }


@app.post("/projects/preview-3d")
def preview_3d(request: Preview3DRequest) -> dict:
    try:
        render_model_path, conversion_report = _resolve_model_for_rendering(
            request.model_path,
            convert_point_cloud=request.convert_point_cloud,
        )
        model_report = validate_model_contract(render_model_path)
        model_report = _enrich_model_report_with_conversion(model_report, conversion_report)
        model_report["input_model_path"] = request.model_path
        model_report["render_model_path"] = render_model_path
        preview_path = generate_3d_preview(
            model_path=render_model_path,
            texture_path=request.texture_path,
            output_path=request.output_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "project_id": request.project_id,
        "preview_path": preview_path,
        "model_report": model_report,
    }
