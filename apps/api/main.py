from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import HTTPException
import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = (BASE_DIR / "data").resolve()
UI_DIR = (BASE_DIR / "apps" / "api" / "ui").resolve()
DEEPFASHION_DIR = (BASE_DIR / "data_deepfashon").resolve()
DEEPFASHION_ANTIGUO_DIR = (BASE_DIR / "data_deepfasho_antiguo").resolve()
FOTOS_PERSONAS_DIR = (BASE_DIR / "fotos_personas").resolve()
LOOKS_DIR = DATA_DIR / "looks"
HUMAN_SHAPE_LAB_DIR = DATA_DIR / "processed" / "human_shape_lab"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)
LOOKS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/artifacts", StaticFiles(directory=str(DATA_DIR)), name="artifacts")
app.mount("/ui-static", StaticFiles(directory=str(UI_DIR)), name="ui-static")
if DEEPFASHION_DIR.exists():
    app.mount("/deepfashion-artifacts", StaticFiles(directory=str(DEEPFASHION_DIR)), name="deepfashion-artifacts")
if DEEPFASHION_ANTIGUO_DIR.exists():
    app.mount("/deepfashion-antiguo-artifacts", StaticFiles(directory=str(DEEPFASHION_ANTIGUO_DIR)), name="deepfashion-antiguo-artifacts")
if FOTOS_PERSONAS_DIR.exists():
    app.mount("/fotos-personas-static", StaticFiles(directory=str(FOTOS_PERSONAS_DIR)), name="fotos-personas-static")


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
    is_point_cloud = ("/point_cloud/" in logical_path or "/pointcloud/" in logical_path or logical_path.endswith(".ply")) and "/mesh/" not in logical_path
    
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
    
    source_rank = {"local": 0, "deepfashion_antiguo": 1, "deepfashion": 2, "unknown": 3}.get(source, 3)
    
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
    """Index garment OBJ meshes from the older DeepFashion dataset.

    Uses the same cloth_type_list.txt for garment classification.
    The directory structure is: mesh/<id>-<pose>/model_cleaned.obj
    The garment ID is the numeric prefix before the first '-' in the folder name.
    """
    out: list[dict] = []
    mesh_dir = base_dir / "mesh"
    if not mesh_dir.exists():
        return out

    for path in sorted(mesh_dir.rglob("*.obj")):
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        if size_bytes > MAX_MODEL_FILE_BYTES:
            continue

        rel = path.relative_to(base_dir).as_posix()  # mesh/<id>-<pose>/model_cleaned.obj
        folder = rel.split("/", 2)[1] if rel.startswith("mesh/") and "/" in rel else ""
        top_id = folder.split("-")[0] if "-" in folder else folder
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


class Fase23BodyReconstructRequest(BaseModel):
    image_path: str
    case_id: str | None = None
    checkpoint_path: str | None = None
    mhr_path: str | None = None
    sam3d_root: str = "sam-3d-body"
    sam3d_python_bin: str = "python"
    detector_name: str = ""
    bbox_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    use_mask: bool = False
    export_glb: bool = True
    use_mock: bool = False


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


def _run_deepfashion_coverage(include_samples: bool = True, sample_limit: int = 25) -> dict:
    from scripts.analyze_deepfashion_coverage import build_coverage_report

    return build_coverage_report(
        old_root=DEEPFASHION_ANTIGUO_DIR,
        new_root=DEEPFASHION_DIR,
        include_samples=include_samples,
        sample_limit=max(1, int(sample_limit)),
    )


def _to_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path)


def _resolve_input_path(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    return (BASE_DIR / raw).resolve()


def _path_to_repo_relative(path_str: str | None) -> str | None:
    if not path_str:
        return None
    raw = Path(path_str)
    resolved = raw if raw.is_absolute() else (BASE_DIR / raw)
    return _to_repo_relative(resolved)


def _artifact_url_from_repo_relative(path_str: str | None) -> str | None:
    if not path_str:
        return None
    clean = str(path_str).replace("\\", "/")
    if clean.startswith("data/"):
        return f"/artifacts/{clean[5:]}"
    return None


def _resolve_fase23_output_dir(case_name: str) -> tuple[Path, str]:
    primary = DATA_DIR / "processed" / "fase2_3" / case_name
    try:
        primary.mkdir(parents=True, exist_ok=True)
        return primary, "default"
    except PermissionError:
        fallback = DATA_DIR / "processed" / "fase2_3_user" / case_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback, "fallback_user"


def _write_mock_body_preview(image_path: Path, preview_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw

        base = Image.open(image_path).convert("RGBA")
        width, height = base.size
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        center_x = width * 0.5
        top_y = height * 0.16
        head_w = width * 0.1
        head_h = height * 0.1
        shoulder_y = height * 0.27
        hip_y = height * 0.55
        knee_y = height * 0.76
        foot_y = height * 0.93
        shoulder_half = width * 0.12
        hip_half = width * 0.09

        mannequin_fill = (195, 225, 255, 126)
        mannequin_outline = (215, 240, 255, 220)
        accent = (62, 207, 207, 235)

        draw.ellipse(
            [
                center_x - head_w / 2,
                top_y,
                center_x + head_w / 2,
                top_y + head_h,
            ],
            fill=mannequin_fill,
            outline=mannequin_outline,
            width=4,
        )
        draw.polygon(
            [
                (center_x - shoulder_half, shoulder_y),
                (center_x + shoulder_half, shoulder_y),
                (center_x + hip_half, hip_y),
                (center_x - hip_half, hip_y),
            ],
            fill=mannequin_fill,
            outline=mannequin_outline,
        )
        draw.line(
            [(center_x - shoulder_half, shoulder_y + 8), (center_x - width * 0.22, height * 0.49)],
            fill=mannequin_outline,
            width=max(8, int(width * 0.018)),
        )
        draw.line(
            [(center_x + shoulder_half, shoulder_y + 8), (center_x + width * 0.22, height * 0.49)],
            fill=mannequin_outline,
            width=max(8, int(width * 0.018)),
        )
        draw.line(
            [(center_x - hip_half / 1.5, hip_y), (center_x - width * 0.08, knee_y), (center_x - width * 0.11, foot_y)],
            fill=mannequin_outline,
            width=max(10, int(width * 0.022)),
        )
        draw.line(
            [(center_x + hip_half / 1.5, hip_y), (center_x + width * 0.1, knee_y), (center_x + width * 0.13, foot_y)],
            fill=mannequin_outline,
            width=max(10, int(width * 0.022)),
        )
        draw.rounded_rectangle(
            [width * 0.05, height * 0.04, width * 0.32, height * 0.12],
            radius=14,
            fill=(11, 15, 20, 210),
            outline=accent,
            width=2,
        )
        draw.text((width * 0.08, height * 0.06), "MOCK BODY", fill=accent)

        merged = Image.alpha_composite(base, overlay).convert("RGB")
        merged.save(preview_path, quality=95)
    except Exception:
        shutil.copyfile(image_path, preview_path)


def _build_fase23_mock_payload(
    image_path: Path,
    case_name: str,
    out_dir: Path,
    output_strategy: str,
    *,
    fallback_reason: str | None = None,
) -> dict:
    run_meta_path = out_dir / "run_meta.json"
    preview_path = out_dir / f"{image_path.stem}_mock_preview.jpg"
    _write_mock_body_preview(image_path, preview_path)
    preview_rel = _to_repo_relative(preview_path)
    mode = "mock"
    if fallback_reason:
        mode = "mock_fallback"

    run_meta = {
        "status": "ok",
        "mode": mode,
        "case_id": case_name,
        "input_image": _to_repo_relative(image_path),
        "output_dir": _to_repo_relative(out_dir),
        "output_strategy": output_strategy,
        "preview_path": preview_rel,
        "fallback_reason": fallback_reason,
    }
    run_meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "status": "ok",
        "mode": mode,
        "case_id": case_name,
        "input_image": _to_repo_relative(image_path),
        "output_dir": _to_repo_relative(out_dir),
        "output_strategy": output_strategy,
        "mesh_paths": [],
        "preview_path": preview_rel,
        "preview_url": _artifact_url_from_repo_relative(preview_rel),
        "meta_path": _to_repo_relative(run_meta_path),
    }
    if fallback_reason:
        payload["warning"] = fallback_reason
    return payload


def _resolve_fase23_runtime_config(request: Fase23BodyReconstructRequest) -> dict:
    sam3d_root = request.sam3d_root or os.getenv("SAM3D_ROOT", "sam-3d-body")
    sam3d_python_bin = request.sam3d_python_bin or os.getenv("SAM3D_PYTHON_BIN", "python")
    detector_name = request.detector_name or os.getenv("SAM3D_DETECTOR_NAME", "")
    checkpoint_path = request.checkpoint_path or os.getenv("SAM3D_CHECKPOINT_PATH")
    mhr_path = request.mhr_path or os.getenv("SAM3D_MHR_PATH")

    return {
        "sam3d_root": sam3d_root,
        "sam3d_python_bin": sam3d_python_bin,
        "detector_name": detector_name,
        "checkpoint_path": checkpoint_path,
        "mhr_path": mhr_path,
    }


def _run_fase23_body_reconstruct(request: Fase23BodyReconstructRequest) -> dict:
    image_path = _resolve_input_path(request.image_path)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Imagen no encontrada: {request.image_path}")

    runtime_cfg = _resolve_fase23_runtime_config(request)

    case_name = _sanitize_name(request.case_id or f"sam3d-{image_path.stem}-{int(time.time())}")
    out_dir, output_strategy = _resolve_fase23_output_dir(case_name)

    run_meta_path = out_dir / "run_meta.json"
    missing_checkpoint = not runtime_cfg["checkpoint_path"]
    missing_mhr = not runtime_cfg["mhr_path"]

    if request.use_mock:
        return _build_fase23_mock_payload(
            image_path=image_path,
            case_name=case_name,
            out_dir=out_dir,
            output_strategy=output_strategy,
        )

    if missing_checkpoint or missing_mhr:
        missing_parts: list[str] = []
        if missing_checkpoint:
            missing_parts.append("checkpoint_path")
        if missing_mhr:
            missing_parts.append("mhr_path")
        return _build_fase23_mock_payload(
            image_path=image_path,
            case_name=case_name,
            out_dir=out_dir,
            output_strategy=output_strategy,
            fallback_reason=(
                "Fase 2.3 ejecuto fallback mock porque faltan "
                + ", ".join(missing_parts)
                + ". Define SAM3D_CHECKPOINT_PATH/SAM3D_MHR_PATH para reconstruccion real."
            ),
        )

    script_path = BASE_DIR / "scripts" / "run_sam3d_body_single.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Script no encontrado: {script_path}")

    cmd = [
        runtime_cfg["sam3d_python_bin"],
        str(script_path),
        "--repo-root",
        str(BASE_DIR),
        "--sam3d-root",
        runtime_cfg["sam3d_root"],
        "--image-path",
        str(image_path),
        "--output-dir",
        str(out_dir),
        "--checkpoint-path",
        runtime_cfg["checkpoint_path"],
        "--mhr-path",
        runtime_cfg["mhr_path"],
        "--detector-name",
        runtime_cfg["detector_name"],
        "--bbox-thresh",
        str(request.bbox_thresh),
    ]
    if request.use_mask:
        cmd.append("--use-mask")
    if request.export_glb:
        cmd.append("--export-glb")
    else:
        cmd.append("--no-export-glb")

    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, check=False)

    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "Error ejecutando SAM 3D Body"
        raise RuntimeError(msg)

    result_payload: dict = {}
    result_json = out_dir / "result.json"
    if result_json.exists():
        try:
            result_payload = json.loads(result_json.read_text(encoding="utf-8"))
        except Exception:
            result_payload = {}

    mesh_paths_from_result = [p for p in (result_payload.get("mesh_paths") or []) if p]
    glb_paths_from_result = [p for p in (result_payload.get("glb_paths") or []) if p]
    overlay_path_from_result = result_payload.get("overlay_path")

    mesh_paths = [_path_to_repo_relative(p) for p in mesh_paths_from_result]
    mesh_paths = [p for p in mesh_paths if p]
    if not mesh_paths:
        mesh_paths = [_to_repo_relative(p) for p in sorted(out_dir.glob("*.ply"))]

    glb_paths = [_path_to_repo_relative(p) for p in glb_paths_from_result]
    glb_paths = [p for p in glb_paths if p]
    if not glb_paths:
        glb_paths = [_to_repo_relative(p) for p in sorted(out_dir.glob("*.glb"))]

    preview_rel = _path_to_repo_relative(overlay_path_from_result)
    if not preview_rel:
        previews = sorted(out_dir.glob("*.jpg"))
        preview_rel = _to_repo_relative(previews[0]) if previews else None

    run_meta = {
        "status": "ok",
        "mode": "sam3d_body",
        "case_id": case_name,
        "input_image": _to_repo_relative(image_path),
        "output_dir": _to_repo_relative(out_dir),
        "output_strategy": output_strategy,
        "mesh_paths": mesh_paths,
        "glb_paths": glb_paths,
        "preview_path": preview_rel,
        "n_people": result_payload.get("n_people", len(mesh_paths)),
        "device": result_payload.get("device", "unknown"),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]) if proc.stderr else "",
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]) if proc.stdout else "",
    }
    run_meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "status": "ok",
        "mode": "sam3d_body",
        "case_id": case_name,
        "input_image": _to_repo_relative(image_path),
        "output_dir": _to_repo_relative(out_dir),
        "output_strategy": output_strategy,
        "mesh_paths": mesh_paths,
        "glb_paths": glb_paths,
        "preview_path": preview_rel,
        "n_people": result_payload.get("n_people", len(mesh_paths)),
        "device": result_payload.get("device", "unknown"),
        "meta_path": _to_repo_relative(run_meta_path),
        "runtime": {
            "sam3d_root": runtime_cfg["sam3d_root"],
            "sam3d_python_bin": runtime_cfg["sam3d_python_bin"],
            "detector_name": runtime_cfg["detector_name"],
            "checkpoint_path": runtime_cfg["checkpoint_path"],
            "mhr_path": runtime_cfg["mhr_path"],
            "use_mask": request.use_mask,
            "bbox_thresh": request.bbox_thresh,
            "export_glb": request.export_glb,
        },
    }
    if preview_rel:
        payload["preview_url"] = _artifact_url_from_repo_relative(preview_rel)
    if glb_paths:
        payload["avatar_glb_path"] = glb_paths[0]
    return payload


@app.get("/fase2_3/body/preflight")
def fase23_body_preflight(
    checkpoint_path: str | None = None,
    mhr_path: str | None = None,
    sam3d_root: str | None = None,
    sam3d_python_bin: str | None = None,
    detector_name: str | None = None,
) -> dict:
    runtime_cfg = {
        "sam3d_root": sam3d_root or os.getenv("SAM3D_ROOT", "sam-3d-body"),
        "sam3d_python_bin": sam3d_python_bin or os.getenv("SAM3D_PYTHON_BIN", "python"),
        "detector_name": detector_name or os.getenv("SAM3D_DETECTOR_NAME", ""),
        "checkpoint_path": checkpoint_path or os.getenv("SAM3D_CHECKPOINT_PATH"),
        "mhr_path": mhr_path or os.getenv("SAM3D_MHR_PATH"),
    }

    script_path = BASE_DIR / "scripts" / "run_sam3d_body_single.py"
    sam3d_root_path = _resolve_input_path(runtime_cfg["sam3d_root"])
    checkpoint_abs = _resolve_input_path(runtime_cfg["checkpoint_path"]) if runtime_cfg["checkpoint_path"] else None
    mhr_abs = _resolve_input_path(runtime_cfg["mhr_path"]) if runtime_cfg["mhr_path"] else None
    output_root = DATA_DIR / "processed" / "fase2_3"
    fallback_output_root = DATA_DIR / "processed" / "fase2_3_user"

    output_dir_writable = True
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception:
        output_dir_writable = False

    fallback_output_dir_writable = True
    try:
        fallback_output_root.mkdir(parents=True, exist_ok=True)
        probe = fallback_output_root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception:
        fallback_output_dir_writable = False

    checks = {
        "script_exists": script_path.exists(),
        "sam3d_root_exists": sam3d_root_path.exists(),
        "checkpoint_exists": bool(checkpoint_abs and checkpoint_abs.exists()),
        "mhr_exists": bool(mhr_abs and mhr_abs.exists()),
        "output_dir_writable": output_dir_writable,
        "fallback_output_dir_writable": fallback_output_dir_writable,
    }

    mock_ready = checks["output_dir_writable"] or checks["fallback_output_dir_writable"]
    real_ready = (
        checks["script_exists"]
        and checks["sam3d_root_exists"]
        and checks["checkpoint_exists"]
        and checks["mhr_exists"]
        and mock_ready
    )

    issues: list[str] = []
    warnings: list[str] = []

    if not checks["script_exists"]:
        warnings.append(f"Script no encontrado: {_to_repo_relative(script_path)}")
    if not checks["sam3d_root_exists"]:
        warnings.append(f"Repositorio sam-3d-body no encontrado: {runtime_cfg['sam3d_root']}")
    if not checks["checkpoint_exists"]:
        warnings.append("Checkpoint no encontrado. Define SAM3D_CHECKPOINT_PATH o envialo en request para reconstruccion real.")
    if not checks["mhr_exists"]:
        warnings.append("MHR path no encontrado. Define SAM3D_MHR_PATH o envialo en request para reconstruccion real.")
    if not mock_ready:
        issues.append("No hay carpeta de salida escribible para Fase 2.3 (ni primaria ni fallback).")

    status = "error" if issues else ("ok" if real_ready else "ok_mock_only")

    return {
        "status": status,
        "checks": checks,
        "issues": issues,
        "warnings": warnings,
        "capabilities": {
            "mock_ready": mock_ready,
            "real_ready": real_ready,
        },
        "runtime": {
            "sam3d_root": runtime_cfg["sam3d_root"],
            "sam3d_python_bin": runtime_cfg["sam3d_python_bin"],
            "detector_name": runtime_cfg["detector_name"],
            "checkpoint_path": _to_repo_relative(checkpoint_abs) if checkpoint_abs else None,
            "mhr_path": _to_repo_relative(mhr_abs) if mhr_abs else None,
        },
    }


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


@app.get("/assets/deepfashion-coverage")
def deepfashion_coverage(refresh: bool = False, include_samples: bool = True, sample_limit: int = 25) -> dict:
    report_path = DATA_DIR / "processed" / "deepfashion" / "coverage_report.json"

    if not refresh and report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["cached"] = True
            payload["report_path"] = str(report_path)
            return payload
        except Exception:
            # If cached file is corrupted, regenerate below.
            pass

    payload = _run_deepfashion_coverage(include_samples=include_samples, sample_limit=sample_limit)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["cached"] = False
    payload["report_path"] = str(report_path)
    return payload


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


@app.post("/fase2_3/body/reconstruct")
def run_body_reconstruct_fase23(request: Fase23BodyReconstructRequest) -> dict:
    """Genera reconstruccion 3D de cuerpo desde una foto usando SAM 3D Body."""
    try:
        payload = _run_fase23_body_reconstruct(request)
        track_event(
            event_name="fase2_3_body_reconstruct",
            payload={
                "status": payload.get("status", "unknown"),
                "mode": payload.get("mode", "unknown"),
                "case_id": payload.get("case_id", ""),
                "input_image": payload.get("input_image", request.image_path),
                "output_dir": payload.get("output_dir", ""),
                "mesh_count": len(payload.get("mesh_paths", [])),
            },
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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


# ---------------------------------------------------------------------------
# Looks — guardar modelos creados para prueba virtual
# ---------------------------------------------------------------------------

class SaveLookRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(..., min_length=1, max_length=100)
    model_path: str
    texture_path: str | None = None
    garment_type: str = "other"
    project_id: str | None = None
    notes: str | None = None


class TryOnApplyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    look_id: str
    foto_nombre: str


class HumanShapeBootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    image_names: list[str] | None = None
    force_regenerate_auto_masks: bool = False


class HumanShapeSaveMaskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sample_id: str
    mask_data_url: str = Field(..., min_length=32)


class HumanShapeTrainRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    min_iou: float = Field(default=0.05, ge=0.0, le=1.0)
    max_samples: int | None = Field(default=None, ge=1)
    candidate_thresholds: list[float] | None = None


class HumanShapeKeypointPoint(BaseModel):
    x: float  # normalized 0-1 relative to image width
    y: float  # normalized 0-1 relative to image height
    visible: bool = True
    optional: bool = False


class HumanShapeSaveKeypointsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    keypoints: dict[str, dict]  # keypoint_name -> {x, y, visible, optional}
    accepted_parts: list[str] | None = None   # parts marked accepted in review mode
    rejected_parts: list[str] | None = None   # parts marked rejected


class HumanShapeObjectiveSnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_only_corrected: bool = True


class HumanShapeIterativeTrainRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    epochs: int = Field(default=1, ge=1, le=200)
    learning_rate: float = Field(default=0.35, ge=0.001, le=1.0)
    reset_model: bool = False
    use_objective_snapshot: bool = True


# ---------------------------------------------------------------------------
# Body keypoints definitions
# ---------------------------------------------------------------------------

BODY_KEYPOINT_NAMES: list[str] = [
    "crown", "chin",
    "neck_base",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand_tip", "right_hand_tip",
    "hip_center",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_toe", "right_toe",
]

BODY_PARTS_SEGMENTS: list[dict] = [
    {"id": "head",              "label": "Cabeza",     "kp_a": "crown",          "kp_b": "chin",           "color": "#ff6b6b", "optional": False, "side": None},
    {"id": "neck",              "label": "Cuello",     "kp_a": "chin",           "kp_b": "neck_base",      "color": "#ffa94d", "optional": False, "side": None},
    {"id": "left_shoulder",     "label": "Hombro Izq", "kp_a": "neck_base",      "kp_b": "left_shoulder",  "color": "#ffe066", "optional": False, "side": "left"},
    {"id": "right_shoulder",    "label": "Hombro Der", "kp_a": "neck_base",      "kp_b": "right_shoulder", "color": "#ffe066", "optional": False, "side": "right"},
    {"id": "torso",             "label": "Tronco",     "kp_a": "neck_base",      "kp_b": "hip_center",     "color": "#a9e34b", "optional": False, "side": None},
    {"id": "left_arm",          "label": "Brazo Izq",  "kp_a": "left_shoulder",  "kp_b": "left_elbow",     "color": "#69db7c", "optional": False, "side": "left"},
    {"id": "right_arm",         "label": "Brazo Der",  "kp_a": "right_shoulder", "kp_b": "right_elbow",    "color": "#69db7c", "optional": False, "side": "right"},
    {"id": "left_forearm",      "label": "Antebrazo Izq", "kp_a": "left_elbow",  "kp_b": "left_wrist",     "color": "#38d9a9", "optional": False, "side": "left"},
    {"id": "right_forearm",     "label": "Antebrazo Der", "kp_a": "right_elbow", "kp_b": "right_wrist",    "color": "#38d9a9", "optional": False, "side": "right"},
    {"id": "left_hand",         "label": "Mano Izq",   "kp_a": "left_wrist",    "kp_b": "left_hand_tip",  "color": "#74c0fc", "optional": True,  "side": "left"},
    {"id": "right_hand",        "label": "Mano Der",   "kp_a": "right_wrist",   "kp_b": "right_hand_tip", "color": "#74c0fc", "optional": True,  "side": "right"},
    {"id": "waist",             "label": "Cintura",    "kp_a": "left_hip",      "kp_b": "right_hip",      "color": "#e599f7", "optional": False, "side": None},
    {"id": "left_leg",          "label": "Pierna Izq", "kp_a": "left_hip",      "kp_b": "left_knee",      "color": "#da77f2", "optional": False, "side": "left"},
    {"id": "right_leg",         "label": "Pierna Der", "kp_a": "right_hip",     "kp_b": "right_knee",     "color": "#da77f2", "optional": False, "side": "right"},
    {"id": "left_calf",         "label": "Pantorrilla Izq", "kp_a": "left_knee", "kp_b": "left_ankle",    "color": "#f783ac", "optional": False, "side": "left"},
    {"id": "right_calf",        "label": "Pantorrilla Der", "kp_a": "right_knee", "kp_b": "right_ankle",  "color": "#f783ac", "optional": False, "side": "right"},
    {"id": "left_foot",         "label": "Pie Izq",    "kp_a": "left_ankle",    "kp_b": "left_toe",       "color": "#ffa8a8", "optional": True,  "side": "left"},
    {"id": "right_foot",        "label": "Pie Der",    "kp_a": "right_ankle",   "kp_b": "right_toe",      "color": "#ffa8a8", "optional": True,  "side": "right"},
]


def _geometric_keypoints() -> dict[str, dict]:
    """Return normalized (0-1) keypoint estimates for a standing person centred in frame."""
    kp: dict[str, dict] = {
        "crown":           {"x": 0.50, "y": 0.04, "visible": True, "optional": False},
        "chin":            {"x": 0.50, "y": 0.14, "visible": True, "optional": False},
        "neck_base":       {"x": 0.50, "y": 0.21, "visible": True, "optional": False},
        "left_shoulder":   {"x": 0.35, "y": 0.24, "visible": True, "optional": False},
        "right_shoulder":  {"x": 0.65, "y": 0.24, "visible": True, "optional": False},
        "left_elbow":      {"x": 0.27, "y": 0.40, "visible": True, "optional": False},
        "right_elbow":     {"x": 0.73, "y": 0.40, "visible": True, "optional": False},
        "left_wrist":      {"x": 0.23, "y": 0.55, "visible": True, "optional": False},
        "right_wrist":     {"x": 0.77, "y": 0.55, "visible": True, "optional": False},
        "left_hand_tip":   {"x": 0.21, "y": 0.61, "visible": True, "optional": True},
        "right_hand_tip":  {"x": 0.79, "y": 0.61, "visible": True, "optional": True},
        "hip_center":      {"x": 0.50, "y": 0.56, "visible": True, "optional": False},
        "left_hip":        {"x": 0.41, "y": 0.56, "visible": True, "optional": False},
        "right_hip":       {"x": 0.59, "y": 0.56, "visible": True, "optional": False},
        "left_knee":       {"x": 0.40, "y": 0.73, "visible": True, "optional": False},
        "right_knee":      {"x": 0.60, "y": 0.73, "visible": True, "optional": False},
        "left_ankle":      {"x": 0.41, "y": 0.89, "visible": True, "optional": False},
        "right_ankle":     {"x": 0.59, "y": 0.89, "visible": True, "optional": False},
        "left_toe":        {"x": 0.39, "y": 0.95, "visible": True, "optional": True},
        "right_toe":       {"x": 0.61, "y": 0.95, "visible": True, "optional": True},
    }
    return kp


def _mediapipe_keypoints(image_path: Path) -> dict[str, dict] | None:
    """Try MediaPipe Pose to estimate keypoints. Returns None if unavailable."""
    try:
        import mediapipe as mp  # type: ignore
        import numpy as np
        from PIL import Image as PILImage

        mp_pose = mp.solutions.pose
        img = PILImage.open(image_path).convert("RGB")
        import io, cv2  # noqa: E401
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        arr = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
            result = pose.process(arr_rgb)

        if not result or not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark
        L = mp_pose.PoseLandmark

        def pt(idx: int, vis_thresh: float = 0.4) -> dict:
            p = lm[idx]
            return {"x": float(p.x), "y": float(p.y), "visible": float(p.visibility) >= vis_thresh, "optional": False}

        kp = {
            "crown":          {"x": float(lm[L.NOSE].x), "y": max(0.0, float(lm[L.NOSE].y) - 0.08), "visible": True, "optional": False},
            "chin":           {"x": float((lm[L.LEFT_MOUTH_CORNER].x + lm[L.RIGHT_MOUTH_CORNER].x) / 2), "y": float(max(lm[L.LEFT_MOUTH_CORNER].y, lm[L.RIGHT_MOUTH_CORNER].y)), "visible": True, "optional": False},
            "neck_base":      {"x": float((lm[L.LEFT_SHOULDER].x + lm[L.RIGHT_SHOULDER].x) / 2), "y": float((lm[L.LEFT_SHOULDER].y + lm[L.RIGHT_SHOULDER].y) / 2), "visible": True, "optional": False},
            "left_shoulder":  pt(L.LEFT_SHOULDER),
            "right_shoulder": pt(L.RIGHT_SHOULDER),
            "left_elbow":     pt(L.LEFT_ELBOW),
            "right_elbow":    pt(L.RIGHT_ELBOW),
            "left_wrist":     pt(L.LEFT_WRIST),
            "right_wrist":    pt(L.RIGHT_WRIST),
            "left_hand_tip":  {**pt(L.LEFT_INDEX), "optional": True},
            "right_hand_tip": {**pt(L.RIGHT_INDEX), "optional": True},
            "hip_center":     {"x": float((lm[L.LEFT_HIP].x + lm[L.RIGHT_HIP].x) / 2), "y": float((lm[L.LEFT_HIP].y + lm[L.RIGHT_HIP].y) / 2), "visible": True, "optional": False},
            "left_hip":       pt(L.LEFT_HIP),
            "right_hip":      pt(L.RIGHT_HIP),
            "left_knee":      pt(L.LEFT_KNEE),
            "right_knee":     pt(L.RIGHT_KNEE),
            "left_ankle":     pt(L.LEFT_ANKLE),
            "right_ankle":    pt(L.RIGHT_ANKLE),
            "left_toe":       {**pt(L.LEFT_FOOT_INDEX), "optional": True},
            "right_toe":      {**pt(L.RIGHT_FOOT_INDEX), "optional": True},
        }
        return kp
    except Exception:
        return None


def _look_manifest_path(look_id: str) -> Path:
    return LOOKS_DIR / look_id / "manifest.json"


def _human_shape_lab_dirs() -> dict[str, Path]:
    root = HUMAN_SHAPE_LAB_DIR
    dirs = {
        "root": root,
        "images": root / "images",
        "auto_masks": root / "auto_masks",
        "masks": root / "masks",
        "samples": root / "samples",
        "dataset": root / "dataset",
        "models": root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _human_shape_current_model_config_path() -> Path:
    return _human_shape_lab_dirs()["models"] / "segmentation_model_config.json"


def _human_shape_objective_snapshot_path() -> Path:
    return _human_shape_lab_dirs()["models"] / "keypoint_objective_snapshot.json"


def _human_shape_keypoint_refiner_path() -> Path:
    return _human_shape_lab_dirs()["models"] / "keypoint_refiner_v1.json"


def _human_shape_sample_id(image_name: str) -> str:
    src = Path(image_name)
    return _sanitize_name(f"{src.stem}-{src.suffix.lower().replace('.', '')}")


def _human_shape_sample_manifest_path(sample_id: str) -> Path:
    return _human_shape_lab_dirs()["samples"] / f"{sample_id}.json"


def _human_shape_image_to_mask_name(image_name: str) -> str:
    return f"{Path(image_name).stem}.png"


def _human_shape_sample_payload(manifest: dict) -> dict:
    image_rel = manifest.get("image_path")
    auto_mask_rel = manifest.get("auto_mask_path")
    corrected_mask_rel = manifest.get("mask_path")
    current_mask_rel = corrected_mask_rel if corrected_mask_rel else auto_mask_rel
    image_url = _artifact_url_from_repo_relative(image_rel)
    auto_mask_url = _artifact_url_from_repo_relative(auto_mask_rel)
    corrected_mask_url = _artifact_url_from_repo_relative(corrected_mask_rel)
    current_mask_url = _artifact_url_from_repo_relative(current_mask_rel)
    return {
        **manifest,
        "image_url": image_url,
        "auto_mask_url": auto_mask_url,
        "corrected_mask_url": corrected_mask_url,
        "current_mask_url": current_mask_url,
        "is_corrected": bool(manifest.get("is_corrected")),
        "has_keypoints": bool(manifest.get("has_keypoints")),
        "accepted_parts": manifest.get("accepted_parts") or [],
        "rejected_parts": manifest.get("rejected_parts") or [],
    }


def _save_human_shape_sample_manifest(manifest: dict) -> dict:
    sample_id = str(manifest["sample_id"])
    path = _human_shape_sample_manifest_path(sample_id)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _load_human_shape_sample_manifest(sample_id: str) -> dict:
    path = _human_shape_sample_manifest_path(sample_id)
    if not path.exists():
        raise FileNotFoundError(f"Sample no encontrado: {sample_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _list_human_shape_samples() -> list[dict]:
    samples: list[dict] = []
    sample_dir = _human_shape_lab_dirs()["samples"]
    for manifest_path in sorted(sample_dir.glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            samples.append(_human_shape_sample_payload(manifest))
        except Exception:
            continue
    samples.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return samples


def _bootstrap_human_shape_sample(image_name: str, force_regenerate_auto_masks: bool = False) -> dict:
    from fase2_2.core.tryon.parsing_adapter import segment_person_v2

    source = FOTOS_PERSONAS_DIR / image_name
    if not source.exists() or not source.is_file() or source.suffix.lower() not in FOTO_EXT:
        raise FileNotFoundError(f"Foto invalida para laboratorio: {image_name}")

    dirs = _human_shape_lab_dirs()
    sample_id = _human_shape_sample_id(image_name)
    image_dst = dirs["images"] / source.name
    auto_mask_dst = dirs["auto_masks"] / _human_shape_image_to_mask_name(source.name)
    mask_dst = dirs["masks"] / _human_shape_image_to_mask_name(source.name)
    shutil.copy2(source, image_dst)

    model_config_path = _human_shape_current_model_config_path()
    if force_regenerate_auto_masks or not auto_mask_dst.exists():
        seg = segment_person_v2(
            image_path=str(image_dst),
            output_mask_path=str(auto_mask_dst),
            model_config_path=str(model_config_path) if model_config_path.exists() else None,
        )
        auto_backend = seg.backend
        auto_note = seg.note
        auto_mask_coverage = seg.mask_coverage
    else:
        auto_backend = "cached"
        auto_note = "auto mask reutilizada"
        auto_mask_coverage = None

    if force_regenerate_auto_masks or not mask_dst.exists():
        shutil.copy2(auto_mask_dst, mask_dst)
        is_corrected = False
    else:
        is_corrected = True

    manifest = {
        "sample_id": sample_id,
        "image_name": source.name,
        "source_path": _to_repo_relative(source),
        "image_path": _to_repo_relative(image_dst),
        "auto_mask_path": _to_repo_relative(auto_mask_dst),
        "mask_path": _to_repo_relative(mask_dst),
        "is_corrected": is_corrected,
        "auto_backend": auto_backend,
        "auto_note": auto_note,
        "auto_mask_coverage": auto_mask_coverage,
        "updated_at": int(time.time()),
    }
    return _save_human_shape_sample_manifest(manifest)


def _decode_image_data_url(data_url: str) -> bytes:
    m = re.match(r"^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$", data_url)
    if not m:
        raise ValueError("Formato data_url invalido")
    try:
        return base64.b64decode(m.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Base64 invalido") from exc


def _human_shape_lab_status() -> dict:
    model_config_path = _human_shape_current_model_config_path()
    objective_snapshot_path = _human_shape_objective_snapshot_path()
    keypoint_refiner_path = _human_shape_keypoint_refiner_path()
    model_config = None
    objective_snapshot = None
    keypoint_refiner = None
    if model_config_path.exists():
        try:
            model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        except Exception:
            model_config = None
    if objective_snapshot_path.exists():
        try:
            objective_snapshot = json.loads(objective_snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            objective_snapshot = None
    if keypoint_refiner_path.exists():
        try:
            keypoint_refiner = json.loads(keypoint_refiner_path.read_text(encoding="utf-8"))
        except Exception:
            keypoint_refiner = None

    fotos_total = 0
    if FOTOS_PERSONAS_DIR.exists():
        fotos_total = len([f for f in FOTOS_PERSONAS_DIR.iterdir() if f.is_file() and f.suffix.lower() in FOTO_EXT])

    samples = _list_human_shape_samples()
    return {
        "status": "ok",
        "source": {
            "fotos_personas_dir": str(FOTOS_PERSONAS_DIR),
            "available_photos_total": fotos_total,
        },
        "lab": {
            "root": _to_repo_relative(HUMAN_SHAPE_LAB_DIR),
            "samples_total": len(samples),
            "corrected_total": len([s for s in samples if s.get("is_corrected")]),
            "model_config_path": _to_repo_relative(model_config_path) if model_config_path.exists() else None,
            "model_config": model_config,
            "objective_snapshot_path": _to_repo_relative(objective_snapshot_path) if objective_snapshot_path.exists() else None,
            "objective_snapshot": objective_snapshot,
            "keypoint_refiner_path": _to_repo_relative(keypoint_refiner_path) if keypoint_refiner_path.exists() else None,
            "keypoint_refiner": keypoint_refiner,
        },
        "samples": samples,
    }


def _keypoint_template() -> dict[str, dict]:
    return {name: {"dx": 0.0, "dy": 0.0} for name in BODY_KEYPOINT_NAMES}


def _load_keypoint_refiner_model() -> dict:
    path = _human_shape_keypoint_refiner_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("offsets"), dict):
                return data
        except Exception:
            pass
    return {
        "model_name": "keypoint_refiner_v1",
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "iterations": 0,
        "offsets": _keypoint_template(),
        "metrics": {},
    }


def _save_keypoint_refiner_model(model: dict) -> dict:
    model["updated_at"] = int(time.time())
    path = _human_shape_keypoint_refiner_path()
    path.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return model


def _estimate_base_keypoints_for_manifest(manifest: dict) -> tuple[dict[str, dict], str]:
    image_rel = manifest.get("image_path")
    if image_rel:
        image_path = _resolve_input_path(image_rel)
        if image_path.exists():
            kp = _mediapipe_keypoints(image_path)
            if kp:
                return kp, "mediapipe_pose"
    return _geometric_keypoints(), "geometric"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _apply_refiner(base: dict[str, dict], model: dict) -> dict[str, dict]:
    offsets = model.get("offsets") or {}
    out: dict[str, dict] = {}
    for name in BODY_KEYPOINT_NAMES:
        src = base.get(name) or {}
        off = offsets.get(name) or {"dx": 0.0, "dy": 0.0}
        out[name] = {
            "x": _clamp01(float(src.get("x", 0.5)) + float(off.get("dx", 0.0))),
            "y": _clamp01(float(src.get("y", 0.5)) + float(off.get("dy", 0.0))),
            "visible": bool(src.get("visible", True)),
            "optional": bool(src.get("optional", False)),
        }
    return out


def _point_dist(a: dict, b: dict) -> float:
    dx = float(a.get("x", 0.0)) - float(b.get("x", 0.0))
    dy = float(a.get("y", 0.0)) - float(b.get("y", 0.0))
    return (dx * dx + dy * dy) ** 0.5


def _evaluate_keypoints_against_target(pred: dict[str, dict], target: dict[str, dict]) -> dict:
    by_keypoint: dict[str, float] = {}
    values: list[float] = []
    for name in BODY_KEYPOINT_NAMES:
        t = target.get(name)
        p = pred.get(name)
        if not t or not p:
            continue
        d = _point_dist(p, t)
        by_keypoint[name] = d
        values.append(d)

    per_part: dict[str, float] = {}
    for part in BODY_PARTS_SEGMENTS:
        a = by_keypoint.get(part["kp_a"])
        b = by_keypoint.get(part["kp_b"])
        if a is None or b is None:
            continue
        per_part[part["id"]] = (a + b) / 2.0

    mean_l2 = (sum(values) / len(values)) if values else None
    pck05 = None
    if values:
        pck05 = sum(1 for v in values if v <= 0.05) / len(values)

    worst_parts = sorted(per_part.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "keypoints_evaluated": len(values),
        "mean_l2": mean_l2,
        "pck_0_05": pck05,
        "per_keypoint_l2": by_keypoint,
        "per_part_l2": per_part,
        "worst_parts": [{"part": name, "l2": dist} for name, dist in worst_parts],
    }


def _objective_samples_from_manifests(include_only_corrected: bool = True) -> list[dict]:
    samples: list[dict] = []
    for sample in _list_human_shape_samples():
        if include_only_corrected and not sample.get("is_corrected"):
            continue
        kp = sample.get("keypoints")
        if isinstance(kp, dict) and kp:
            samples.append({
                "sample_id": sample.get("sample_id"),
                "image_name": sample.get("image_name"),
                "keypoints": kp,
            })
    return samples


def _load_objective_samples(use_snapshot: bool = True) -> tuple[list[dict], str]:
    snapshot_path = _human_shape_objective_snapshot_path()
    if use_snapshot and snapshot_path.exists():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            samples = snapshot.get("samples") or []
            if samples:
                return samples, "snapshot"
        except Exception:
            pass
    return _objective_samples_from_manifests(include_only_corrected=False), "live_manifests"


def _iterative_refiner_train(epochs: int, learning_rate: float, reset_model: bool, use_objective_snapshot: bool) -> dict:
    objective_samples, objective_source = _load_objective_samples(use_snapshot=use_objective_snapshot)
    if not objective_samples:
        raise ValueError("No hay keypoints objetivo guardados para entrenar")

    model = _load_keypoint_refiner_model()
    if reset_model:
        model["offsets"] = _keypoint_template()
        model["iterations"] = 0

    epoch_reports: list[dict] = []

    for epoch_idx in range(1, epochs + 1):
        accum = {name: {"dx": 0.0, "dy": 0.0, "n": 0} for name in BODY_KEYPOINT_NAMES}
        eval_before: list[float] = []

        for item in objective_samples:
            sample_id = str(item.get("sample_id"))
            target = item.get("keypoints") or {}
            try:
                manifest = _load_human_shape_sample_manifest(sample_id)
            except Exception:
                continue
            base, _base_source = _estimate_base_keypoints_for_manifest(manifest)
            pred = _apply_refiner(base, model)
            metrics = _evaluate_keypoints_against_target(pred, target)
            if metrics.get("mean_l2") is not None:
                eval_before.append(float(metrics["mean_l2"]))

            for name in BODY_KEYPOINT_NAMES:
                t = target.get(name)
                p = pred.get(name)
                if not t or not p:
                    continue
                accum[name]["dx"] += float(t.get("x", p.get("x", 0.5))) - float(p.get("x", 0.5))
                accum[name]["dy"] += float(t.get("y", p.get("y", 0.5))) - float(p.get("y", 0.5))
                accum[name]["n"] += 1

        for name in BODY_KEYPOINT_NAMES:
            n = accum[name]["n"]
            if n <= 0:
                continue
            mean_dx = accum[name]["dx"] / n
            mean_dy = accum[name]["dy"] / n
            off = model["offsets"].setdefault(name, {"dx": 0.0, "dy": 0.0})
            off["dx"] = float(off.get("dx", 0.0)) + learning_rate * mean_dx
            off["dy"] = float(off.get("dy", 0.0)) + learning_rate * mean_dy

        eval_after: list[float] = []
        for item in objective_samples:
            sample_id = str(item.get("sample_id"))
            target = item.get("keypoints") or {}
            try:
                manifest = _load_human_shape_sample_manifest(sample_id)
            except Exception:
                continue
            base, _base_source = _estimate_base_keypoints_for_manifest(manifest)
            pred = _apply_refiner(base, model)
            metrics = _evaluate_keypoints_against_target(pred, target)
            if metrics.get("mean_l2") is not None:
                eval_after.append(float(metrics["mean_l2"]))

        before_mean = (sum(eval_before) / len(eval_before)) if eval_before else None
        after_mean = (sum(eval_after) / len(eval_after)) if eval_after else None
        epoch_reports.append(
            {
                "epoch": epoch_idx,
                "samples": len(eval_after),
                "mean_l2_before": before_mean,
                "mean_l2_after": after_mean,
                "improvement": (before_mean - after_mean) if before_mean is not None and after_mean is not None else None,
            }
        )

    model["iterations"] = int(model.get("iterations", 0)) + epochs
    if epoch_reports:
        last = epoch_reports[-1]
        model["metrics"] = {
            "mean_l2": last.get("mean_l2_after"),
            "samples": last.get("samples"),
            "objective_source": objective_source,
            "epochs": epochs,
            "learning_rate": learning_rate,
        }

    _save_keypoint_refiner_model(model)
    return {
        "status": "ok",
        "objective_source": objective_source,
        "objective_samples": len(objective_samples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "epoch_reports": epoch_reports,
        "model": model,
    }


@app.post("/looks/save")
def save_look(request: SaveLookRequest) -> dict:
    import uuid as _uuid
    look_id = _uuid.uuid4().hex[:12]
    look_dir = LOOKS_DIR / look_id
    look_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "look_id": look_id,
        "name": request.name,
        "model_path": request.model_path,
        "texture_path": request.texture_path,
        "garment_type": request.garment_type,
        "project_id": request.project_id,
        "notes": request.notes,
        "created_at": int(time.time()),
    }
    (_look_manifest_path(look_id)).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"status": "saved", "look_id": look_id, "look": manifest}


@app.get("/looks")
def list_looks() -> dict:
    looks = []
    if LOOKS_DIR.exists():
        for manifest_path in sorted(LOOKS_DIR.glob("*/manifest.json")):
            try:
                looks.append(json.loads(manifest_path.read_text()))
            except Exception:
                pass
    looks.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"total": len(looks), "looks": looks}


@app.delete("/looks/{look_id}")
def delete_look(look_id: str) -> dict:
    import shutil
    look_dir = LOOKS_DIR / look_id
    if not look_dir.exists():
        raise HTTPException(status_code=404, detail="Look no encontrado")
    shutil.rmtree(look_dir)
    return {"status": "deleted", "look_id": look_id}


# ---------------------------------------------------------------------------
# Fotos personas — listar fotos disponibles para prueba virtual
# ---------------------------------------------------------------------------

FOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


@app.get("/fotos-personas")
def list_fotos_personas() -> dict:
    fotos = []
    if FOTOS_PERSONAS_DIR.exists():
        for f in sorted(FOTOS_PERSONAS_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in FOTO_EXT:
                fotos.append({
                    "nombre": f.name,
                    "url": f"/fotos-personas-static/{f.name}",
                    "size_bytes": f.stat().st_size,
                })
    return {"total": len(fotos), "fotos": fotos}


@app.get("/human-shape-lab/status")
def human_shape_lab_status() -> dict:
    return _human_shape_lab_status()


@app.post("/human-shape-lab/bootstrap")
def human_shape_lab_bootstrap(request: HumanShapeBootstrapRequest) -> dict:
    image_names = request.image_names
    if not image_names:
        image_names = []
        if FOTOS_PERSONAS_DIR.exists():
            image_names = [
                f.name for f in sorted(FOTOS_PERSONAS_DIR.iterdir()) if f.is_file() and f.suffix.lower() in FOTO_EXT
            ]

    imported: list[dict] = []
    for name in image_names:
        try:
            imported.append(
                _human_shape_sample_payload(
                    _bootstrap_human_shape_sample(
                        image_name=name,
                        force_regenerate_auto_masks=request.force_regenerate_auto_masks,
                    )
                )
            )
        except Exception as exc:
            imported.append({"image_name": name, "status": "error", "detail": str(exc)})

    return {
        "status": "ok",
        "imported_total": len(imported),
        "imported": imported,
        "lab": _human_shape_lab_status()["lab"],
    }


@app.post("/human-shape-lab/save-mask")
def human_shape_lab_save_mask(request: HumanShapeSaveMaskRequest) -> dict:
    try:
        manifest = _load_human_shape_sample_manifest(request.sample_id)
        raw = _decode_image_data_url(request.mask_data_url)
        mask_rel = manifest.get("mask_path")
        if not mask_rel:
            raise FileNotFoundError(f"Sample sin mask_path: {request.sample_id}")
        mask_path = _resolve_input_path(mask_rel)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.write_bytes(raw)
        manifest["is_corrected"] = True
        manifest["updated_at"] = int(time.time())
        _save_human_shape_sample_manifest(manifest)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "ok", "sample": _human_shape_sample_payload(manifest)}


@app.post("/human-shape-lab/samples/{sample_id}/refresh-auto-mask")
def human_shape_lab_refresh_auto_mask(sample_id: str) -> dict:
    try:
        manifest = _load_human_shape_sample_manifest(sample_id)
        updated = _bootstrap_human_shape_sample(
            image_name=str(manifest["image_name"]),
            force_regenerate_auto_masks=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "sample": _human_shape_sample_payload(updated)}


@app.post("/human-shape-lab/train")
def human_shape_lab_train(request: HumanShapeTrainRequest) -> dict:
    try:
        from fase2_2.scripts.prepare_segmentation_real_dataset import build_real_segmentation_manifest
        from fase2_2.scripts.train_segmentation_lite import train_segmentation_lite

        dirs = _human_shape_lab_dirs()
        manifest = build_real_segmentation_manifest(
            source_images_dir=str(dirs["images"]),
            source_masks_dir=str(dirs["masks"]),
            output_dir=str(dirs["dataset"]),
            max_samples=request.max_samples,
            min_iou=request.min_iou,
        )
        result = train_segmentation_lite(
            train_jsonl=manifest["manifests"]["train"],
            val_jsonl=manifest["manifests"]["val"],
            output_config_path=str(_human_shape_current_model_config_path()),
            candidate_thresholds=request.candidate_thresholds,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "dataset": manifest["dataset"],
        "manifests": manifest["manifests"],
        "training": result,
        "lab": _human_shape_lab_status()["lab"],
    }


@app.get("/human-shape-lab/parts")
def human_shape_lab_parts() -> dict:
    """Return the static body part/segment definitions used by the UI."""
    return {
        "keypoint_names": BODY_KEYPOINT_NAMES,
        "parts": BODY_PARTS_SEGMENTS,
    }


@app.get("/human-shape-lab/samples/{sample_id}/keypoints")
def human_shape_lab_get_keypoints(sample_id: str, force_auto: bool = False) -> dict:
    """Return current saved keypoints for a sample, or geometric estimate if none."""
    try:
        manifest = _load_human_shape_sample_manifest(sample_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

    if not force_auto and "keypoints" in manifest and manifest["keypoints"]:
        return {
            "status": "ok",
            "sample_id": sample_id,
            "source": manifest.get("keypoints_source", "saved"),
            "keypoints": manifest["keypoints"],
            "parts": BODY_PARTS_SEGMENTS,
        }

    # Auto-estimate: try MediaPipe first, fall back to geometry
    image_rel = manifest.get("image_path")
    auto_source = "geometric"
    kp = None
    if image_rel:
        image_path = _resolve_input_path(image_rel)
        if image_path.exists():
            kp = _mediapipe_keypoints(image_path)
            if kp:
                auto_source = "mediapipe_pose"

    if kp is None:
        kp = _geometric_keypoints()

    return {
        "status": "ok",
        "sample_id": sample_id,
        "source": auto_source,
        "keypoints": kp,
        "parts": BODY_PARTS_SEGMENTS,
    }


@app.post("/human-shape-lab/samples/{sample_id}/keypoints")
def human_shape_lab_save_keypoints(sample_id: str, request: HumanShapeSaveKeypointsRequest) -> dict:
    """Save corrected keypoints for a sample."""
    try:
        manifest = _load_human_shape_sample_manifest(sample_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

    manifest["keypoints"] = request.keypoints
    manifest["keypoints_source"] = "corrected"
    if request.accepted_parts is not None:
        manifest["accepted_parts"] = request.accepted_parts
    if request.rejected_parts is not None:
        manifest["rejected_parts"] = request.rejected_parts
    manifest["has_keypoints"] = True
    manifest["is_corrected"] = True
    manifest["updated_at"] = int(time.time())
    _save_human_shape_sample_manifest(manifest)

    return {
        "status": "ok",
        "sample_id": sample_id,
        "keypoints_saved": len(request.keypoints),
        "sample": _human_shape_sample_payload(manifest),
    }


@app.post("/human-shape-lab/objective/snapshot")
def human_shape_lab_objective_snapshot(request: HumanShapeObjectiveSnapshotRequest) -> dict:
    samples = _objective_samples_from_manifests(include_only_corrected=request.include_only_corrected)
    if not samples:
        raise HTTPException(status_code=400, detail="No hay muestras con keypoints para snapshot objetivo")

    payload = {
        "created_at": int(time.time()),
        "include_only_corrected": request.include_only_corrected,
        "samples_total": len(samples),
        "samples": samples,
    }
    path = _human_shape_objective_snapshot_path()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "ok",
        "objective_snapshot_path": _to_repo_relative(path),
        "samples_total": len(samples),
    }


@app.get("/human-shape-lab/samples/{sample_id}/proposal")
def human_shape_lab_sample_proposal(sample_id: str, use_objective_snapshot: bool = True) -> dict:
    try:
        manifest = _load_human_shape_sample_manifest(sample_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

    base, base_source = _estimate_base_keypoints_for_manifest(manifest)
    refiner = _load_keypoint_refiner_model()
    proposal = _apply_refiner(base, refiner)

    objectives, objective_source = _load_objective_samples(use_snapshot=use_objective_snapshot)
    target_map = {str(item.get("sample_id")): item.get("keypoints") or {} for item in objectives}
    target = target_map.get(sample_id)
    metrics = _evaluate_keypoints_against_target(proposal, target) if target else None

    return {
        "status": "ok",
        "sample_id": sample_id,
        "base_source": base_source,
        "proposal_source": "base_plus_refiner",
        "proposal_keypoints": proposal,
        "target_source": objective_source if target else None,
        "target_keypoints": target,
        "metrics": metrics,
        "parts": BODY_PARTS_SEGMENTS,
    }


@app.post("/human-shape-lab/keypoints/iterate")
def human_shape_lab_keypoints_iterate(request: HumanShapeIterativeTrainRequest) -> dict:
    try:
        result = _iterative_refiner_train(
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            reset_model=request.reset_model,
            use_objective_snapshot=request.use_objective_snapshot,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **result,
        "lab": _human_shape_lab_status()["lab"],
    }


# ---------------------------------------------------------------------------
# Prueba virtual — aplicar look sobre foto de persona
# ---------------------------------------------------------------------------

@app.post("/tryon/apply")
def tryon_apply(request: TryOnApplyRequest) -> dict:
    """Superpone el look guardado sobre una foto de fotos_personas/ usando pipeline_v2."""
    manifest_path = _look_manifest_path(request.look_id)
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Look no encontrado")
    manifest = json.loads(manifest_path.read_text())

    foto_path = FOTOS_PERSONAS_DIR / request.foto_nombre
    if not foto_path.exists() or foto_path.suffix.lower() not in FOTO_EXT:
        raise HTTPException(status_code=404, detail="Foto no encontrada")

    garment_path = manifest.get("texture_path") or manifest.get("model_path")
    if not garment_path:
        raise HTTPException(status_code=400, detail="El look no tiene textura ni modelo asignado")

    garment_full = BASE_DIR / garment_path if not Path(garment_path).is_absolute() else Path(garment_path)
    if not garment_full.exists():
        raise HTTPException(status_code=400, detail=f"Archivo de prenda no encontrado: {garment_path}")

    output_dir = DATA_DIR / "tryon_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"tryon_{request.look_id}_{foto_path.stem}_{int(time.time())}.png"
    output_path = str(output_dir / output_name)

    try:
        from fase2_2.core.tryon.pipeline_v2 import run_tryon_v2
        from fase2_1.core.tryon.schemas import TryOnRequest as _TryOnRequest
        tryon_req = _TryOnRequest(
            image_path=str(foto_path),
            garment_path=str(garment_full),
            output_path=output_path,
        )
        result = run_tryon_v2(tryon_req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en pipeline: {exc}") from exc

    # run_tryon_v2 returns a Pydantic TryOnResult; support dicts defensively.
    if isinstance(result, dict):
        result_output = result.get("output_path", output_path)
        result_status = result.get("status", "ok")
    else:
        result_output = getattr(result, "output_path", output_path)
        result_status = getattr(result, "status", "ok")

    url = None
    try:
        rel = Path(result_output).relative_to(DATA_DIR)
        url = f"/artifacts/{rel}"
    except ValueError:
        pass

    return {
        "status": result_status,
        "look_id": request.look_id,
        "look_name": manifest.get("name"),
        "foto": request.foto_nombre,
        "output_path": result_output,
        "url": url,
    }


# ---------------------------------------------------------------------------
# Rutas HTML para navegador
# ---------------------------------------------------------------------------

@app.get("/tryon")
def tryon_page() -> FileResponse:
    return FileResponse(str(UI_DIR / "tryon.html"))


@app.get("/human-shape-lab")
def human_shape_lab_page() -> FileResponse:
    return FileResponse(str(UI_DIR / "human_shape_lab_v2.html"))
