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
import urllib.request
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
    update_only_erroneous_sections: bool = True
    section_error_threshold: float = Field(default=0.03, ge=0.0, le=1.0)
    enable_golden_ratio_prior: bool = True
    golden_ratio_alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    enable_arm_pose_prior: bool = True
    arm_pose_alpha: float = Field(default=0.40, ge=0.0, le=1.0)
    enable_leg_pose_prior: bool = True
    leg_pose_alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    enable_arm_pose_curriculum: bool = True
    curriculum_start_fraction: float = Field(default=0.35, ge=0.1, le=1.0)


class HumanShapeSaveModelRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    include_objective_snapshot: bool = True


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


def _human_shape_pose_landmarker_model_path() -> Path:
    return _human_shape_lab_dirs()["models"] / "pose_landmarker_lite.task"


def _ensure_pose_landmarker_model() -> Path | None:
    """Ensure pose landmarker task model exists (download once if missing)."""
    model_path = _human_shape_pose_landmarker_model_path()
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        if not data:
            return None
        model_path.write_bytes(data)
        return model_path
    except Exception:
        return None


def _mediapipe_landmarker_keypoints(image_path: Path) -> tuple[dict[str, dict] | None, dict]:
    """Try MediaPipe Pose Landmarker (Tasks API) as an isolated second backend."""
    try:
        import mediapipe as mp  # type: ignore

        model_path = _ensure_pose_landmarker_model()
        if model_path is None or not model_path.exists():
            return None, {"strategy": "mediapipe_pose_landmarker", "reason": "model_unavailable"}

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )

        mp_image = mp.Image.create_from_file(str(image_path))
        with PoseLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)

        if not result or not getattr(result, "pose_landmarks", None):
            return None, {"strategy": "mediapipe_pose_landmarker", "reason": "no_pose"}

        poses = result.pose_landmarks
        if not poses:
            return None, {"strategy": "mediapipe_pose_landmarker", "reason": "empty_pose_list"}
        lm = poses[0]

        def pt(idx: int, vis_thresh: float = 0.4) -> dict:
            p = lm[idx]
            vis = float(getattr(p, "visibility", 1.0))
            pres = float(getattr(p, "presence", 1.0))
            return {
                "x": float(p.x),
                "y": float(p.y),
                "visible": (vis >= vis_thresh) and (pres >= 0.3),
                "optional": False,
            }

        # BlazePose landmark indices.
        NOSE = 0
        LEFT_MOUTH = 9
        RIGHT_MOUTH = 10
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_INDEX = 19
        RIGHT_INDEX = 20
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32

        kp = {
            "crown": {
                "x": float(lm[NOSE].x),
                "y": max(0.0, float(lm[NOSE].y) - 0.08),
                "visible": True,
                "optional": False,
            },
            "chin": {
                "x": float((lm[LEFT_MOUTH].x + lm[RIGHT_MOUTH].x) / 2),
                "y": float(max(lm[LEFT_MOUTH].y, lm[RIGHT_MOUTH].y)),
                "visible": True,
                "optional": False,
            },
            "neck_base": {
                "x": float((lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2),
                "y": float((lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2),
                "visible": True,
                "optional": False,
            },
            "left_shoulder": pt(LEFT_SHOULDER),
            "right_shoulder": pt(RIGHT_SHOULDER),
            "left_elbow": pt(LEFT_ELBOW),
            "right_elbow": pt(RIGHT_ELBOW),
            "left_wrist": pt(LEFT_WRIST),
            "right_wrist": pt(RIGHT_WRIST),
            "left_hand_tip": {**pt(LEFT_INDEX), "optional": True},
            "right_hand_tip": {**pt(RIGHT_INDEX), "optional": True},
            "hip_center": {
                "x": float((lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2),
                "y": float((lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2),
                "visible": True,
                "optional": False,
            },
            "left_hip": pt(LEFT_HIP),
            "right_hip": pt(RIGHT_HIP),
            "left_knee": pt(LEFT_KNEE),
            "right_knee": pt(RIGHT_KNEE),
            "left_ankle": pt(LEFT_ANKLE),
            "right_ankle": pt(RIGHT_ANKLE),
            "left_toe": {**pt(LEFT_FOOT_INDEX), "optional": True},
            "right_toe": {**pt(RIGHT_FOOT_INDEX), "optional": True},
        }
        return kp, {
            "strategy": "mediapipe_pose_landmarker",
            "model_path": _to_repo_relative(model_path),
        }
    except Exception as exc:
        return None, {"strategy": "mediapipe_pose_landmarker", "reason": f"error:{exc}"}


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


def _human_shape_model_checkpoints_dir() -> Path:
    path = _human_shape_lab_dirs()["models"] / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_human_shape_model_checkpoint(name: str | None, include_objective_snapshot: bool = True) -> dict:
    refiner_path = _human_shape_keypoint_refiner_path()
    if not refiner_path.exists():
        raise FileNotFoundError("No existe modelo de keypoints para guardar checkpoint")

    ts = int(time.time())
    raw_name = (name or "checkpoint").strip()
    safe_name = _sanitize_name(raw_name) if raw_name else "checkpoint"
    ckpt_dir = _human_shape_model_checkpoints_dir() / f"{safe_name}_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    refiner_dst = ckpt_dir / "keypoint_refiner_v1.json"
    shutil.copy2(refiner_path, refiner_dst)

    objective_src = _human_shape_objective_snapshot_path()
    objective_dst = None
    if include_objective_snapshot and objective_src.exists():
        objective_dst = ckpt_dir / "keypoint_objective_snapshot.json"
        shutil.copy2(objective_src, objective_dst)

    meta = {
        "saved_at": ts,
        "name": raw_name or "checkpoint",
        "checkpoint_dir": _to_repo_relative(ckpt_dir),
        "refiner_path": _to_repo_relative(refiner_dst),
        "objective_snapshot_path": _to_repo_relative(objective_dst) if objective_dst else None,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


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


def _detect_face_bbox(image_path: Path) -> tuple[dict | None, dict]:
    """Detect a frontal face using OpenCV Haar cascade (simple and fast)."""
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(image_path))
        if img is None:
            return None, {"strategy": "opencv_haar", "reason": "image_load_failed"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            return None, {"strategy": "opencv_haar", "reason": "cascade_not_loaded"}

        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(36, 36),
        )
        if faces is None or len(faces) == 0:
            return None, {"strategy": "opencv_haar", "reason": "no_face"}

        x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        ih, iw = gray.shape
        bbox = {
            "x_min": _clamp01(float(x) / max(1.0, float(iw))),
            "x_max": _clamp01(float(x + w) / max(1.0, float(iw))),
            "y_min": _clamp01(float(y) / max(1.0, float(ih))),
            "y_max": _clamp01(float(y + h) / max(1.0, float(ih))),
            "cx": _clamp01(float(x + w * 0.5) / max(1.0, float(iw))),
            "cy": _clamp01(float(y + h * 0.5) / max(1.0, float(ih))),
            "w": _clamp01(float(w) / max(1.0, float(iw))),
            "h": _clamp01(float(h) / max(1.0, float(ih))),
        }
        return bbox, {
            "strategy": "opencv_haar",
            "image_path": _to_repo_relative(image_path),
            "cascade": "haarcascade_frontalface_default.xml",
            "faces_found": int(len(faces)),
        }
    except Exception as exc:
        return None, {"strategy": "opencv_haar", "reason": f"error:{exc}"}


def _face_anchored_geometric(face_bbox: dict | None) -> tuple[dict[str, dict], dict]:
    """Shift geometric template using face as anchor when pose/silhouette are weak."""
    base = _geometric_keypoints()
    if not face_bbox:
        return base, {"strategy": "geometric"}

    cx = float(face_bbox.get("cx", 0.5))
    y_top = float(face_bbox.get("y_min", 0.05))
    f_w = max(0.08, float(face_bbox.get("w", 0.15)))
    f_h = max(0.08, float(face_bbox.get("h", 0.18)))

    x_scale = min(0.55, max(0.25, f_w * 2.3))
    y_scale = min(0.95, max(0.60, f_h * 6.8))

    anchored: dict[str, dict] = {}
    for name, kp in base.items():
        x = cx + (float(kp.get("x", 0.5)) - 0.5) * x_scale
        y = y_top + (float(kp.get("y", 0.04)) - 0.04) * y_scale
        anchored[name] = {
            "x": _clamp01(x),
            "y": _clamp01(y),
            "visible": bool(kp.get("visible", True)),
            "optional": bool(kp.get("optional", False)),
        }

    return anchored, {
        "strategy": "face_anchored_geometric",
        "face_bbox": face_bbox,
        "x_scale": x_scale,
        "y_scale": y_scale,
    }


def _silhouette_keypoints_from_manifest(manifest: dict, face_hint: dict | None = None) -> tuple[dict[str, dict] | None, dict]:
    """Estimate keypoints from the sample mask silhouette when pose landmarks are unavailable."""
    mask_rel = manifest.get("mask_path") or manifest.get("auto_mask_path")
    if not mask_rel:
        return None, {"strategy": "silhouette_mask", "reason": "missing_mask_path"}

    mask_path = _resolve_input_path(mask_rel)
    if not mask_path.exists():
        return None, {"strategy": "silhouette_mask", "reason": "mask_not_found"}

    try:
        from PIL import Image as PILImage

        img = PILImage.open(mask_path).convert("L")
        w, h = img.size
        pix = img.load()

        row_min: list[int | None] = [None] * h
        row_max: list[int | None] = [None] * h
        x_min, x_max = w, -1
        y_min, y_max = h, -1
        fg_pixels = 0

        for y in range(h):
            left = None
            right = None
            for x in range(w):
                if int(pix[x, y]) > 12:
                    fg_pixels += 1
                    if left is None:
                        left = x
                    right = x
            if left is not None and right is not None:
                row_min[y] = left
                row_max[y] = right
                x_min = min(x_min, left)
                x_max = max(x_max, right)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

        if fg_pixels < 100 or x_max < x_min or y_max < y_min:
            return None, {"strategy": "silhouette_mask", "reason": "weak_foreground"}

        body_h = max(1, y_max - y_min)
        body_w = max(1, x_max - x_min)

        def _span_at(rel_y: float) -> tuple[int, int, int]:
            y0 = int(y_min + rel_y * body_h)
            y0 = max(0, min(h - 1, y0))
            max_search = max(6, int(0.15 * h))
            for d in range(0, max_search + 1):
                for cand in (y0 - d, y0 + d):
                    if cand < 0 or cand >= h:
                        continue
                    l = row_min[cand]
                    r = row_max[cand]
                    if l is not None and r is not None:
                        return l, r, cand
            return x_min, x_max, y0

        def _mk(rel_y: float, frac_x: float, optional: bool = False) -> dict:
            l, r, yy = _span_at(rel_y)
            span = max(1, r - l)
            xx = l + frac_x * span
            return {
                "x": _clamp01(xx / max(1.0, float(w))),
                "y": _clamp01(yy / max(1.0, float(h))),
                "visible": True,
                "optional": optional,
            }

        crown_y = _clamp01(y_min / max(1.0, float(h)))
        chin_rel = 0.13
        neck_rel = 0.20
        shoulder_rel = 0.24
        if face_hint:
            fx_min = float(face_hint.get("x_min", 0.0))
            fx_max = float(face_hint.get("x_max", 1.0))
            fy_min = float(face_hint.get("y_min", crown_y))
            fy_max = float(face_hint.get("y_max", crown_y + 0.10))
            crown_y = _clamp01(fy_min)
            chin_abs = _clamp01(fy_max)
            y0_norm = _clamp01(y_min / max(1.0, float(h)))
            body_h_norm = max(1e-6, float(body_h) / max(1.0, float(h)))
            chin_rel = _clamp01((chin_abs - y0_norm) / body_h_norm)
            neck_rel = _clamp01(chin_rel + 0.07)
            shoulder_rel = _clamp01(chin_rel + 0.11)

        kp = {
            "crown": {"x": _clamp01(((x_min + x_max) * 0.5) / max(1.0, float(w))), "y": crown_y, "visible": True, "optional": False},
            "chin": _mk(chin_rel, 0.50),
            "neck_base": _mk(neck_rel, 0.50),
            "left_shoulder": _mk(shoulder_rel, 0.22),
            "right_shoulder": _mk(shoulder_rel, 0.78),
            "left_elbow": _mk(0.43, 0.16),
            "right_elbow": _mk(0.43, 0.84),
            "left_wrist": _mk(0.58, 0.10),
            "right_wrist": _mk(0.58, 0.90),
            "left_hand_tip": _mk(0.64, 0.08, optional=True),
            "right_hand_tip": _mk(0.64, 0.92, optional=True),
            "hip_center": _mk(0.58, 0.50),
            "left_hip": _mk(0.58, 0.35),
            "right_hip": _mk(0.58, 0.65),
            "left_knee": _mk(0.77, 0.38),
            "right_knee": _mk(0.77, 0.62),
            "left_ankle": _mk(0.91, 0.40),
            "right_ankle": _mk(0.91, 0.60),
            "left_toe": _mk(0.97, 0.36, optional=True),
            "right_toe": _mk(0.97, 0.64, optional=True),
        }

        # If face is horizontally displaced from mask center, shift full skeleton accordingly.
        if face_hint:
            mask_cx = _clamp01(((x_min + x_max) * 0.5) / max(1.0, float(w)))
            face_cx = _clamp01(float(face_hint.get("cx", mask_cx)))
            dx = face_cx - mask_cx
            if abs(dx) > 0.01:
                for name in BODY_KEYPOINT_NAMES:
                    kp[name]["x"] = _clamp01(float(kp[name]["x"]) + dx)

        params = {
            "strategy": "silhouette_mask",
            "mask_path": _to_repo_relative(mask_path),
            "bbox_norm": {
                "x_min": _clamp01(x_min / max(1.0, float(w))),
                "x_max": _clamp01(x_max / max(1.0, float(w))),
                "y_min": _clamp01(y_min / max(1.0, float(h))),
                "y_max": _clamp01(y_max / max(1.0, float(h))),
            },
            "foreground_ratio": fg_pixels / float(max(1, w * h)),
            "face_hint_used": bool(face_hint),
            "face_hint": face_hint,
            "face_shift_dx": (float(face_hint.get("cx", 0.5)) - _clamp01(((x_min + x_max) * 0.5) / max(1.0, float(w)))) if face_hint else 0.0,
        }
        return kp, params
    except Exception as exc:
        return None, {"strategy": "silhouette_mask", "reason": f"error:{exc}"}


def _detect_skin_body_lines(image_path: Path, mask_path: Path | None = None, face_hint: dict | None = None) -> tuple[dict | None, dict]:
    """Detect body side lines from skin-color ranges with adaptive thresholds and confidence gating."""
    try:
        import cv2  # type: ignore
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            return None, {"strategy": "skin_lines", "reason": "image_load_failed"}

        h, w = img.shape[:2]
        if h < 16 or w < 16:
            return None, {"strategy": "skin_lines", "reason": "image_too_small"}

        # Restrict detection to person region when a body mask is available.
        body_mask = np.full((h, w), 255, dtype=np.uint8)
        mask_used = False
        if mask_path and mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                body_mask = np.where(m > 12, 255, 0).astype(np.uint8)
                mask_used = bool(np.count_nonzero(body_mask) > 64)

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Fixed ranges to cover tones + shadows.
        mask_ycrcb_main = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        mask_ycrcb_shadow = cv2.inRange(ycrcb, (0, 125, 70), (255, 180, 138))
        mask_hsv_main = cv2.inRange(hsv, (0, 22, 35), (28, 230, 255))
        mask_hsv_shadow = cv2.inRange(hsv, (0, 10, 18), (35, 255, 210))
        fixed_mask = cv2.bitwise_or(mask_ycrcb_main, mask_hsv_main)
        fixed_mask = cv2.bitwise_or(fixed_mask, cv2.bitwise_and(mask_ycrcb_shadow, mask_hsv_shadow))

        adaptive_used = False
        adaptive_mask = np.zeros((h, w), dtype=np.uint8)

        # Adaptive range from detected face to better match image-specific skin tone.
        if face_hint:
            x0 = int(max(0, min(w - 1, float(face_hint.get("x_min", 0.0)) * w)))
            x1 = int(max(0, min(w, float(face_hint.get("x_max", 1.0)) * w)))
            y0 = int(max(0, min(h - 1, float(face_hint.get("y_min", 0.0)) * h)))
            y1 = int(max(0, min(h, float(face_hint.get("y_max", 1.0)) * h)))
            if x1 > x0 + 6 and y1 > y0 + 6:
                roi_ycrcb = ycrcb[y0:y1, x0:x1]
                roi_hsv = hsv[y0:y1, x0:x1]
                roi_mask = body_mask[y0:y1, x0:x1]
                sel = roi_mask > 0
                if np.count_nonzero(sel) > 32:
                    cr = roi_ycrcb[:, :, 1][sel]
                    cb = roi_ycrcb[:, :, 2][sel]
                    hh = roi_hsv[:, :, 0][sel]
                    ss = roi_hsv[:, :, 1][sel]
                    if cr.size > 24 and cb.size > 24:
                        cr_lo, cr_hi = np.percentile(cr, [8, 92])
                        cb_lo, cb_hi = np.percentile(cb, [8, 92])
                        h_lo, h_hi = np.percentile(hh, [8, 92]) if hh.size > 0 else (0, 30)
                        s_lo, s_hi = np.percentile(ss, [6, 96]) if ss.size > 0 else (20, 240)
                        # Add margin for lighting changes and shadows.
                        cr_lo, cr_hi = max(115, cr_lo - 8), min(185, cr_hi + 8)
                        cb_lo, cb_hi = max(60, cb_lo - 8), min(145, cb_hi + 8)
                        h_lo, h_hi = max(0, h_lo - 4), min(40, h_hi + 4)
                        s_lo, s_hi = max(8, s_lo - 12), min(255, s_hi + 18)
                        mask_y_ad = cv2.inRange(ycrcb, (0, int(cr_lo), int(cb_lo)), (255, int(cr_hi), int(cb_hi)))
                        mask_h_ad = cv2.inRange(hsv, (int(h_lo), int(s_lo), 12), (int(h_hi), int(s_hi), 255))
                        adaptive_mask = cv2.bitwise_and(mask_y_ad, mask_h_ad)
                        adaptive_used = True

        mask = fixed_mask
        if adaptive_used:
            mask = cv2.bitwise_or(mask, adaptive_mask)

        # Restrict to body and clean noise.
        mask = cv2.bitwise_and(mask, body_mask)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_mid)

        # Keep largest connected components only.
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        if n_labels > 1:
            areas = []
            for i in range(1, n_labels):
                areas.append((int(stats[i, cv2.CC_STAT_AREA]), i))
            areas.sort(reverse=True)
            keep = {idx for _a, idx in areas[:4] if _a > max(24, int(0.0005 * h * w))}
            filtered = np.zeros_like(mask)
            for idx in keep:
                filtered[labels == idx] = 255
            mask = filtered

        rows: list[dict] = []
        width_ratios: list[float] = []
        min_row_px = max(3, int(0.008 * w))
        for y in range(0, h):
            xs = np.where(mask[y] > 0)[0]
            if xs.size < min_row_px:
                continue
            # Percentiles are more robust than min/max against outliers and shadows.
            x0 = int(np.percentile(xs, 8))
            x1 = int(np.percentile(xs, 92))
            if x1 <= x0:
                continue
            row_w = x1 - x0
            if row_w < max(4, int(0.02 * w)) or row_w > int(0.78 * w):
                continue
            width_ratio = float(row_w) / max(1.0, float(w))
            width_ratios.append(width_ratio)
            rows.append({
                "y": _clamp01(float(y) / max(1.0, float(h))),
                "x_min": _clamp01(float(x0) / max(1.0, float(w))),
                "x_max": _clamp01(float(x1) / max(1.0, float(w))),
                "width_ratio": width_ratio,
            })

        coverage = float(len(rows)) / max(1.0, float(h))
        median_width_ratio = float(np.median(width_ratios)) if width_ratios else 0.0
        confidence = min(1.0, coverage * 3.2 + (0.22 if adaptive_used else 0.0) + (0.12 if mask_used else 0.0))

        # If skin seems too spread across body, likely clothing/background contamination.
        if coverage > 0.45:
            confidence *= 0.25
        if median_width_ratio > 0.32:
            confidence *= 0.45

        if len(rows) < max(8, int(0.05 * h)) or confidence < 0.28:
            return None, {
                "strategy": "skin_lines",
                "reason": "low_confidence",
                "rows": len(rows),
                "coverage": coverage,
                "median_width_ratio": median_width_ratio,
                "confidence": confidence,
                "adaptive_used": adaptive_used,
                "mask_used": mask_used,
            }

        return {
            "rows": rows,
            "coverage": coverage,
            "median_width_ratio": median_width_ratio,
            "confidence": confidence,
            "adaptive_used": adaptive_used,
            "mask_used": mask_used,
        }, {
            "strategy": "skin_lines",
            "image_path": _to_repo_relative(image_path),
            "rows_detected": len(rows),
            "coverage": coverage,
            "median_width_ratio": median_width_ratio,
            "confidence": confidence,
            "adaptive_used": adaptive_used,
            "mask_used": mask_used,
        }
    except Exception as exc:
        return None, {"strategy": "skin_lines", "reason": f"error:{exc}"}


def _apply_skin_guided_body_lines(kp: dict[str, dict], skin_lines: dict | None, alpha: float = 0.22) -> tuple[dict[str, dict], dict]:
    if not skin_lines or not isinstance(skin_lines.get("rows"), list):
        return _normalize_keypoint_map(kp, force_visible=True), {"applied": False, "reason": "no_skin_lines"}

    out = _normalize_keypoint_map(kp, force_visible=True)
    rows = skin_lines.get("rows") or []
    confidence = float(skin_lines.get("confidence", 0.0))
    if not rows:
        return out, {"applied": False, "reason": "empty_rows"}
    if confidence < 0.22:
        return out, {"applied": False, "reason": "low_confidence", "confidence": confidence}

    def span_at(y_norm: float) -> tuple[float, float, float] | None:
        best = None
        best_d = None
        for r in rows:
            yy = float(r.get("y", 0.0))
            d = abs(yy - y_norm)
            if best_d is None or d < best_d:
                best_d = d
                best = r
        if best is None or best_d is None:
            return None
        x_min = float(best.get("x_min", 0.0))
        x_max = float(best.get("x_max", 1.0))
        if x_max <= x_min:
            return None
        return x_min, x_max, float(best_d)

    side_pairs = [
        ("left_shoulder", "right_shoulder", 0.14),
        ("left_elbow", "right_elbow", 0.11),
        ("left_wrist", "right_wrist", 0.09),
        ("left_hip", "right_hip", 0.16),
        ("left_knee", "right_knee", 0.20),
        ("left_ankle", "right_ankle", 0.23),
        ("left_toe", "right_toe", 0.25),
    ]

    # Strong guidance only when confidence is high; weak otherwise.
    alpha_eff = float(alpha) * max(0.35, min(1.2, confidence))

    touched = 0
    for left_name, right_name, side_margin in side_pairs:
        y_ref = (float(out[left_name]["y"]) + float(out[right_name]["y"])) / 2.0
        span = span_at(y_ref)
        if not span:
            continue
        x_min, x_max, y_delta = span
        if y_delta > 0.06:
            continue
        width = max(1e-6, x_max - x_min)
        left_target = _clamp01(x_min + width * side_margin)
        right_target = _clamp01(x_max - width * side_margin)
        out[left_name]["x"] = _clamp01((1.0 - alpha_eff) * float(out[left_name]["x"]) + alpha_eff * left_target)
        out[right_name]["x"] = _clamp01((1.0 - alpha_eff) * float(out[right_name]["x"]) + alpha_eff * right_target)
        touched += 2

    return out, {
        "applied": touched > 0,
        "alpha": alpha_eff,
        "touched_keypoints": touched,
        "rows": len(rows),
        "coverage": float(skin_lines.get("coverage", 0.0)),
        "confidence": confidence,
        "adaptive_used": bool(skin_lines.get("adaptive_used", False)),
        "mask_used": bool(skin_lines.get("mask_used", False)),
    }


def _detect_skin_limb_profiles(
    image_path: "Path",
    mask_path: "Path | None",
    face_hint: "dict | None",
    kp_hint: "dict[str, dict] | None",
) -> dict:
    """Detect per-limb skin presence, continuity, and flex hints.
    Returns {limb_name: {has_skin, rows, coverage, continuity, suggest_flex}} for
    left/right arm, forearm, leg, calf.
    """
    try:
        import cv2  # type: ignore
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            return {}
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return {}

        # Build body mask
        body_mask = np.full((h, w), 255, dtype=np.uint8)
        if mask_path and mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                body_mask = np.where(m > 12, 255, 0).astype(np.uint8)

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_y = cv2.inRange(ycrcb, (0, 125, 70), (255, 180, 138))
        mask_h = cv2.inRange(hsv, (0, 10, 18), (35, 255, 210))
        skin_mask = cv2.bitwise_or(mask_y, mask_h)

        # Adaptive calibration from face ROI
        if face_hint:
            x0 = int(max(0, float(face_hint.get("x_min", 0.0)) * w))
            x1 = int(min(w, float(face_hint.get("x_max", 1.0)) * w))
            y0 = int(max(0, float(face_hint.get("y_min", 0.0)) * h))
            y1 = int(min(h, float(face_hint.get("y_max", 1.0)) * h))
            if x1 > x0 + 6 and y1 > y0 + 6:
                roi_ycrcb = ycrcb[y0:y1, x0:x1]
                roi_hsv = hsv[y0:y1, x0:x1]
                sel = body_mask[y0:y1, x0:x1] > 0
                if np.count_nonzero(sel) > 32:
                    cr = roi_ycrcb[:, :, 1][sel]
                    cb = roi_ycrcb[:, :, 2][sel]
                    hh = roi_hsv[:, :, 0][sel]
                    ss = roi_hsv[:, :, 1][sel]
                    if cr.size > 24:
                        cr_lo = max(115, float(np.percentile(cr, 8)) - 8)
                        cr_hi = min(185, float(np.percentile(cr, 92)) + 8)
                        cb_lo = max(60, float(np.percentile(cb, 8)) - 8)
                        cb_hi = min(145, float(np.percentile(cb, 92)) + 8)
                        h_lo = max(0, float(np.percentile(hh, 8)) - 4)
                        h_hi = min(40, float(np.percentile(hh, 92)) + 4)
                        s_lo = max(8, float(np.percentile(ss, 6)) - 12)
                        s_hi = min(255, float(np.percentile(ss, 96)) + 18)
                        ad_y = cv2.inRange(ycrcb, (0, int(cr_lo), int(cb_lo)), (255, int(cr_hi), int(cb_hi)))
                        ad_h = cv2.inRange(hsv, (int(h_lo), int(s_lo), 12), (int(h_hi), int(s_hi), 255))
                        skin_mask = cv2.bitwise_or(skin_mask, cv2.bitwise_and(ad_y, ad_h))

        skin_mask = cv2.bitwise_and(skin_mask, body_mask)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k3)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k5)

        kp = _normalize_keypoint_map(kp_hint or {}, force_visible=True)

        def _ypx(name: str) -> int:
            return int(_clamp01(float(kp[name]["y"])) * h)

        def _xpx(name: str) -> int:
            return int(_clamp01(float(kp[name]["x"])) * w)

        margin = max(8, int(0.07 * w))

        def _profile(y_top_px: int, y_bot_px: int, x_lo_px: int, x_hi_px: int, name: str) -> dict:
            yt = max(0, min(h - 1, y_top_px))
            yb = max(0, min(h, y_bot_px))
            xl = max(0, min(w - 1, x_lo_px))
            xr = max(0, min(w, x_hi_px))
            if yb <= yt + 2 or xr <= xl + 2:
                return {"has_skin": False, "rows": [], "coverage": 0.0, "continuity": 0.0, "suggest_flex": False}

            reg = skin_mask[yt:yb, xl:xr]
            min_skin_px = max(2, int(0.05 * (xr - xl)))
            rows_with: list[dict] = []
            runs: list[int] = []
            in_skin = False
            run_len = 0

            for ry in range(reg.shape[0]):
                row_skin = np.where(reg[ry] > 0)[0]
                has = row_skin.size >= min_skin_px
                if has:
                    cx = int(np.mean(row_skin)) + xl
                    rows_with.append({
                        "y": _clamp01(float(ry + yt) / h),
                        "x_center": _clamp01(float(cx) / w),
                        "x_min": _clamp01(float(int(row_skin.min()) + xl) / w),
                        "x_max": _clamp01(float(int(row_skin.max()) + xl) / w),
                    })
                    run_len += 1
                    in_skin = True
                else:
                    if in_skin:
                        runs.append(run_len)
                        run_len = 0
                        in_skin = False
            if in_skin:
                runs.append(run_len)

            total = yb - yt
            coverage = float(len(rows_with)) / max(1, total)
            continuity = min(1.0, coverage * 1.6)
            # Discontinuity: multiple runs with total coverage < 0.55 suggests flex/gap
            suggest_flex = len(runs) >= 2 and coverage < 0.58 and max(runs) < 0.85 * total
            return {
                "has_skin": coverage > 0.12,
                "rows": rows_with,
                "coverage": coverage,
                "continuity": continuity,
                "suggest_flex": suggest_flex,
            }

        profiles: dict[str, dict] = {}
        for side in ("left", "right"):
            sy, sx = _ypx(f"{side}_shoulder"), _xpx(f"{side}_shoulder")
            ey, ex = _ypx(f"{side}_elbow"), _xpx(f"{side}_elbow")
            wy, wx = _ypx(f"{side}_wrist"), _xpx(f"{side}_wrist")
            hy, hx = _ypx(f"{side}_hip"), _xpx(f"{side}_hip")
            ky, kx = _ypx(f"{side}_knee"), _xpx(f"{side}_knee")
            ay, ax = _ypx(f"{side}_ankle"), _xpx(f"{side}_ankle")

            profiles[f"{side}_arm"] = _profile(
                min(sy, ey), max(sy, ey) + 1,
                min(sx, ex) - margin, max(sx, ex) + margin,
                f"{side}_arm"
            )
            profiles[f"{side}_forearm"] = _profile(
                min(ey, wy), max(ey, wy) + 1,
                min(ex, wx) - margin, max(ex, wx) + margin,
                f"{side}_forearm"
            )
            profiles[f"{side}_leg"] = _profile(
                min(hy, ky), max(hy, ky) + 1,
                min(hx, kx) - margin, max(hx, kx) + margin,
                f"{side}_leg"
            )
            profiles[f"{side}_calf"] = _profile(
                min(ky, ay), max(ky, ay) + 1,
                min(kx, ax) - margin, max(kx, ax) + margin,
                f"{side}_calf"
            )
        return profiles
    except Exception:
        return {}


def _detect_waist_color_boundary(
    image_path: "Path",
    mask_path: "Path | None",
    kp_hint: "dict[str, dict] | None",
) -> dict:
    """Detect the waist position via a horizontal color-change boundary between top and bottom garments."""
    try:
        import cv2  # type: ignore
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            return {"found": False, "reason": "image_load_failed"}
        h, w = img.shape[:2]

        body_mask = np.full((h, w), 255, dtype=np.uint8)
        if mask_path and mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                body_mask = np.where(m > 12, 255, 0).astype(np.uint8)

        kp = _normalize_keypoint_map(kp_hint or {}, force_visible=True)
        neck_y = float(kp["neck_base"]["y"])
        hip_y = float(kp["hip_center"]["y"])
        torso_h = max(0.10, hip_y - neck_y)

        # Search band: 30-90% of the torso span
        sy_top = int((neck_y + 0.30 * torso_h) * h)
        sy_bot = int((neck_y + 0.90 * torso_h) * h)
        sy_top = max(0, min(h - 4, sy_top))
        sy_bot = max(sy_top + 4, min(h - 1, sy_bot))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Smooth horizontally to reduce noise
        hsv_blur = cv2.GaussianBlur(hsv, (5, 1), 0)

        row_feats: list[tuple[int, float, float, float]] = []  # (abs_y, hue, sat, val)
        for y in range(sy_top, sy_bot):
            row_mask = body_mask[y] > 0
            n = int(np.count_nonzero(row_mask))
            if n < max(4, int(0.08 * w)):
                continue
            hue = float(np.mean(hsv_blur[y, :, 0][row_mask]))
            sat = float(np.mean(hsv_blur[y, :, 1][row_mask]))
            val = float(np.mean(hsv_blur[y, :, 2][row_mask]))
            row_feats.append((y, hue, sat, val))

        if len(row_feats) < 8:
            return {"found": False, "reason": "insufficient_rows", "waist_y": hip_y - torso_h * 0.15}

        window = max(3, len(row_feats) // 5)
        best_score = 0.0
        best_idx = len(row_feats) // 2

        for i in range(window, len(row_feats) - window):
            up_h = float(np.mean([r[1] for r in row_feats[max(0, i - window):i]]))
            dn_h = float(np.mean([r[1] for r in row_feats[i:i + window]]))
            up_s = float(np.mean([r[2] for r in row_feats[max(0, i - window):i]]))
            dn_s = float(np.mean([r[2] for r in row_feats[i:i + window]]))
            up_v = float(np.mean([r[3] for r in row_feats[max(0, i - window):i]]))
            dn_v = float(np.mean([r[3] for r in row_feats[i:i + window]]))
            # Hue difference is most meaningful for garment change; wrap-safe
            hd = min(abs(up_h - dn_h), 180.0 - abs(up_h - dn_h))
            score = hd * 0.5 + abs(up_s - dn_s) * 0.3 + abs(up_v - dn_v) * 0.2
            if score > best_score:
                best_score = score
                best_idx = i

        boundary_y_abs = row_feats[best_idx][0]
        waist_y = _clamp01(float(boundary_y_abs) / h)
        confidence = min(1.0, best_score / 28.0)
        return {
            "found": confidence > 0.18,
            "waist_y": waist_y,
            "confidence": confidence,
            "score": best_score,
        }
    except Exception as exc:
        return {"found": False, "reason": f"error:{exc}", "waist_y": None}


def _detect_torso_inclination_from_kp(kp: "dict[str, dict]") -> dict:
    """Estimate body tilt from shoulder/hip asymmetry in the current keypoints."""
    import math
    k = _normalize_keypoint_map(kp, force_visible=True)
    ls, rs = k["left_shoulder"], k["right_shoulder"]
    lh, rh = k["left_hip"], k["right_hip"]
    neck, hipc = k["neck_base"], k["hip_center"]

    sh_dx = max(1e-6, float(rs["x"]) - float(ls["x"]))
    sh_dy = float(rs["y"]) - float(ls["y"])
    shoulder_tilt = math.degrees(math.atan2(sh_dy, sh_dx))

    hip_dx = max(1e-6, float(rh["x"]) - float(lh["x"]))
    hip_dy = float(rh["y"]) - float(lh["y"])
    hip_tilt = math.degrees(math.atan2(hip_dy, hip_dx))

    torso_dx = float(hipc["x"]) - float(neck["x"])
    torso_dy = max(1e-6, float(hipc["y"]) - float(neck["y"]))
    # Angle of torso from true vertical (positive = leaning right)
    torso_lean = math.degrees(math.atan2(torso_dx, torso_dy))

    return {
        "shoulder_tilt_deg": shoulder_tilt,
        "hip_tilt_deg": hip_tilt,
        "torso_lean_deg": torso_lean,
        "avg_tilt_deg": (shoulder_tilt + hip_tilt) / 2.0,
        "is_tilted": abs(torso_lean) > 2.5,
    }


def _apply_skin_guided_limb_lines(
    kp: "dict[str, dict]",
    limb_profiles: dict,
    waist_info: "dict | None",
    inclination_info: "dict | None",
    alpha: float = 0.25,
) -> "tuple[dict[str, dict], dict]":
    """
    Enhanced skin-based keypoint correction:
    - Guides arm/forearm joints to follow skin-colour columns.
    - Guides leg/calf joints to follow skin-colour columns.
    - Simulates a bend at the joint when skin has an internal gap.
    - Applies torso lean to crown/chin when the shoulder line is tilted.
    - Corrects hip_center/left_hip/right_hip to the detected waist boundary.
    """
    import math

    out = _normalize_keypoint_map(kp, force_visible=True)

    notes: list[str] = []

    def blend(old: float, new: float, a: float) -> float:
        return _clamp01((1.0 - a) * old + a * new)

    def skin_x_near_y(profile: dict, y_norm: float, delta: float = 0.06) -> "float | None":
        rows = profile.get("rows") or []
        close = [r for r in rows if abs(float(r["y"]) - y_norm) <= delta]
        if not close:
            return None
        return float(sum(r["x_center"] for r in close) / len(close))

    def skin_x_range_near_y(profile: dict, y_norm: float, delta: float = 0.06) -> "tuple[float, float] | None":
        rows = profile.get("rows") or []
        close = [r for r in rows if abs(float(r["y"]) - y_norm) <= delta]
        if not close:
            return None
        return (float(sum(r["x_min"] for r in close) / len(close)),
                float(sum(r["x_max"] for r in close) / len(close)))

    # ── ARM / FOREARM guidance ──────────────────────────────────────────────
    arm_segs = [
        ("left_arm",     "left_shoulder",  "left_elbow",  "left"),
        ("right_arm",    "right_shoulder", "right_elbow", "right"),
        ("left_forearm", "left_elbow",     "left_wrist",  "left"),
        ("right_forearm","right_elbow",    "right_wrist", "right"),
    ]
    arm_alpha = min(0.38, alpha * 1.6)

    for limb_id, top_kp, bot_kp, side in arm_segs:
        prof = limb_profiles.get(limb_id) or {}
        if not prof.get("has_skin"):
            continue
        cont = float(prof.get("continuity", 0.4))
        a = arm_alpha * cont

        # Guide top joint x
        top_y = float(out[top_kp]["y"])
        bot_y = float(out[bot_kp]["y"])
        sx_top = skin_x_near_y(prof, top_y)
        if sx_top is not None:
            out[top_kp]["x"] = blend(float(out[top_kp]["x"]), sx_top, a)
            notes.append(f"{top_kp}_x")

        sx_bot = skin_x_near_y(prof, bot_y)
        if sx_bot is not None:
            out[bot_kp]["x"] = blend(float(out[bot_kp]["x"]), sx_bot, a)
            notes.append(f"{bot_kp}_x")

        # Flex simulation: if there is an internal gap in skin rows, interpret it as
        # a bend of the joint (elbow or wrist hidden by body / clothing transition).
        if prof.get("suggest_flex"):
            rows_ys = sorted(float(r["y"]) for r in (prof.get("rows") or []))
            if len(rows_ys) >= 4:
                gaps = [
                    (rows_ys[i + 1] - rows_ys[i], (rows_ys[i] + rows_ys[i + 1]) / 2.0)
                    for i in range(len(rows_ys) - 1)
                ]
                if gaps:
                    biggest = max(gaps, key=lambda g: g[0])
                    gap_y = biggest[1]
                    joint_y = float(out[bot_kp]["y"])
                    if abs(joint_y - gap_y) < 0.07 and biggest[0] > 0.02:
                        # Nudge joint outward to simulate fold
                        sign = -1.0 if side == "left" else 1.0
                        nudge = sign * 0.035 * a * 2.5
                        out[bot_kp]["x"] = _clamp01(float(out[bot_kp]["x"]) + nudge)
                        notes.append(f"{bot_kp}_flex")

    # ── LEG / CALF guidance ─────────────────────────────────────────────────
    leg_segs = [
        ("left_leg",  "left_hip",  "left_knee",  "left"),
        ("right_leg", "right_hip", "right_knee", "right"),
        ("left_calf", "left_knee", "left_ankle", "left"),
        ("right_calf","right_knee","right_ankle","right"),
    ]
    leg_alpha = min(0.45, alpha * 2.0)  # skin on bare legs is very reliable

    for limb_id, top_kp, bot_kp, side in leg_segs:
        prof = limb_profiles.get(limb_id) or {}
        if not prof.get("has_skin"):
            continue
        rows = prof.get("rows") or []
        if len(rows) < 3:
            continue
        cont = float(prof.get("continuity", 0.4))
        a = leg_alpha * cont

        # Sample the skin column at multiple points along the segment and guide each joint
        top_y = float(out[top_kp]["y"])
        bot_y = float(out[bot_kp]["y"])
        for t in (0.15, 0.5, 0.85):
            y_sample = top_y + t * (bot_y - top_y)
            sx = skin_x_near_y(prof, y_sample, delta=0.04)
            if sx is None:
                continue
            kp_name = top_kp if t < 0.5 else bot_kp
            out[kp_name]["x"] = blend(float(out[kp_name]["x"]), sx, a * 0.7)
        notes.append(f"{limb_id}_guided")

        # Also refine the y-position of the bottom joint when a discontinuity suggests
        # a clothing boundary (e.g., pants end) or a bent knee.
        if prof.get("suggest_flex"):
            rows_ys = sorted(float(r["y"]) for r in rows)
            if len(rows_ys) >= 4:
                gaps = [
                    (rows_ys[i + 1] - rows_ys[i], (rows_ys[i] + rows_ys[i + 1]) / 2.0)
                    for i in range(len(rows_ys) - 1)
                ]
                if gaps:
                    biggest = max(gaps, key=lambda g: g[0])
                    gap_y, gap_sz = biggest[1], biggest[0]
                    joint_y = float(out[bot_kp]["y"])
                    # If the gap is bigger than 3% of image height and near the joint, adjust
                    if gap_sz > 0.03 and abs(joint_y - gap_y) < 0.08:
                        out[bot_kp]["y"] = blend(joint_y, gap_y, a * 0.4)
                        notes.append(f"{bot_kp}_y_gap")

    # ── TORSO LEAN ─────────────────────────────────────────────────────────
    if inclination_info and inclination_info.get("is_tilted"):
        lean_deg = float(inclination_info.get("torso_lean_deg", 0.0))
        if 2.5 < abs(lean_deg) < 28.0:
            tilt_alpha = min(0.20, alpha * 0.8)
            neck = out["neck_base"]
            for name in ("crown", "chin"):
                pt = out[name]
                rel_y = float(neck["y"]) - float(pt["y"])  # positive when pt is above neck
                # Expected x: follow the lean direction
                shift = math.tan(math.radians(lean_deg)) * rel_y * 0.35
                target_x = _clamp01(float(neck["x"]) - shift)
                out[name]["x"] = blend(float(pt["x"]), target_x, tilt_alpha)
            notes.append(f"torso_lean_{lean_deg:.1f}deg")

    # ── WAIST BOUNDARY ─────────────────────────────────────────────────────
    if waist_info and waist_info.get("found") and float(waist_info.get("confidence", 0.0)) > 0.18:
        waist_y = float(waist_info["waist_y"])
        waist_conf = float(waist_info["confidence"])
        # Hip joints should be at or just below the waist boundary
        hipy_now = float(out["hip_center"]["y"])
        # Only update if detected waist is above current hip estimate (shouldn't push hips up into torso)
        if waist_y < hipy_now - 0.01:
            wa = min(0.30, waist_conf * 0.45)
            # Place hip slightly below waist boundary (waist line is top of hips)
            offset_below = max(0.02, (hipy_now - waist_y) * 0.25)
            target_hip_y = _clamp01(waist_y + offset_below)
            out["hip_center"]["y"] = blend(hipy_now, target_hip_y, wa)
            for side in ("left", "right"):
                hn = f"{side}_hip"
                out[hn]["y"] = blend(float(out[hn]["y"]), target_hip_y, wa)
            notes.append(f"waist_y={waist_y:.3f}")

    return out, {
        "applied": len(notes) > 0,
        "notes": notes,
        "limbs_with_skin": [k for k, v in limb_profiles.items() if v.get("has_skin")],
    }


def _estimate_base_keypoints_for_manifest(
    manifest: dict,
    pose_backend_override: str | None = None,
) -> tuple[dict[str, dict], str, dict]:
    face_bbox = None
    face_meta = {"strategy": "opencv_haar", "reason": "no_image"}
    skin_lines = None
    skin_meta = {"strategy": "skin_lines", "reason": "no_image"}
    image_path: "Path | None" = None

    def _with_skin(kp_map: dict[str, dict], source: str, params: dict) -> tuple[dict[str, dict], str, dict]:
        guided, guided_meta = _apply_skin_guided_body_lines(kp_map, skin_lines)
        # Enhanced per-limb skin guidance + waist + inclination
        limb_profiles: dict = {}
        waist_info: dict = {}
        inclination_info: dict = {}
        limb_meta: dict = {}
        if image_path is not None:
            limb_profiles = _detect_skin_limb_profiles(image_path, mask_path, face_bbox, guided)
            waist_info = _detect_waist_color_boundary(image_path, mask_path, guided)
            inclination_info = _detect_torso_inclination_from_kp(guided)
            guided, limb_meta = _apply_skin_guided_limb_lines(
                guided, limb_profiles, waist_info, inclination_info
            )
        merged = {
            **params,
            "skin_lines": {
                "detector": skin_meta,
                "guidance": guided_meta,
                "limb_guidance": limb_meta,
                "waist": waist_info,
                "inclination": inclination_info,
            },
        }
        return _normalize_keypoint_map(guided, force_visible=True), source, merged

    image_rel = manifest.get("image_path")
    mask_rel = manifest.get("mask_path") or manifest.get("auto_mask_path")
    mask_path = _resolve_input_path(mask_rel) if mask_rel else None
    if image_rel:
        _img_p = _resolve_input_path(image_rel)
        if _img_p.exists():
            image_path = _img_p
            face_bbox, face_meta = _detect_face_bbox(image_path)
            skin_lines, skin_meta = _detect_skin_body_lines(image_path, mask_path=mask_path, face_hint=face_bbox)
            pose_backend = str(pose_backend_override or os.getenv("HUMAN_SHAPE_POSE_BACKEND", "legacy")).strip().lower()

            if pose_backend in ("landmarker", "tasks", "v2", "both"):
                kp_v2, mp_v2_meta = _mediapipe_landmarker_keypoints(image_path)
                if kp_v2:
                    return _with_skin(_normalize_keypoint_map(kp_v2, force_visible=True), "mediapipe_pose_landmarker", {
                        "strategy": "mediapipe_pose_landmarker",
                        "image_path": _to_repo_relative(image_path),
                        "pose_backend": pose_backend,
                        "pose_backend_meta": mp_v2_meta,
                        "face_detection": {"bbox": face_bbox, **face_meta},
                    })

            if pose_backend in ("legacy", "v1", "both", ""):
                kp = _mediapipe_keypoints(image_path)
                if kp:
                    return _with_skin(_normalize_keypoint_map(kp, force_visible=True), "mediapipe_pose", {
                        "strategy": "mediapipe_pose",
                        "image_path": _to_repo_relative(image_path),
                        "pose_backend": pose_backend,
                        "face_detection": {"bbox": face_bbox, **face_meta},
                    })

    auto_backend = str(manifest.get("auto_backend") or "")
    is_corrected = bool(manifest.get("is_corrected"))
    if auto_backend == "fallback_ellipse" and not is_corrected:
        # Fallback ellipse is centered by design; for these samples prefer face anchor directly.
        face_geo, face_geo_params = _face_anchored_geometric(face_bbox)
        params = {
            **face_geo_params,
            "reason": "skip_silhouette_due_to_fallback_ellipse",
            "face_detection": {"bbox": face_bbox, **face_meta},
        }
        return _with_skin(face_geo, str(face_geo_params.get("strategy", "geometric")), params)

    sil_kp, sil_params = _silhouette_keypoints_from_manifest(manifest, face_hint=face_bbox)
    if sil_kp:
        params = {
            **sil_params,
            "face_detection": {"bbox": face_bbox, **face_meta},
        }
        return _with_skin(sil_kp, "silhouette_mask", params)

    face_geo, face_geo_params = _face_anchored_geometric(face_bbox)
    params = {
        **face_geo_params,
        "face_detection": {"bbox": face_bbox, **face_meta},
    }
    return _with_skin(face_geo, face_geo_params.get("strategy", "geometric"), params)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_keypoint_map(raw: dict | None, force_visible: bool = True) -> dict[str, dict]:
    """Ensure every sample has the same controllable keypoint schema."""
    base_tpl = _geometric_keypoints()
    src_map = raw or {}
    out: dict[str, dict] = {}
    for name in BODY_KEYPOINT_NAMES:
        tpl = base_tpl.get(name) or {"x": 0.5, "y": 0.5, "visible": True, "optional": False}
        src = src_map.get(name) or {}
        out[name] = {
            "x": _clamp01(src.get("x", tpl.get("x", 0.5))),
            "y": _clamp01(src.get("y", tpl.get("y", 0.5))),
            "visible": True if force_visible else bool(src.get("visible", tpl.get("visible", True))),
            "optional": bool(src.get("optional", tpl.get("optional", False))),
        }
    return out


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


def _apply_golden_ratio_prior(kp: dict[str, dict], alpha: float = 0.35) -> tuple[dict[str, dict], dict]:
    """Apply a soft full-body proportion prior (golden-ratio-inspired) over keypoints.

    This keeps predictions consistent when full body is visible while preserving detected pose/offsets.
    """
    if alpha <= 0.0:
        return kp, {"applied": False, "reason": "alpha_zero"}

    phi = 1.61803398875
    out = _normalize_keypoint_map(kp, force_visible=True)

    crown = out.get("crown") or {"x": 0.5, "y": 0.04}
    left_ankle = out.get("left_ankle") or {"y": 0.89}
    right_ankle = out.get("right_ankle") or {"y": 0.89}
    ankle_y = (float(left_ankle.get("y", 0.89)) + float(right_ankle.get("y", 0.89))) / 2.0
    top_y = float(crown.get("y", 0.04))
    body_h = max(0.18, ankle_y - top_y)

    center_x = float((out.get("left_hip", {}).get("x", 0.41) + out.get("right_hip", {}).get("x", 0.59)) / 2.0)
    shoulder_half_now = abs(float(out.get("right_shoulder", {}).get("x", 0.65)) - float(out.get("left_shoulder", {}).get("x", 0.35))) / 2.0
    shoulder_half = max(0.06, shoulder_half_now)
    hip_half = max(0.04, shoulder_half / phi)
    knee_half = max(0.03, hip_half / phi)
    ankle_half = max(0.02, knee_half / phi)

    # Golden-section inspired vertical anchors.
    chin_y = top_y + body_h * 0.09
    neck_y = top_y + body_h * 0.14
    shoulder_y = top_y + body_h * 0.18
    hip_y = top_y + body_h * 0.618
    knee_y = hip_y + (ankle_y - hip_y) * 0.618

    def blend(v_old: float, v_new: float) -> float:
        return _clamp01((1.0 - alpha) * float(v_old) + alpha * float(v_new))

    # Centerline points
    for name, yv in {
        "crown": top_y,
        "chin": chin_y,
        "neck_base": neck_y,
        "hip_center": hip_y,
    }.items():
        if name in out:
            out[name]["x"] = blend(out[name].get("x", center_x), center_x)
            out[name]["y"] = blend(out[name].get("y", yv), yv)

    # Bilateral points with taper from shoulders to ankles.
    for l_name, r_name, yv, half_w in [
        ("left_shoulder", "right_shoulder", shoulder_y, shoulder_half),
        ("left_hip", "right_hip", hip_y, hip_half),
        ("left_knee", "right_knee", knee_y, knee_half),
        ("left_ankle", "right_ankle", ankle_y, ankle_half),
    ]:
        if l_name in out:
            out[l_name]["x"] = blend(out[l_name].get("x", center_x - half_w), center_x - half_w)
            out[l_name]["y"] = blend(out[l_name].get("y", yv), yv)
        if r_name in out:
            out[r_name]["x"] = blend(out[r_name].get("x", center_x + half_w), center_x + half_w)
            out[r_name]["y"] = blend(out[r_name].get("y", yv), yv)

    # Keep elbow/wrist/toe tied to nearby joints while preserving existing asymmetry lightly.
    for elbow, shoulder, hip in [
        ("left_elbow", "left_shoulder", "left_hip"),
        ("right_elbow", "right_shoulder", "right_hip"),
    ]:
        if elbow in out and shoulder in out and hip in out:
            target_y = (float(out[shoulder]["y"]) + float(out[hip]["y"])) * 0.55
            out[elbow]["y"] = blend(out[elbow].get("y", target_y), target_y)

    for wrist, elbow, knee in [
        ("left_wrist", "left_elbow", "left_knee"),
        ("right_wrist", "right_elbow", "right_knee"),
    ]:
        if wrist in out and elbow in out and knee in out:
            target_y = (float(out[elbow]["y"]) + float(out[knee]["y"])) * 0.52
            out[wrist]["y"] = blend(out[wrist].get("y", target_y), target_y)

    for toe, ankle, sign in [
        ("left_toe", "left_ankle", -1.0),
        ("right_toe", "right_ankle", 1.0),
    ]:
        if toe in out and ankle in out:
            out[toe]["y"] = blend(out[toe].get("y", ankle_y + body_h * 0.04), ankle_y + body_h * 0.04)
            out[toe]["x"] = blend(out[toe].get("x", out[ankle]["x"] + sign * ankle_half * 0.7), out[ankle]["x"] + sign * ankle_half * 0.7)

    return out, {
        "applied": True,
        "alpha": alpha,
        "phi": phi,
        "center_x": center_x,
        "body_height": body_h,
    }


def _elbow_angle_deg(shoulder: dict, elbow: dict, wrist: dict) -> float:
    v1 = (float(shoulder.get("x", 0.0)) - float(elbow.get("x", 0.0)), float(shoulder.get("y", 0.0)) - float(elbow.get("y", 0.0)))
    v2 = (float(wrist.get("x", 0.0)) - float(elbow.get("x", 0.0)), float(wrist.get("y", 0.0)) - float(elbow.get("y", 0.0)))
    n1 = (v1[0] * v1[0] + v1[1] * v1[1]) ** 0.5
    n2 = (v2[0] * v2[0] + v2[1] * v2[1]) ** 0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    import math
    return float(math.degrees(math.acos(c)))


def _joint_angle_deg(a: dict, b: dict, c: dict) -> float:
    v1 = (float(a.get("x", 0.0)) - float(b.get("x", 0.0)), float(a.get("y", 0.0)) - float(b.get("y", 0.0)))
    v2 = (float(c.get("x", 0.0)) - float(b.get("x", 0.0)), float(c.get("y", 0.0)) - float(b.get("y", 0.0)))
    n1 = (v1[0] * v1[0] + v1[1] * v1[1]) ** 0.5
    n2 = (v2[0] * v2[0] + v2[1] * v2[1]) ** 0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    ccos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    import math
    return float(math.degrees(math.acos(ccos)))


def _classify_arm_side_pose(side: str, kp: dict[str, dict]) -> dict:
    shoulder = kp.get(f"{side}_shoulder") or {}
    elbow = kp.get(f"{side}_elbow") or {}
    wrist = kp.get(f"{side}_wrist") or {}
    dx = float(wrist.get("x", 0.0)) - float(shoulder.get("x", 0.0))
    dy = float(wrist.get("y", 0.0)) - float(shoulder.get("y", 0.0))
    angle = _elbow_angle_deg(shoulder, elbow, wrist)
    straight = angle >= 150.0
    if dy < -0.03:
        orient = "up"
    elif abs(dx) > abs(dy) * 1.2 and abs(dy) < 0.12:
        orient = "side"
    elif dy > 0.14:
        orient = "down"
    else:
        orient = "front"
    return {
        "label": f"{side}_{orient}_{'straight' if straight else 'bent'}",
        "orientation": orient,
        "straight": straight,
        "elbow_angle": angle,
    }


def _arm_pose_pair_label(kp: dict[str, dict]) -> dict:
    left = _classify_arm_side_pose("left", kp)
    right = _classify_arm_side_pose("right", kp)
    return {
        "left": left,
        "right": right,
        "pair": f"{left['label']} | {right['label']}",
    }


ARM_POSE_CATALOG: list[dict] = [
    {"id": "arms_slightly_apart", "label": "Brazos ligeramente separados del cuerpo", "frequency": "alta"},
    {"id": "arms_relaxed_hanging", "label": "Brazos relajados (colgando)", "frequency": "alta"},
    {"id": "hands_on_waist", "label": "Manos en la cintura", "frequency": "alta"},
    {"id": "hands_in_pockets", "label": "Manos en bolsillos", "frequency": "alta"},
    {"id": "one_relaxed_one_action", "label": "Un brazo relajado y otro en accion", "frequency": "alta"},
    {"id": "holding_garment", "label": "Sujetando la prenda", "frequency": "media"},
    {"id": "adjusting_garment", "label": "Ajustando la prenda", "frequency": "media"},
    {"id": "walking_motion", "label": "Brazos en movimiento (caminata)", "frequency": "media"},
    {"id": "one_forward_one_back", "label": "Un brazo adelantado y otro retrasado", "frequency": "media"},
    {"id": "semi_flexed", "label": "Brazos semiflexionados", "frequency": "media"},
    {"id": "arms_back", "label": "Brazos hacia atras", "frequency": "media"},
    {"id": "lateral_extended", "label": "Brazos extendidos lateralmente", "frequency": "baja"},
    {"id": "one_crossing_torso", "label": "Un brazo cruzando el torso (ligero)", "frequency": "baja"},
    {"id": "arms_crossed", "label": "Brazos cruzados", "frequency": "baja"},
    {"id": "one_hand_face_neck", "label": "Una mano en el rostro o cuello", "frequency": "baja"},
    {"id": "one_arm_raised", "label": "Un brazo levantado", "frequency": "baja"},
    {"id": "both_arms_raised", "label": "Brazos elevados (ambos)", "frequency": "baja"},
    {"id": "a_pose", "label": "A-Pose (brazos inclinados)", "frequency": "tecnica"},
    {"id": "t_pose", "label": "T-Pose (brazos horizontales)", "frequency": "tecnica"},
    {"id": "arms_close_to_body", "label": "Brazos pegados al cuerpo", "frequency": "tecnica"},
    {"id": "arms_straight_rigid", "label": "Brazos rectos rigidos", "frequency": "tecnica"},
]


LEG_FOOT_POSE_CATALOG: list[dict] = [
    {"id": "legs_straight_parallel", "label": "Piernas rectas paralelas", "frequency": "alta"},
    {"id": "legs_slightly_apart", "label": "Piernas ligeramente separadas", "frequency": "alta"},
    {"id": "weight_evenly_distributed", "label": "Peso distribuido en ambas piernas", "frequency": "alta"},
    {"id": "contrapposto", "label": "Peso en una pierna (contrapposto)", "frequency": "alta"},
    {"id": "front_leg", "label": "Pierna adelantada", "frequency": "media"},
    {"id": "crossed_front", "label": "Pierna cruzada al frente", "frequency": "media"},
    {"id": "crossed_back", "label": "Pierna cruzada atras", "frequency": "media"},
    {"id": "wide_base", "label": "Piernas abiertas (base amplia)", "frequency": "media"},
    {"id": "one_knee_slightly_bent", "label": "Una rodilla ligeramente flexionada", "frequency": "alta"},
    {"id": "both_knees_semiflexed", "label": "Ambas rodillas semiflexionadas", "frequency": "media"},
    {"id": "one_leg_lateral_extended", "label": "Pierna extendida lateralmente", "frequency": "baja"},
    {"id": "walking_step", "label": "Paso en movimiento (caminar)", "frequency": "media"},
    {"id": "heel_raised", "label": "Talon levantado", "frequency": "media"},
    {"id": "leg_backward", "label": "Pierna en retroceso", "frequency": "media"},
    {"id": "marked_crossed", "label": "Piernas cruzadas marcadas", "frequency": "baja"},
    {"id": "tiptoe_support", "label": "Apoyo en puntas (tipo ballet ligero)", "frequency": "baja"},
    {"id": "knees_together_feet_apart", "label": "Rodillas juntas y pies separados", "frequency": "baja"},
    {"id": "a_pose_lower", "label": "Piernas en A (A-Pose inferior)", "frequency": "tecnica"},
    {"id": "legs_straight_rigid", "label": "Piernas totalmente rectas rigidas", "frequency": "tecnica"},
    {"id": "legs_fully_together", "label": "Piernas completamente juntas", "frequency": "tecnica"},
]


def _arm_pose_catalog_index() -> dict[str, dict]:
    return {str(x["id"]): x for x in ARM_POSE_CATALOG}


def _leg_pose_catalog_index() -> dict[str, dict]:
    return {str(x["id"]): x for x in LEG_FOOT_POSE_CATALOG}


def _torso_thickness_bucket(ratio: float) -> str:
    if ratio < 0.33:
        return "delgado"
    if ratio < 0.46:
        return "medio"
    return "ancho"


def _analyze_torso_and_arm_fold(kp: dict[str, dict]) -> dict:
    k = _normalize_keypoint_map(kp, force_visible=True)
    ls, rs = k["left_shoulder"], k["right_shoulder"]
    lh, rh = k["left_hip"], k["right_hip"]
    neck, hipc = k["neck_base"], k["hip_center"]
    le, re = k["left_elbow"], k["right_elbow"]
    lw, rw = k["left_wrist"], k["right_wrist"]

    shoulder_w = abs(float(rs["x"]) - float(ls["x"]))
    hip_w = abs(float(rh["x"]) - float(lh["x"]))
    torso_h = max(1e-6, abs(float(hipc["y"]) - float(neck["y"])))
    torso_thickness = (shoulder_w + hip_w) / 2.0
    torso_ratio = torso_thickness / torso_h

    left_elbow_angle = _elbow_angle_deg(ls, le, lw)
    right_elbow_angle = _elbow_angle_deg(rs, re, rw)
    left_fold_deg = max(0.0, 180.0 - left_elbow_angle)
    right_fold_deg = max(0.0, 180.0 - right_elbow_angle)

    def _fold_level(deg: float) -> str:
        if deg < 20.0:
            return "leve"
        if deg < 45.0:
            return "medio"
        return "marcado"

    return {
        "torso": {
            "shoulder_width": shoulder_w,
            "hip_width": hip_w,
            "torso_height": torso_h,
            "thickness": torso_thickness,
            "thickness_ratio": torso_ratio,
            "thickness_bucket": _torso_thickness_bucket(torso_ratio),
        },
        "arm_fold": {
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_fold_deg": left_fold_deg,
            "right_fold_deg": right_fold_deg,
            "left_fold_level": _fold_level(left_fold_deg),
            "right_fold_level": _fold_level(right_fold_deg),
            "avg_fold_deg": (left_fold_deg + right_fold_deg) / 2.0,
        },
    }


def _recognize_arm_pose_catalog(kp: dict[str, dict]) -> dict:
    """Heuristic classifier for 21 arm pose categories with ranked candidates."""
    k = _normalize_keypoint_map(kp, force_visible=True)
    ls, rs = k["left_shoulder"], k["right_shoulder"]
    le, re = k["left_elbow"], k["right_elbow"]
    lw, rw = k["left_wrist"], k["right_wrist"]
    lh, rh = k["left_hip"], k["right_hip"]
    neck, chin = k["neck_base"], k["chin"]

    sh_w = max(1e-6, abs(float(rs["x"]) - float(ls["x"])))
    torso_h = max(1e-6, abs(float(k["hip_center"]["y"]) - float(neck["y"])))
    body_arm = _analyze_torso_and_arm_fold(k)
    left_ang = float(body_arm["arm_fold"]["left_elbow_angle"])
    right_ang = float(body_arm["arm_fold"]["right_elbow_angle"])
    both_straight = left_ang >= 160 and right_ang >= 160
    both_bent = left_ang < 145 and right_ang < 145

    left_up = float(lw["y"]) < float(ls["y"]) - 0.03
    right_up = float(rw["y"]) < float(rs["y"]) - 0.03
    left_down = float(lw["y"]) > float(lh["y"]) - 0.02
    right_down = float(rw["y"]) > float(rh["y"]) - 0.02

    left_outer = float(lw["x"]) < float(ls["x"]) - 0.10 * sh_w
    right_outer = float(rw["x"]) > float(rs["x"]) + 0.10 * sh_w
    arms_apart = left_outer and right_outer

    wrists_close_waist = (abs(float(lw["y"]) - float(lh["y"])) < 0.09 * torso_h and abs(float(rw["y"]) - float(rh["y"])) < 0.09 * torso_h)
    wrists_near_center = (abs(float(lw["x"]) - float(k["hip_center"]["x"])) < 0.28 * sh_w and abs(float(rw["x"]) - float(k["hip_center"]["x"])) < 0.28 * sh_w)

    crossed = float(lw["x"]) > float(k["hip_center"]["x"]) and float(rw["x"]) < float(k["hip_center"]["x"])
    one_cross = (float(lw["x"]) > float(k["hip_center"]["x"]) > float(ls["x"])) ^ (float(rw["x"]) < float(k["hip_center"]["x"]) < float(rs["x"]))

    near_face = (abs(float(lw["y"]) - float(chin["y"])) < 0.16 * torso_h or abs(float(rw["y"]) - float(chin["y"])) < 0.16 * torso_h)
    t_like = both_straight and abs(float(lw["y"]) - float(ls["y"])) < 0.08 * torso_h and abs(float(rw["y"]) - float(rs["y"])) < 0.08 * torso_h and arms_apart
    a_like = both_straight and float(lw["y"]) > float(ls["y"]) and float(rw["y"]) > float(rs["y"]) and arms_apart
    rigid = both_straight and abs(float(lw["x"]) - float(ls["x"])) < 0.10 * sh_w and abs(float(rw["x"]) - float(rs["x"])) < 0.10 * sh_w
    close_body = abs(float(lw["x"]) - float(ls["x"])) < 0.18 * sh_w and abs(float(rw["x"]) - float(rs["x"])) < 0.18 * sh_w and left_down and right_down

    scores: dict[str, float] = {item["id"]: 0.0 for item in ARM_POSE_CATALOG}
    def add(pid: str, val: float):
        scores[pid] = scores.get(pid, 0.0) + float(val)

    if arms_apart and left_down and right_down: add("arms_slightly_apart", 1.3)
    if left_down and right_down and both_bent: add("arms_relaxed_hanging", 1.2)
    if wrists_close_waist and wrists_near_center: add("hands_on_waist", 1.4)
    if wrists_close_waist and both_bent: add("hands_in_pockets", 1.0)
    if (left_down and right_up) or (right_down and left_up): add("one_relaxed_one_action", 1.1)
    if abs(float(lw["y"]) - float(neck["y"])) < 0.18 * torso_h or abs(float(rw["y"]) - float(neck["y"])) < 0.18 * torso_h: add("holding_garment", 0.9)
    if both_bent and wrists_near_center: add("adjusting_garment", 0.95)
    if (float(lw["x"]) - float(ls["x"])) * (float(rw["x"]) - float(rs["x"])) < 0: add("walking_motion", 0.9)
    if (float(lw["x"]) - float(ls["x"])) * (float(rw["x"]) - float(rs["x"])) < 0 and both_straight: add("one_forward_one_back", 0.85)
    if left_ang < 155 and right_ang < 155: add("semi_flexed", 1.0)
    if float(lw["x"]) > float(ls["x"]) and float(rw["x"]) < float(rs["x"]) and left_down and right_down: add("arms_back", 0.8)
    if t_like: add("lateral_extended", 1.2)
    if one_cross: add("one_crossing_torso", 1.1)
    if crossed: add("arms_crossed", 1.3)
    if near_face: add("one_hand_face_neck", 1.2)
    if (left_up ^ right_up): add("one_arm_raised", 1.15)
    if left_up and right_up: add("both_arms_raised", 1.3)
    if a_like: add("a_pose", 1.4)
    if t_like: add("t_pose", 1.5)
    if close_body: add("arms_close_to_body", 1.1)
    if rigid: add("arms_straight_rigid", 1.0)
    if float(body_arm["arm_fold"]["avg_fold_deg"]) >= 45.0:
        add("semi_flexed", 0.35)
        add("adjusting_garment", 0.25)

    # Soft defaults for common catalog poses
    if scores["arms_relaxed_hanging"] == 0 and left_down and right_down:
        add("arms_relaxed_hanging", 0.6)
    if scores["arms_slightly_apart"] == 0 and arms_apart:
        add("arms_slightly_apart", 0.5)

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    idx = _arm_pose_catalog_index()
    candidates = [
        {
            "id": pid,
            "label": idx.get(pid, {}).get("label", pid),
            "frequency": idx.get(pid, {}).get("frequency", "desconocida"),
            "score": sc,
        }
        for pid, sc in ranking[:5]
    ]
    top = candidates[0] if candidates else None
    return {
        "top": top,
        "candidates": candidates,
        "features": {
            "left_elbow_angle": left_ang,
            "right_elbow_angle": right_ang,
            "left_fold_deg": float(body_arm["arm_fold"]["left_fold_deg"]),
            "right_fold_deg": float(body_arm["arm_fold"]["right_fold_deg"]),
            "torso_thickness": float(body_arm["torso"]["thickness"]),
            "torso_thickness_ratio": float(body_arm["torso"]["thickness_ratio"]),
            "torso_thickness_bucket": str(body_arm["torso"]["thickness_bucket"]),
            "arms_apart": arms_apart,
            "crossed": crossed,
            "left_up": left_up,
            "right_up": right_up,
            "t_like": t_like,
            "a_like": a_like,
        },
    }


def _recognize_leg_foot_pose_catalog(kp: dict[str, dict]) -> dict:
    k = _normalize_keypoint_map(kp, force_visible=True)
    lh, rh = k["left_hip"], k["right_hip"]
    lk, rk = k["left_knee"], k["right_knee"]
    la, ra = k["left_ankle"], k["right_ankle"]
    lt, rt = k["left_toe"], k["right_toe"]

    hip_w = max(1e-6, abs(float(rh["x"]) - float(lh["x"])))
    leg_h = max(1e-6, (abs(float(la["y"]) - float(lh["y"])) + abs(float(ra["y"]) - float(rh["y"]))) / 2.0)
    knee_gap = abs(float(rk["x"]) - float(lk["x"]))
    ankle_gap = abs(float(ra["x"]) - float(la["x"]))
    foot_gap = abs(float(rt["x"]) - float(lt["x"]))

    left_knee_angle = _joint_angle_deg(lh, lk, la)
    right_knee_angle = _joint_angle_deg(rh, rk, ra)
    both_straight = left_knee_angle >= 167.0 and right_knee_angle >= 167.0
    one_bent = (left_knee_angle < 160.0) ^ (right_knee_angle < 160.0)
    both_semiflex = left_knee_angle < 165.0 and right_knee_angle < 165.0

    left_heel_up = abs(float(lt["y"]) - float(la["y"])) < 0.02 * leg_h
    right_heel_up = abs(float(rt["y"]) - float(ra["y"])) < 0.02 * leg_h
    heel_raised = left_heel_up ^ right_heel_up
    tiptoe = left_heel_up and right_heel_up

    ankles_crossed = (float(la["x"]) > float(ra["x"])) and (float(lh["x"]) < float(rh["x"]))
    cross_strength = abs(float(la["x"]) - float(ra["x"])) / max(hip_w, 1e-6)
    marked_cross = ankles_crossed and cross_strength > 0.30

    one_leg_forward = abs(float(la["y"]) - float(ra["y"])) > 0.08 * leg_h
    one_leg_lateral = (float(la["x"]) < float(lh["x"]) - 0.28 * hip_w) or (float(ra["x"]) > float(rh["x"]) + 0.28 * hip_w)
    wide_base = ankle_gap > 1.45 * hip_w
    slightly_apart = ankle_gap > 0.95 * hip_w
    fully_together = ankle_gap < 0.35 * hip_w and knee_gap < 0.45 * hip_w
    knees_together_feet_apart = knee_gap < 0.40 * hip_w and foot_gap > 0.70 * hip_w
    contrapposto = one_bent and abs(float(lh["y"]) - float(rh["y"])) > 0.02 * leg_h
    a_pose_lower = both_straight and ankle_gap > 1.05 * hip_w and ankle_gap < 1.45 * hip_w
    rigid = both_straight and abs(float(lk["x"]) - float(lh["x"])) < 0.12 * hip_w and abs(float(rk["x"]) - float(rh["x"])) < 0.12 * hip_w

    scores: dict[str, float] = {item["id"]: 0.0 for item in LEG_FOOT_POSE_CATALOG}
    def add(pid: str, val: float):
        scores[pid] = scores.get(pid, 0.0) + float(val)

    if both_straight and ankle_gap >= 0.85 * hip_w and ankle_gap <= 1.15 * hip_w:
        add("legs_straight_parallel", 1.3)
    if slightly_apart:
        add("legs_slightly_apart", 1.2)
    if not one_bent and abs(float(lh["y"]) - float(rh["y"])) < 0.02 * leg_h:
        add("weight_evenly_distributed", 1.1)
    if contrapposto:
        add("contrapposto", 1.3)
    if one_leg_forward and not ankles_crossed:
        add("front_leg", 1.1)
    if ankles_crossed and (float(la["y"]) < float(ra["y"])):
        add("crossed_front", 1.0)
    if ankles_crossed and (float(la["y"]) >= float(ra["y"])):
        add("crossed_back", 0.95)
    if wide_base:
        add("wide_base", 1.25)
    if one_bent:
        add("one_knee_slightly_bent", 1.25)
    if both_semiflex:
        add("both_knees_semiflexed", 1.2)
    if one_leg_lateral:
        add("one_leg_lateral_extended", 1.25)
    if one_leg_forward and one_bent:
        add("walking_step", 1.05)
    if heel_raised:
        add("heel_raised", 1.25)
    if one_leg_forward and (float(la["y"]) > float(ra["y"]) + 0.08 * leg_h or float(ra["y"]) > float(la["y"]) + 0.08 * leg_h):
        add("leg_backward", 0.95)
    if marked_cross:
        add("marked_crossed", 1.35)
    if tiptoe:
        add("tiptoe_support", 1.35)
    if knees_together_feet_apart:
        add("knees_together_feet_apart", 1.25)
    if a_pose_lower:
        add("a_pose_lower", 1.2)
    if rigid:
        add("legs_straight_rigid", 1.0)
    if fully_together:
        add("legs_fully_together", 1.2)

    if scores["legs_slightly_apart"] == 0 and ankle_gap > 0.80 * hip_w:
        add("legs_slightly_apart", 0.6)
    if scores["weight_evenly_distributed"] == 0 and not one_bent:
        add("weight_evenly_distributed", 0.5)

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    idx = _leg_pose_catalog_index()
    candidates = [
        {
            "id": pid,
            "label": idx.get(pid, {}).get("label", pid),
            "frequency": idx.get(pid, {}).get("frequency", "desconocida"),
            "score": sc,
        }
        for pid, sc in ranking[:5]
    ]
    top = candidates[0] if candidates else None
    return {
        "top": top,
        "candidates": candidates,
        "features": {
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "ankle_gap": ankle_gap,
            "knee_gap": knee_gap,
            "one_bent": one_bent,
            "both_straight": both_straight,
            "heels_raised": heel_raised,
            "crossed": ankles_crossed,
            "wide_base": wide_base,
            "tiptoe": tiptoe,
        },
    }


def _build_arm_pose_prototypes(samples: list[dict]) -> dict:
    arm_points = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hand_tip", "right_hand_tip",
    ]
    groups: dict[str, list[dict]] = {}
    for item in samples:
        kp = _normalize_keypoint_map(item.get("target") or item.get("keypoints") or {}, force_visible=True)
        label = _arm_pose_pair_label(kp)["pair"]
        groups.setdefault(label, []).append(kp)

    prototypes: dict[str, dict] = {}
    for label, arr in groups.items():
        proto: dict[str, dict] = {}
        n = float(len(arr))
        for name in arm_points:
            x = sum(float(k[name]["x"]) for k in arr) / n
            y = sum(float(k[name]["y"]) for k in arr) / n
            proto[name] = {"x": x, "y": y, "visible": True, "optional": "hand_tip" in name}
        prototypes[label] = {
            "count": len(arr),
            "keypoints": proto,
        }
    return prototypes

def _build_arm_pose_catalog_prototypes(samples: list[dict]) -> dict:
    arm_points = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hand_tip", "right_hand_tip",
    ]
    groups: dict[str, list[dict]] = {}
    for item in samples:
        kp = _normalize_keypoint_map(item.get("target") or item.get("keypoints") or {}, force_visible=True)
        rec = _recognize_arm_pose_catalog(kp)
        top = rec.get("top") or {}
        pose_id = str(top.get("id") or "")
        if not pose_id:
            continue
        groups.setdefault(pose_id, []).append(kp)

    prototypes: dict[str, dict] = {}
    for pose_id, arr in groups.items():
        proto: dict[str, dict] = {}
        n = float(len(arr))
        for name in arm_points:
            x = sum(float(k[name]["x"]) for k in arr) / n
            y = sum(float(k[name]["y"]) for k in arr) / n
            proto[name] = {"x": x, "y": y, "visible": True, "optional": "hand_tip" in name}
        prototypes[pose_id] = {
            "count": len(arr),
            "keypoints": proto,
        }
    return prototypes


def _build_leg_pose_catalog_prototypes(samples: list[dict]) -> dict:
    leg_points = [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_toe", "right_toe",
    ]
    groups: dict[str, list[dict]] = {}
    for item in samples:
        kp = _normalize_keypoint_map(item.get("target") or item.get("keypoints") or {}, force_visible=True)
        rec = _recognize_leg_foot_pose_catalog(kp)
        top = rec.get("top") or {}
        pose_id = str(top.get("id") or "")
        if not pose_id:
            continue
        groups.setdefault(pose_id, []).append(kp)

    prototypes: dict[str, dict] = {}
    for pose_id, arr in groups.items():
        proto: dict[str, dict] = {}
        n = float(len(arr))
        for name in leg_points:
            x = sum(float(k[name]["x"]) for k in arr) / n
            y = sum(float(k[name]["y"]) for k in arr) / n
            proto[name] = {"x": x, "y": y, "visible": True, "optional": "toe" in name}
        prototypes[pose_id] = {
            "count": len(arr),
            "keypoints": proto,
        }
    return prototypes


def _apply_arm_pose_prior(kp: dict[str, dict], model: dict, alpha: float = 0.40) -> tuple[dict[str, dict], dict]:
    if alpha <= 0.0:
        return kp, {"applied": False, "reason": "alpha_zero"}

    out = _normalize_keypoint_map(kp, force_visible=True)
    prototypes = (model.get("arm_pose_prototypes") or {})
    catalog_prototypes = (model.get("arm_pose_catalog_prototypes") or {})
    arm_points = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hand_tip", "right_hand_tip",
    ]

    rec = _recognize_arm_pose_catalog(out)
    body_arm = _analyze_torso_and_arm_fold(out)
    top_pose = rec.get("top") or {}
    top_pose_id = str(top_pose.get("id") or "")
    top_pose_label = str(top_pose.get("label") or "")

    # Find nearest prototype by arm keypoint L2.
    best_label = None
    best_score = None
    chosen_proto = None
    chosen_mode = None

    if top_pose_id and isinstance(catalog_prototypes, dict) and top_pose_id in catalog_prototypes:
        chosen_proto = catalog_prototypes[top_pose_id].get("keypoints") or {}
        best_label = top_pose_label or top_pose_id
        chosen_mode = "catalog"

    if not chosen_proto:
        for label, item in prototypes.items():
            p = item.get("keypoints") or {}
            vals = []
            for name in arm_points:
                if name not in p:
                    continue
                vals.append(_point_dist(out.get(name, {}), p[name]))
            if not vals:
                continue
            score = sum(vals) / len(vals)
            if best_score is None or score < best_score:
                best_score = score
                best_label = label
        if best_label:
            chosen_proto = prototypes[best_label]["keypoints"]
            chosen_mode = "nearest_pair"

    adaptive_alpha = float(alpha)
    avg_fold = float(body_arm["arm_fold"]["avg_fold_deg"])
    if avg_fold >= 45.0:
        adaptive_alpha = min(0.85, float(alpha) + 0.08)
    if chosen_proto:
        for name in arm_points:
            if name not in chosen_proto:
                continue
            out[name]["x"] = _clamp01((1.0 - adaptive_alpha) * float(out[name]["x"]) + adaptive_alpha * float(chosen_proto[name]["x"]))
            out[name]["y"] = _clamp01((1.0 - adaptive_alpha) * float(out[name]["y"]) + adaptive_alpha * float(chosen_proto[name]["y"]))

    # Golden-size consistency check for arms: if mismatch but elbow angle is bent, keep bend.
    phi = 1.61803398875
    torso_h = max(0.08, float(out["hip_center"]["y"]) - float(out["neck_base"]["y"]))
    expected_upper = torso_h / phi
    expected_lower = expected_upper / phi

    bent_flags: dict[str, bool] = {}
    for side, sign in [("left", -1.0), ("right", 1.0)]:
        s = out[f"{side}_shoulder"]
        e = out[f"{side}_elbow"]
        w = out[f"{side}_wrist"]
        upper = _point_dist(s, e)
        lower = _point_dist(e, w)
        angle = _elbow_angle_deg(s, e, w)
        mismatch = (abs(upper - expected_upper) / max(expected_upper, 1e-6) > 0.35) or (abs(lower - expected_lower) / max(expected_lower, 1e-6) > 0.35)
        bent = mismatch and angle < 150.0
        bent_flags[side] = bent

        # If straight and very off expected length, softly normalize chain length.
        if (not bent) and angle >= 150.0:
            import math
            vx = float(w["x"]) - float(s["x"])
            vy = float(w["y"]) - float(s["y"])
            n = math.hypot(vx, vy)
            if n > 1e-6:
                ux, uy = vx / n, vy / n
                target_e = {
                    "x": float(s["x"]) + ux * expected_upper,
                    "y": float(s["y"]) + uy * expected_upper,
                }
                target_w = {
                    "x": float(target_e["x"]) + ux * expected_lower,
                    "y": float(target_e["y"]) + uy * expected_lower,
                }
                out[f"{side}_elbow"]["x"] = _clamp01((1.0 - alpha) * float(out[f"{side}_elbow"]["x"]) + alpha * target_e["x"])
                out[f"{side}_elbow"]["y"] = _clamp01((1.0 - alpha) * float(out[f"{side}_elbow"]["y"]) + alpha * target_e["y"])
                out[f"{side}_wrist"]["x"] = _clamp01((1.0 - alpha) * float(out[f"{side}_wrist"]["x"]) + alpha * target_w["x"])
                out[f"{side}_wrist"]["y"] = _clamp01((1.0 - alpha) * float(out[f"{side}_wrist"]["y"]) + alpha * target_w["y"])
                out[f"{side}_hand_tip"]["x"] = _clamp01((1.0 - alpha) * float(out[f"{side}_hand_tip"]["x"]) + alpha * (float(target_w["x"]) + sign * expected_lower * 0.2))
                out[f"{side}_hand_tip"]["y"] = _clamp01((1.0 - alpha) * float(out[f"{side}_hand_tip"]["y"]) + alpha * (float(target_w["y"]) + expected_lower * 0.12))

    return out, {
        "applied": True,
        "alpha": alpha,
        "adaptive_alpha": adaptive_alpha,
        "prototype_label": best_label,
        "prototype_mode": chosen_mode,
        "prototype_score": best_score,
        "bent_flags": bent_flags,
        "torso_and_arm_fold": body_arm,
        "catalog_recognition": rec,
    }


def _apply_leg_pose_prior(kp: dict[str, dict], model: dict, alpha: float = 0.35) -> tuple[dict[str, dict], dict]:
    if alpha <= 0.0:
        return kp, {"applied": False, "reason": "alpha_zero"}

    out = _normalize_keypoint_map(kp, force_visible=True)
    leg_points = [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_toe", "right_toe",
    ]
    prototypes = (model.get("leg_pose_catalog_prototypes") or {})
    rec = _recognize_leg_foot_pose_catalog(out)
    top_pose = rec.get("top") or {}
    top_pose_id = str(top_pose.get("id") or "")
    top_pose_label = str(top_pose.get("label") or "")

    chosen_proto = None
    chosen_label = None
    chosen_mode = None
    best_score = None

    if top_pose_id and isinstance(prototypes, dict) and top_pose_id in prototypes:
        chosen_proto = prototypes[top_pose_id].get("keypoints") or {}
        chosen_label = top_pose_label or top_pose_id
        chosen_mode = "catalog"

    if not chosen_proto and isinstance(prototypes, dict):
        for pose_id, item in prototypes.items():
            p = item.get("keypoints") or {}
            vals = []
            for name in leg_points:
                if name not in p:
                    continue
                vals.append(_point_dist(out.get(name, {}), p[name]))
            if not vals:
                continue
            score = sum(vals) / len(vals)
            if best_score is None or score < best_score:
                best_score = score
                chosen_label = str(pose_id)
        if chosen_label and chosen_label in prototypes:
            chosen_proto = prototypes[chosen_label].get("keypoints") or {}
            chosen_mode = "nearest_catalog"

    if chosen_proto:
        for name in leg_points:
            if name not in chosen_proto:
                continue
            out[name]["x"] = _clamp01((1.0 - alpha) * float(out[name]["x"]) + alpha * float(chosen_proto[name]["x"]))
            out[name]["y"] = _clamp01((1.0 - alpha) * float(out[name]["y"]) + alpha * float(chosen_proto[name]["y"]))

    return out, {
        "applied": True,
        "alpha": alpha,
        "prototype_label": chosen_label,
        "prototype_mode": chosen_mode,
        "prototype_score": best_score,
        "catalog_recognition": rec,
    }


def _predict_keypoints_from_base(base: dict[str, dict], model: dict) -> tuple[dict[str, dict], dict]:
    pred = _apply_refiner(base, model)
    priors = model.get("priors") or {}
    meta: dict = {}
    if bool(priors.get("enable_golden_ratio_prior", True)):
        alpha = float(priors.get("golden_ratio_alpha", 0.35))
        pred, prior_meta = _apply_golden_ratio_prior(pred, alpha=alpha)
        meta["golden_ratio"] = prior_meta
    else:
        meta["golden_ratio"] = {"applied": False, "reason": "disabled"}

    if bool(priors.get("enable_arm_pose_prior", True)):
        arm_alpha = float(priors.get("arm_pose_alpha", 0.40))
        pred, arm_meta = _apply_arm_pose_prior(pred, model=model, alpha=arm_alpha)
        meta["arm_pose"] = arm_meta
    else:
        meta["arm_pose"] = {"applied": False, "reason": "disabled"}

    if bool(priors.get("enable_leg_pose_prior", True)):
        leg_alpha = float(priors.get("leg_pose_alpha", 0.35))
        pred, leg_meta = _apply_leg_pose_prior(pred, model=model, alpha=leg_alpha)
        meta["leg_pose"] = leg_meta
    else:
        meta["leg_pose"] = {"applied": False, "reason": "disabled"}

    meta["body_shape"] = _analyze_torso_and_arm_fold(pred)

    return pred, meta


def _point_dist(a: dict, b: dict) -> float:
    dx = float(a.get("x", 0.0)) - float(b.get("x", 0.0))
    dy = float(a.get("y", 0.0)) - float(b.get("y", 0.0))
    return (dx * dx + dy * dy) ** 0.5


def _evaluate_keypoints_against_target(pred: dict[str, dict], target: dict[str, dict]) -> dict:
    pred = _normalize_keypoint_map(pred, force_visible=True)
    target = _normalize_keypoint_map(target, force_visible=True)
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
                "keypoints": _normalize_keypoint_map(kp, force_visible=True),
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


def _iterative_refiner_train(
    epochs: int,
    learning_rate: float,
    reset_model: bool,
    use_objective_snapshot: bool,
    update_only_erroneous_sections: bool,
    section_error_threshold: float,
    enable_golden_ratio_prior: bool,
    golden_ratio_alpha: float,
    enable_arm_pose_prior: bool,
    arm_pose_alpha: float,
    enable_leg_pose_prior: bool,
    leg_pose_alpha: float,
    enable_arm_pose_curriculum: bool,
    curriculum_start_fraction: float,
) -> dict:
    objective_samples, objective_source = _load_objective_samples(use_snapshot=use_objective_snapshot)
    if not objective_samples:
        raise ValueError("No hay keypoints objetivo guardados para entrenar")

    # Precompute per-sample image analysis once (face/silhouette/pose) to avoid repeated expensive IO per epoch.
    prepared_samples: list[dict] = []
    for item in objective_samples:
        sample_id = str(item.get("sample_id"))
        target = _normalize_keypoint_map(item.get("keypoints") or {}, force_visible=True)
        try:
            manifest = _load_human_shape_sample_manifest(sample_id)
        except Exception:
            continue
        base, _base_source, _base_params = _estimate_base_keypoints_for_manifest(manifest)
        prepared_samples.append(
            {
                "sample_id": sample_id,
                "target": target,
                "base": base,
            }
        )

    if not prepared_samples:
        raise ValueError("No se pudieron preparar muestras para iterar")

    model = _load_keypoint_refiner_model()
    model["arm_pose_prototypes"] = _build_arm_pose_prototypes(prepared_samples)
    model["arm_pose_catalog_prototypes"] = _build_arm_pose_catalog_prototypes(prepared_samples)
    model["leg_pose_catalog_prototypes"] = _build_leg_pose_catalog_prototypes(prepared_samples)
    model["priors"] = {
        "enable_golden_ratio_prior": bool(enable_golden_ratio_prior),
        "golden_ratio_alpha": float(golden_ratio_alpha),
        "enable_arm_pose_prior": bool(enable_arm_pose_prior),
        "arm_pose_alpha": float(arm_pose_alpha),
        "enable_leg_pose_prior": bool(enable_leg_pose_prior),
        "leg_pose_alpha": float(leg_pose_alpha),
    }
    if reset_model:
        model["offsets"] = _keypoint_template()
        model["iterations"] = 0

    epoch_reports: list[dict] = []

    # Curriculum: start with samples whose current arm pose already matches better; then progressively add harder ones.
    sample_arm_error: list[tuple[float, dict]] = []
    for item in prepared_samples:
        pred0, _meta0 = _predict_keypoints_from_base(item["base"], model)
        m0 = _evaluate_keypoints_against_target(pred0, item["target"])
        per_part = m0.get("per_part_l2") or {}
        arm_parts = [
            "left_arm", "right_arm", "left_forearm", "right_forearm", "left_hand", "right_hand",
        ]
        vals = [float(per_part[p]) for p in arm_parts if p in per_part]
        err = (sum(vals) / len(vals)) if vals else float(m0.get("mean_l2") or 0.0)
        sample_arm_error.append((err, item))
    sample_arm_error.sort(key=lambda x: x[0])

    for epoch_idx in range(1, epochs + 1):
        accum = {name: {"dx": 0.0, "dy": 0.0, "n": 0} for name in BODY_KEYPOINT_NAMES}
        eval_before: list[float] = []
        part_before_accum: dict[str, list[float]] = {str(p["id"]): [] for p in BODY_PARTS_SEGMENTS}
        part_after_accum: dict[str, list[float]] = {str(p["id"]): [] for p in BODY_PARTS_SEGMENTS}
        section_updates_count: dict[str, int] = {str(p["id"]): 0 for p in BODY_PARTS_SEGMENTS}

        if enable_arm_pose_curriculum and len(sample_arm_error) > 1:
            frac = float(curriculum_start_fraction)
            if epochs > 1:
                frac = float(curriculum_start_fraction) + (1.0 - float(curriculum_start_fraction)) * ((epoch_idx - 1) / max(1, epochs - 1))
            frac = max(0.1, min(1.0, frac))
            use_n = max(1, min(len(sample_arm_error), int(round(frac * len(sample_arm_error)))))
            epoch_samples = [it for _, it in sample_arm_error[:use_n]]
        else:
            frac = 1.0
            epoch_samples = [it for _, it in sample_arm_error]

        for item in epoch_samples:
            target = item["target"]
            base = item["base"]
            pred, _pred_meta = _predict_keypoints_from_base(base, model)
            metrics = _evaluate_keypoints_against_target(pred, target)
            if metrics.get("mean_l2") is not None:
                eval_before.append(float(metrics["mean_l2"]))

            per_part_l2 = metrics.get("per_part_l2") or {}
            for part_id, err in per_part_l2.items():
                part_before_accum.setdefault(str(part_id), []).append(float(err))

            bad_parts: set[str] = set()
            for part in BODY_PARTS_SEGMENTS:
                part_id = str(part["id"])
                err = per_part_l2.get(part_id)
                if err is None:
                    continue
                if (not update_only_erroneous_sections) or (float(err) >= section_error_threshold):
                    bad_parts.add(part_id)
                    section_updates_count[part_id] = section_updates_count.get(part_id, 0) + 1

            bad_keypoints: set[str] = set()
            for part in BODY_PARTS_SEGMENTS:
                part_id = str(part["id"])
                if part_id not in bad_parts:
                    continue
                bad_keypoints.add(str(part["kp_a"]))
                bad_keypoints.add(str(part["kp_b"]))

            for name in bad_keypoints:
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
        for item in epoch_samples:
            target = item["target"]
            base = item["base"]
            pred, _pred_meta = _predict_keypoints_from_base(base, model)
            metrics = _evaluate_keypoints_against_target(pred, target)
            if metrics.get("mean_l2") is not None:
                eval_after.append(float(metrics["mean_l2"]))
            for part_id, err in (metrics.get("per_part_l2") or {}).items():
                part_after_accum.setdefault(str(part_id), []).append(float(err))

        part_mean_before = {
            part_id: (sum(vals) / len(vals)) for part_id, vals in part_before_accum.items() if vals
        }
        part_mean_after = {
            part_id: (sum(vals) / len(vals)) for part_id, vals in part_after_accum.items() if vals
        }
        part_improvement = {
            part_id: part_mean_before[part_id] - part_mean_after[part_id]
            for part_id in part_mean_before.keys() & part_mean_after.keys()
        }

        before_mean = (sum(eval_before) / len(eval_before)) if eval_before else None
        after_mean = (sum(eval_after) / len(eval_after)) if eval_after else None
        epoch_reports.append(
            {
                "epoch": epoch_idx,
                "samples": len(eval_after),
                "mean_l2_before": before_mean,
                "mean_l2_after": after_mean,
                "improvement": (before_mean - after_mean) if before_mean is not None and after_mean is not None else None,
                "section_error_threshold": section_error_threshold,
                "update_only_erroneous_sections": update_only_erroneous_sections,
                "sections_updated_counts": section_updates_count,
                "mean_l2_by_section_before": part_mean_before,
                "mean_l2_by_section_after": part_mean_after,
                "section_improvement": part_improvement,
                "curriculum_enabled": bool(enable_arm_pose_curriculum),
                "curriculum_fraction": frac,
                "curriculum_samples_used": len(epoch_samples),
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
        "objective_samples": len(prepared_samples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "priors": model.get("priors") or {},
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
def human_shape_lab_get_keypoints(sample_id: str, force_auto: bool = False, pose_backend: str | None = None) -> dict:
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

    # Auto-estimate: choose backend without affecting the default pipeline.
    image_rel = manifest.get("image_path")
    auto_source = "geometric"
    kp = None
    backend = str(pose_backend or os.getenv("HUMAN_SHAPE_POSE_BACKEND", "legacy")).strip().lower()
    if image_rel:
        image_path = _resolve_input_path(image_rel)
        if image_path.exists():
            if backend in ("landmarker", "tasks", "v2", "both"):
                kp_v2, _meta_v2 = _mediapipe_landmarker_keypoints(image_path)
                if kp_v2:
                    kp = kp_v2
                    auto_source = "mediapipe_pose_landmarker"
            if kp is None and backend in ("legacy", "v1", "both", ""):
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

    manifest["keypoints"] = _normalize_keypoint_map(request.keypoints, force_visible=True)
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
def human_shape_lab_sample_proposal(
    sample_id: str,
    use_objective_snapshot: bool = True,
    pose_backend: str | None = None,
    apply_refiner: bool = True,
) -> dict:
    try:
        manifest = _load_human_shape_sample_manifest(sample_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

    base, base_source, base_params = _estimate_base_keypoints_for_manifest(
        manifest,
        pose_backend_override=pose_backend,
    )
    if apply_refiner:
        refiner = _load_keypoint_refiner_model()
        proposal, proposal_meta = _predict_keypoints_from_base(base, refiner)
        proposal_source = "base_plus_refiner"
    else:
        proposal = _normalize_keypoint_map(base, force_visible=True)
        proposal_meta = {
            "refiner": {
                "applied": False,
                "reason": "disabled_by_request",
            }
        }
        proposal_source = "base_only"
    proposal = _normalize_keypoint_map(proposal, force_visible=True)

    objectives, objective_source = _load_objective_samples(use_snapshot=use_objective_snapshot)
    target_map = {str(item.get("sample_id")): _normalize_keypoint_map(item.get("keypoints") or {}, force_visible=True) for item in objectives}
    target = target_map.get(sample_id)
    target_source = objective_source if target else None
    if not target:
        manifest_kp = manifest.get("keypoints")
        if isinstance(manifest_kp, dict) and manifest_kp:
            target = _normalize_keypoint_map(manifest_kp, force_visible=True)
            target_source = "sample_manifest"
    metrics = _evaluate_keypoints_against_target(proposal, target) if target else None

    control_parts = [str(p.get("id")) for p in BODY_PARTS_SEGMENTS]
    proposal_parts = [p for p in BODY_PARTS_SEGMENTS if proposal.get(p["kp_a"]) and proposal.get(p["kp_b"])]
    target_parts = [p for p in BODY_PARTS_SEGMENTS if (target or {}).get(p["kp_a"]) and (target or {}).get(p["kp_b"])]
    proposal_part_ids = [str(p["id"]) for p in proposal_parts]
    target_part_ids = [str(p["id"]) for p in target_parts]

    return {
        "status": "ok",
        "sample_id": sample_id,
        "base_source": base_source,
        "base_params": base_params,
        "proposal_source": proposal_source,
        "proposal_meta": proposal_meta,
        "proposal_keypoints": proposal,
        "target_source": target_source,
        "target_keypoints": target,
        "metrics": metrics,
        "parts": BODY_PARTS_SEGMENTS,
        "parity": {
            "control_parts_total": len(control_parts),
            "proposal_parts_total": len(proposal_part_ids),
            "target_parts_total": len(target_part_ids),
            "missing_in_proposal": [p for p in control_parts if p not in proposal_part_ids],
            "missing_in_target": [p for p in control_parts if p not in target_part_ids],
            "matches_control_schema": len(proposal_part_ids) == len(control_parts),
        },
    }


@app.post("/human-shape-lab/keypoints/iterate")
def human_shape_lab_keypoints_iterate(request: HumanShapeIterativeTrainRequest) -> dict:
    try:
        result = _iterative_refiner_train(
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            reset_model=request.reset_model,
            use_objective_snapshot=request.use_objective_snapshot,
            update_only_erroneous_sections=request.update_only_erroneous_sections,
            section_error_threshold=request.section_error_threshold,
            enable_golden_ratio_prior=request.enable_golden_ratio_prior,
            golden_ratio_alpha=request.golden_ratio_alpha,
            enable_arm_pose_prior=request.enable_arm_pose_prior,
            arm_pose_alpha=request.arm_pose_alpha,
            enable_leg_pose_prior=request.enable_leg_pose_prior,
            leg_pose_alpha=request.leg_pose_alpha,
            enable_arm_pose_curriculum=request.enable_arm_pose_curriculum,
            curriculum_start_fraction=request.curriculum_start_fraction,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **result,
        "lab": _human_shape_lab_status()["lab"],
    }


@app.post("/human-shape-lab/model/save")
def human_shape_lab_model_save(request: HumanShapeSaveModelRequest) -> dict:
    try:
        meta = _save_human_shape_model_checkpoint(
            name=request.name,
            include_objective_snapshot=request.include_objective_snapshot,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "checkpoint": meta,
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
