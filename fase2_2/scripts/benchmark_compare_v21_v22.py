from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fase2_1.core.tryon.pipeline import run_tryon
from fase2_1.core.tryon.schemas import TryOnRequest
from fase2_2.core.tryon.pipeline_v2 import run_tryon_v2

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(input_dir: str, limit: int | None = None) -> list[Path]:
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directorio invalido: {input_dir}")

    rows = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    if limit is not None and limit > 0:
        return rows[:limit]
    return rows


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)

    data = sorted(values)
    rank = (len(data) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(data) - 1)
    frac = rank - lo
    return round(data[lo] * (1.0 - frac) + data[hi] * frac, 3)


def _build_latency_stats(rows: list[dict]) -> dict:
    elapsed = [_safe_float(r.get("elapsed_ms", 0.0)) for r in rows]
    ok = [r for r in rows if r.get("status") == "ok"]
    return {
        "total": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "success_rate": round((len(ok) / len(rows)) * 100.0, 2) if rows else 0.0,
        "avg_elapsed_ms": round(sum(elapsed) / len(elapsed), 3) if elapsed else 0.0,
        "p50_elapsed_ms": percentile(elapsed, 50),
        "p90_elapsed_ms": percentile(elapsed, 90),
        "p95_elapsed_ms": percentile(elapsed, 95),
        "max_elapsed_ms": round(max(elapsed), 3) if elapsed else 0.0,
        "min_elapsed_ms": round(min(elapsed), 3) if elapsed else 0.0,
    }


def _extract_meta(result_obj) -> dict:
    if hasattr(result_obj, "model_dump"):
        payload = result_obj.model_dump()
        return payload.get("meta") or {}
    if isinstance(result_obj, dict):
        return result_obj.get("meta") or {}
    return {}


def benchmark_compare(
    input_dir: str,
    garment_path: str,
    output_dir: str,
    garment_type: str = "other",
    limit: int | None = None,
    offset_model_path: str | None = None,
    segmentation_model_config_path: str | None = None,
    runner_v21=None,
    runner_v22=None,
) -> dict:
    if runner_v21 is None:
        runner_v21 = run_tryon
    if runner_v22 is None:
        runner_v22 = run_tryon_v2

    images = collect_images(input_dir, limit=limit)
    out = Path(output_dir)
    out_v21 = out / "v21"
    out_v22 = out / "v22"
    out_v21.mkdir(parents=True, exist_ok=True)
    out_v22.mkdir(parents=True, exist_ok=True)

    rows_v21: list[dict] = []
    rows_v22: list[dict] = []
    paired: list[dict] = []

    for image_path in images:
        req_v21 = TryOnRequest(
            image_path=str(image_path),
            garment_path=garment_path,
            output_path=str(out_v21 / image_path.name),
            garment_type=garment_type,
        )

        t0 = time.perf_counter()
        status_21 = "ok"
        err_21 = None
        meta_21: dict = {}
        try:
            r21 = runner_v21(req_v21)
            meta_21 = _extract_meta(r21)
        except Exception as exc:
            status_21 = "error"
            err_21 = str(exc)
        elapsed_21 = round((time.perf_counter() - t0) * 1000.0, 3)

        rows_v21.append(
            {
                "image": str(image_path),
                "output_path": str(out_v21 / image_path.name),
                "status": status_21,
                "error": err_21,
                "elapsed_ms": elapsed_21,
                "segmentation_backend": meta_21.get("segmentation_backend"),
                "segmentation_mask_coverage": meta_21.get("segmentation_mask_coverage"),
                "scale": (meta_21.get("transform_contract") or {}).get("scale"),
                "offset_x": (meta_21.get("transform_contract") or {}).get("offset_x", 0.0),
                "offset_y": (meta_21.get("transform_contract") or {}).get("offset_y", 0.0),
            }
        )

        req_v22 = TryOnRequest(
            image_path=str(image_path),
            garment_path=garment_path,
            output_path=str(out_v22 / image_path.name),
            garment_type=garment_type,
        )

        t1 = time.perf_counter()
        status_22 = "ok"
        err_22 = None
        meta_22: dict = {}
        try:
            r22 = runner_v22(
                req_v22,
                offset_model_path=offset_model_path,
                segmentation_model_config_path=segmentation_model_config_path,
            )
            meta_22 = _extract_meta(r22)
        except Exception as exc:
            status_22 = "error"
            err_22 = str(exc)
        elapsed_22 = round((time.perf_counter() - t1) * 1000.0, 3)

        rows_v22.append(
            {
                "image": str(image_path),
                "output_path": str(out_v22 / image_path.name),
                "status": status_22,
                "error": err_22,
                "elapsed_ms": elapsed_22,
                "segmentation_backend": meta_22.get("segmentation_backend"),
                "segmentation_mask_coverage": meta_22.get("segmentation_mask_coverage"),
                "scale": (meta_22.get("transform_contract") or {}).get("scale"),
                "offset_x": (meta_22.get("transform_contract") or {}).get("offset_x", 0.0),
                "offset_y": (meta_22.get("transform_contract") or {}).get("offset_y", 0.0),
            }
        )

        paired.append(
            {
                "image": str(image_path),
                "v21_elapsed_ms": elapsed_21,
                "v22_elapsed_ms": elapsed_22,
                "elapsed_delta_ms": round(elapsed_22 - elapsed_21, 3),
                "v21_status": status_21,
                "v22_status": status_22,
                "v21_scale": rows_v21[-1]["scale"],
                "v22_scale": rows_v22[-1]["scale"],
                "v22_offset_x": rows_v22[-1]["offset_x"],
                "v22_offset_y": rows_v22[-1]["offset_y"],
            }
        )

    stats_v21 = _build_latency_stats(rows_v21)
    stats_v22 = _build_latency_stats(rows_v22)

    summary = {
        "status": "ok",
        "input_dir": str(Path(input_dir)),
        "garment_path": str(Path(garment_path)),
        "garment_type": garment_type,
        "samples": len(images),
        "v21": stats_v21,
        "v22": stats_v22,
        "delta": {
            "avg_elapsed_ms": round(stats_v22["avg_elapsed_ms"] - stats_v21["avg_elapsed_ms"], 3),
            "success_rate": round(stats_v22["success_rate"] - stats_v21["success_rate"], 3),
        },
        "outputs": {
            "v21_dir": str(out_v21),
            "v22_dir": str(out_v22),
        },
    }

    payload = {
        "summary": summary,
        "results": {
            "v21": rows_v21,
            "v22": rows_v22,
            "paired": paired,
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark comparativo Fase 2.1 vs Fase 2.2")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--garment-path", required=True)
    parser.add_argument("--garment-type", default="other", choices=["shirt", "skirt", "pants", "dress", "other"])
    parser.add_argument("--output-dir", default="data/processed/fase2_2_benchmark")
    parser.add_argument("--report", default="data/processed/fase2_2_benchmark/compare_report.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset-model-path", default=None)
    parser.add_argument("--segmentation-model-config-path", default=None)
    args = parser.parse_args()

    payload = benchmark_compare(
        input_dir=args.input_dir,
        garment_path=args.garment_path,
        output_dir=args.output_dir,
        garment_type=args.garment_type,
        limit=args.limit,
        offset_model_path=args.offset_model_path,
        segmentation_model_config_path=args.segmentation_model_config_path,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
