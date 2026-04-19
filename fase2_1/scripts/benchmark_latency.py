from __future__ import annotations

import argparse
import json
from pathlib import Path

from fase2_1.core.tryon.batch import run_tryon_batch


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


def build_latency_summary(batch_payload: dict, max_elapsed_ms: float = 2500.0) -> dict:
    results = batch_payload.get("results", [])
    elapsed = [float(r.get("elapsed_ms", 0.0) or 0.0) for r in results]
    ok = [r for r in results if r.get("status") == "ok"]

    summary = {
        "total": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "success_rate": round((len(ok) / len(results)) * 100.0, 2) if results else 0.0,
        "avg_elapsed_ms": round(sum(elapsed) / len(elapsed), 3) if elapsed else 0.0,
        "p50_elapsed_ms": percentile(elapsed, 50),
        "p90_elapsed_ms": percentile(elapsed, 90),
        "p95_elapsed_ms": percentile(elapsed, 95),
        "max_elapsed_ms": round(max(elapsed), 3) if elapsed else 0.0,
        "min_elapsed_ms": round(min(elapsed), 3) if elapsed else 0.0,
        "target_max_elapsed_ms": max_elapsed_ms,
    }
    summary["meets_target"] = summary["p95_elapsed_ms"] <= max_elapsed_ms
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de latencia Fase 2.1 (CPU baseline)")
    parser.add_argument("--input-dir", required=True, help="Directorio de imagenes de persona")
    parser.add_argument("--garment-path", required=True, help="Ruta de prenda")
    parser.add_argument(
        "--garment-type",
        default="other",
        choices=["shirt", "skirt", "pants", "dress", "other"],
        help="Categoria de prenda para ajuste por tipo",
    )
    parser.add_argument("--output-dir", default="data/processed/fase2_1_benchmark/outputs")
    parser.add_argument("--report", default="data/processed/fase2_1_benchmark/report.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-max-ms", type=float, default=2500.0)
    parser.add_argument(
        "--benchmark-out",
        default="data/processed/fase2_1_benchmark/latency_benchmark.json",
        help="Archivo JSON con resumen de benchmark",
    )
    args = parser.parse_args()

    batch = run_tryon_batch(
        input_dir=args.input_dir,
        garment_path=args.garment_path,
        garment_type=args.garment_type,
        output_dir=args.output_dir,
        report_path=args.report,
        limit=args.limit,
    )
    batch_payload = batch.model_dump()
    latency = build_latency_summary(batch_payload, max_elapsed_ms=args.target_max_ms)

    payload = {
        "status": "ok",
        "batch_report_path": batch_payload.get("report_path"),
        "latency": latency,
        "batch_summary": batch_payload.get("summary", {}),
    }

    out = Path(args.benchmark_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
