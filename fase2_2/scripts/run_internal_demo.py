from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fase2_2.scripts.benchmark_compare_v21_v22 import benchmark_compare


class _MockResult:
    def __init__(self, scale: float, offset_x: float = 0.0, offset_y: float = 0.0):
        self._payload = {
            "status": "ok",
            "meta": {
                "segmentation_backend": "mock",
                "segmentation_mask_coverage": 33.0,
                "transform_contract": {
                    "scale": scale,
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                },
            },
        }

    def model_dump(self):
        return self._payload


def _runner_v21(req):
    return _MockResult(scale=1.05)


def _runner_v22(req, offset_model_path=None, segmentation_model_config_path=None):
    return _MockResult(scale=1.08, offset_x=0.07, offset_y=-0.02)


def _make_demo_inputs(base_dir: Path) -> tuple[str, str]:
    input_dir = base_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    for i, tone in enumerate([80, 110, 140, 170], start=1):
        arr = np.full((96, 72, 3), tone, dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(input_dir / f"person_{i:02d}.jpg")

    garment_path = base_dir / "garment.png"
    g = np.zeros((64, 48, 4), dtype=np.uint8)
    g[..., 0] = 220
    g[..., 1] = 160
    g[..., 2] = 80
    g[..., 3] = 220
    Image.fromarray(g, mode="RGBA").save(garment_path)

    return str(input_dir), str(garment_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo interna Fase 2.2 (modo mock reproducible)")
    parser.add_argument("--output-dir", default="data/processed/fase2_2_demo")
    parser.add_argument("--report", default="data/processed/fase2_2_demo/demo_report.json")
    parser.add_argument("--garment-type", default="shirt", choices=["shirt", "skirt", "pants", "dress", "other"])
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_dir, garment_path = _make_demo_inputs(out)

    payload = benchmark_compare(
        input_dir=input_dir,
        garment_path=garment_path,
        output_dir=str(out / "compare_outputs"),
        garment_type=args.garment_type,
        runner_v21=_runner_v21,
        runner_v22=_runner_v22,
    )
    payload["summary"]["mode"] = "mock_demo"
    payload["summary"]["note"] = "Demo interna reproducible sin dependencias de vision"

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
