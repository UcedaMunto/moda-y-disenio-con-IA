#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fase2_3.core.compat import build_compatibility_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight de compatibilidad AMD/ROCm para OpenTryOn.")
    parser.add_argument("--json", action="store_true", help="Imprime la salida completa en JSON.")
    args = parser.parse_args()

    report = build_compatibility_report(project_root=PROJECT_ROOT)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0

    print("Fase 2.3 AMD preflight")
    print(f"Host: {report['host']['platform']}")
    print(f"AMD detectada: {report['gpu']['amd_detected']}")
    print(f"ROCm runtime: {report['rocm']['runtime_ready']}")
    print(f"PyTorch instalado: {report['torch']['installed']}")
    print(f"HIP/ROCm en torch: {bool(report['torch']['hip_version'])}")
    print(f"GPU disponible en torch: {report['torch']['cuda_available']}")
    print(f"Modo recomendado: {report['verdict']['recommended_mode']}")
    print(f"Motivo: {report['verdict']['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())