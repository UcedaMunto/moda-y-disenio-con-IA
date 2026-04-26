#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fase2_3.config.settings import Fase23Settings
from fase2_3.core.bootstrap import build_bootstrap_plan, clone_or_update_repo
from fase2_3.core.compat import build_compatibility_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap aislado de OpenTryOn para fase2_3.")
    parser.add_argument("--clone", action="store_true", help="Clona OpenTryOn dentro de fase2_3/vendor.")
    parser.add_argument("--update", action="store_true", help="Actualiza el clon si ya existe.")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime el plan recomendado.")
    parser.add_argument("--json", action="store_true", help="Imprime plan y resultados en JSON.")
    args = parser.parse_args()

    settings = Fase23Settings.discover(PROJECT_ROOT)
    report = build_compatibility_report(project_root=PROJECT_ROOT)
    plan = build_bootstrap_plan(settings, report)
    result = None

    if args.clone or args.update:
        result = clone_or_update_repo(settings, update=args.update)

    payload = {
        "compatibility": report,
        "plan": plan,
        "result": result,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    print("Fase 2.3 OpenTryOn bootstrap")
    print(f"Destino: {plan['opentryon_root']}")
    print(f"Modo recomendado: {plan['recommended_mode']}")
    print(f"Motivo: {plan['reason']}")
    if result:
        print(f"Resultado: {result['status']} - {result['message']}")
    print("Pasos sugeridos:")
    for step in plan["next_steps"]:
        print(f"- {step}")

    if args.dry_run and not (args.clone or args.update):
        print("Dry run completado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())