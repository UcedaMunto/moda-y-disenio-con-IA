from __future__ import annotations

import json
from pathlib import Path

from fase2_2.scripts.run_internal_demo import main


def test_run_internal_demo_generates_report(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"
    report_path = output_dir / "demo_report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_internal_demo.py",
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
            "--garment-type",
            "shirt",
        ],
    )

    main()

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["mode"] == "mock_demo"
    assert payload["summary"]["samples"] == 4
    assert payload["summary"]["v22"]["ok"] == 4
