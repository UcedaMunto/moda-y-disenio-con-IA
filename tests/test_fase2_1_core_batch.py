from pathlib import Path

from fase2_1.core.tryon import batch as batch_module


def test_run_tryon_batch_writes_report(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "a.jpg").write_bytes(b"a")
    (input_dir / "b.jpg").write_bytes(b"b")

    output_dir = tmp_path / "outputs"
    report_path = tmp_path / "report.json"

    class _FakeResult:
        def __init__(self, output_path: str):
            self.status = "ok"
            self.output_path = output_path

    def fake_run_tryon(req):
        out = Path(req.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"ok")
        return _FakeResult(str(out))

    monkeypatch.setattr(batch_module, "run_tryon", fake_run_tryon)

    result = batch_module.run_tryon_batch(
        input_dir=str(input_dir),
        garment_path="data/raw/models/TShirts.obj",
        output_dir=str(output_dir),
        report_path=str(report_path),
        limit=None,
    )

    assert result.status == "ok"
    assert result.summary["total"] == 2
    assert result.summary["ok"] == 2
    assert result.summary["errors"] == 0
    assert "avg_auto_quality_score" in result.summary
    assert "checklist_version" in result.summary
    assert all("auto_quality_score" in row for row in result.results)
    assert all("manual_review_required" in row for row in result.results)
    assert report_path.exists()


def test_run_tryon_batch_missing_input_dir(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    output_dir = tmp_path / "outputs"
    report_path = tmp_path / "report.json"

    try:
        batch_module.run_tryon_batch(
            input_dir=str(missing),
            garment_path="data/raw/models/TShirts.obj",
            output_dir=str(output_dir),
            report_path=str(report_path),
            limit=None,
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True
