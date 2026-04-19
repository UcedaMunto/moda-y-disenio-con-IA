from fase2_1.scripts.benchmark_latency import build_latency_summary, percentile


def test_percentile_basic() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 25.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0


def test_build_latency_summary() -> None:
    batch_payload = {
        "results": [
            {"status": "ok", "elapsed_ms": 1000.0},
            {"status": "ok", "elapsed_ms": 2000.0},
            {"status": "error", "elapsed_ms": 3000.0},
        ]
    }
    summary = build_latency_summary(batch_payload, max_elapsed_ms=2500.0)

    assert summary["total"] == 3
    assert summary["ok"] == 2
    assert summary["errors"] == 1
    assert summary["success_rate"] == 66.67
    assert summary["p95_elapsed_ms"] > 2500.0
    assert summary["meets_target"] is False
