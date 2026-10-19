from __future__ import annotations

from torchsky_harness.benchmarks.bench_native_response import run_case


def test_native_response_gate_recovers_known_star() -> None:
    row = run_case(size=32, runs=1)

    assert row["status"] == "OK"
    assert row["flux_relative_error"] < 0.05
    assert row["background_absolute_error"] < 0.2
