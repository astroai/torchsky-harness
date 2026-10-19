from __future__ import annotations

from torchsky_harness.benchmarks.bench_native_nuisance import run_case


def test_native_nuisance_gate_matches_dense_references() -> None:
    row = run_case()

    assert row["status"] == "OK"
    assert row["profiled_max_abs_error"] <= 1.0e-10
    assert row["calibrated_max_abs_error"] <= 1.0e-10
    assert row["profiled_rank"] == 2
