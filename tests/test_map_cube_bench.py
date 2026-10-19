from __future__ import annotations

from torchsky_harness.benchmarks.bench_map_cube import run_case


def test_cube_gate_recovers_synthetic_truth() -> None:
    row = run_case(size=8, bands=3, runs=1)

    assert row["status"] == "OK"
    assert row["reconstruction_rmse"] < 1.0e-5
    assert row["band_flux_relative_error_max"] < 1.0e-3
    assert row["total_flux_relative_error"] < 1.0e-3
