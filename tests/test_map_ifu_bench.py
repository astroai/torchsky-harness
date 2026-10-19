from __future__ import annotations

from torchsky_harness.benchmarks.bench_map_ifu import run_case


def test_ifu_gate_recovers_matched_sparse_cube() -> None:
    row = run_case(size=12)

    assert row["status"] == "OK"
    assert row["reconstruction_max_abs_err"] == 0.0
    assert 0.5 / 12.0 <= row["valid_fraction"] <= 2.0 / 12.0
