from __future__ import annotations

from torchsky_harness.benchmarks.bench_native_likelihood import run_case


def test_native_likelihood_gate_recovers_known_state_and_rejects_stale_evidence() -> None:
    row = run_case(runs=1)

    assert row["status"] == "OK"
    assert row["max_standardized_error"] <= 3.0
    assert row["stale_evidence_rejected"] is True
