from __future__ import annotations

import sys

import torchsky_harness.benchmarks.bench_all as bench_all


def test_bench_all_returns_nonzero_when_a_domain_fails(
    monkeypatch, tmp_path
) -> None:
    def fail(**_kwargs):
        raise RuntimeError("forced domain failure")

    monkeypatch.setattr(bench_all, "run_wcs_domain", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_all",
            "--wcs-only",
            "--quick",
            "--no-legacy-wcs-bridge",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "forced-failure",
        ],
    )

    assert bench_all.main() == 1
    assert (tmp_path / "forced-failure" / "results.csv").exists()
