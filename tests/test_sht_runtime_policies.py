from __future__ import annotations

import pytest

from torchsky_harness.benchmarks import bench_sht_runtime_policies as benchmark


def test_parse_cases_supports_explicit_matrix() -> None:
    cases = benchmark._parse_cases("small:8:6:3:1,large:64:48:48:2")

    assert [case.case_id for case in cases] == [
        "small_n8_l6_m3_s1",
        "large_n64_l48_m48_s2",
    ]


def test_parse_cases_rejects_invalid_mmax() -> None:
    with pytest.raises(ValueError, match="invalid case"):
        benchmark._parse_cases("small:8:6:7:1")


def test_summarize_comparisons_reports_hybrid_speedup() -> None:
    def row(policy: str, analysis: float, synthesis: float, rss: int) -> dict:
        return {
            "case_id": "small_n8_l6_m6_s2",
            "policy": policy,
            "threads": {"requested": 4},
            "fft": {
                "analysis": {"median_s": analysis},
                "synthesis": {"median_s": synthesis},
                "peak_rss_after_timing_bytes": rss,
            },
        }

    summary = benchmark._summarize_comparisons(
        [row("hybrid", 1.0, 4.0, 100), row("aten", 2.0, 2.0, 120)]
    )

    comparison = summary["rows"][0]
    assert comparison["analysis_hybrid_over_aten_speedup"] == 2.0
    assert comparison["synthesis_hybrid_over_aten_speedup"] == 0.5
    assert comparison["hybrid_over_aten_peak_rss_ratio"] == pytest.approx(5 / 6)
    assert summary["hybrid_regressions_over_5pct"] == {
        "analysis": 0,
        "synthesis": 1,
    }
