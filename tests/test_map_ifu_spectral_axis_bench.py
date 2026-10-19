from __future__ import annotations

from torchsky_harness.benchmarks.bench_map_ifu_spectral_axis import run_case


def test_ifu_spectral_axis_gate_rows() -> None:
    rows = {row["case"]: row for row in run_case()}

    assert set(rows) == {
        "ifu_nonuniform_axis_line_localization",
        "ifu_nonuniform_axis_constant_spectrum",
        "ifu_regular_axis_spectrum_interpolation",
        "ifu_nonuniform_axis_interpolation_envelope",
    }
    for case, row in rows.items():
        assert row["status"] == "OK", case


def test_ifu_spectral_axis_gate_exact_localization() -> None:
    rows = {row["case"]: row for row in run_case()}

    localization = rows["ifu_nonuniform_axis_line_localization"]
    assert localization["max_spurious_value"] == 0.0
    assert set(localization["kernels"]) == {"nearest", "drizzle", "linear"}
    for kernel in localization["kernels"].values():
        assert kernel["line_count"] == 7
        assert kernel["spurious_bin_count"] == 0

    constant = rows["ifu_nonuniform_axis_constant_spectrum"]
    assert constant["max_relative_error"] <= 1.0e-5

    regular = rows["ifu_regular_axis_spectrum_interpolation"]
    assert regular["max_relative_error"] <= regular["declared_tolerance"]

    envelope = rows["ifu_nonuniform_axis_interpolation_envelope"]
    assert envelope["interior_max_relative_error"] <= envelope["declared_envelope"]
