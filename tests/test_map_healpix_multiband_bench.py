from __future__ import annotations

import pytest

from torchsky_harness.benchmarks.bench_map_healpix_multiband import run_case


@pytest.fixture(scope="module")
def rows() -> dict[str, dict]:
    return {row["case"]: row for row in run_case()}


def test_healpix_multiband_gate_passes(rows: dict[str, dict]) -> None:
    assert set(rows) == {
        "healpix_multiband_flat_multiscale",
        "healpix_multiband_sed_registration",
        "healpix_multiband_plane_photometry",
        "healpix_multiband_roundtrip_cube",
    }
    for case, row in rows.items():
        assert row["status"] == "OK", case


def test_healpix_multiband_gate_reports_exact_recovery(rows: dict[str, dict]) -> None:
    flat = rows["healpix_multiband_flat_multiscale"]
    assert flat["constant_recovery_max_rel_error"] <= 1.0e-5
    assert flat["coverage_area_max_rel_error"] <= 0.05
    assert flat["common_pixel_fraction"] >= 0.8

    sed = rows["healpix_multiband_sed_registration"]
    assert sed["max_residual_vs_bilinear_max_rel_error"] <= 1.0e-5
    assert sed["source_position_max_offset_arcsec"] <= 1.0
    assert sed["band_position_max_separation_arcsec"] <= 0.5

    plane = rows["healpix_multiband_plane_photometry"]
    assert plane["plane_max_rel_residual"] <= 1.0e-5
    assert plane["band_ratio_max_rel_error"] <= 1.0e-5
    assert plane["band_flux_ratio_max_rel_error"] <= 1.0e-5

    roundtrip = rows["healpix_multiband_roundtrip_cube"]
    assert roundtrip["stacked_cube_shape"] == [3, 48, 48]
    assert roundtrip["roundtrip_max_rel_error"] <= 1.0e-5
