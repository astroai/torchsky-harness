"""Deterministic CPU settings without subject checkout path injection."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def pytest_configure(config):
    import torch
    torch.set_num_threads(1)


def pytest_runtest_teardown(item, nextitem):
    import torchfits
    torchfits.clear_file_cache()


# --- explicit release gates -------------------------------------------------
# Files below are release gates: when TORCHSKY_HARNESS_REQUIRE_GATES=1 (the CANFAR
# campaign sets it for the environments that ship the comparators), a skip caused
# by a missing comparator is a failure, not a silent pass. Locally and in the
# subject-only CI the same tests skip normally.
RELEASE_GATE_FILES = frozenset(
    {
        "test_sphere_moc.py",  # mocpy
        "test_sphere_polygons.py",  # spherical_geometry
        "test_sphere_exact_regions.py",  # spherical_geometry
        "test_healsparse_upstream.py",  # healsparse
        "test_cosmology_astropy_parity.py",  # astropy.cosmology
        "test_extinction_parity.py",  # dust_extinction
        "test_spectrophotometry_parity.py",  # speclite, synphot
        "test_redshift_units_parity.py",  # astropy.units
        "test_transforms_correctness.py",  # astropy.visualization
        "test_erfa_observed_place_parity.py",  # pyerfa
        "test_gwcs_live_parity.py",  # gwcs
        "test_wcs_optional_extras_parity.py",  # sgp4, jplephem, DE440s kernel
        "test_multiorder_interchange_parity.py",  # mhealpy, cdshealpix
        "test_specutils_parity.py",  # specutils
        "test_torch_harmonics_parity.py",  # torch-harmonics
        "test_sphere_s2fft.py",  # s2fft
    }
)
# SWarp / MontagePy gates exist but only the map-compat environment ships the
# binaries; they are required there (set the variable) and optional elsewhere.
MAP_COMPAT_GATE_FILES = frozenset(
    {
        "test_map_swarp_optional_parity.py",
        "test_map_external_tools_smoke.py",
        "test_montage_projection_parity.py",
        "test_wcstools_parity.py",
    }
)
# External-binary presence gates (SWarp, SCAMP, SExtractor, PSFEx, solve-field, Montage, WCSTools):
# required when TORCHSKY_HARNESS_REQUIRE_TOOLS=1 (IMAGE_PROFILE=improc in run_campaign.sh).
EXTERNAL_TOOL_GATE_FILES = frozenset({"test_external_tools_required.py"})


def _require_gates() -> bool:
    return os.environ.get("TORCHSKY_HARNESS_REQUIRE_GATES") == "1"


def _gated(path_name: str) -> bool:
    if path_name in RELEASE_GATE_FILES:
        return True
    tools = os.environ.get("TORCHSKY_HARNESS_REQUIRE_TOOLS") == "1"
    if path_name in EXTERNAL_TOOL_GATE_FILES:
        return tools
    return path_name in MAP_COMPAT_GATE_FILES and (
        os.environ.get("TORCHSKY_HARNESS_REQUIRE_MAP_COMPAT") == "1" or tools
    )


import pytest  # noqa: E402


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # An unconditional ``@pytest.mark.skip`` is a documented coverage gap, not a
    # missing comparator, so it never counts as a failed release gate.
    declared_skip = item.get_closest_marker("skip") is not None
    if (
        report.skipped
        and _require_gates()
        and _gated(item.path.name)
        and not declared_skip
        and not hasattr(report, "wasxfail")
    ):
        report.outcome = "failed"
        report.longrepr = f"required release gate skipped (missing comparator?): {report.longrepr}"


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    path = getattr(collector, "path", None)
    if report.outcome == "skipped" and _require_gates() and path is not None and _gated(path.name):
        report.outcome = "failed"
        report.longrepr = f"required release gate module skipped at import (missing comparator?): {report.longrepr}"
