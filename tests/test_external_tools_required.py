"""Presence and identity of the external binaries torchsky claims interoperability with.

The ``improc`` CANFAR image ships SWarp, SCAMP, SExtractor, PSFEx, astrometry.net,
Montage and WCSTools. With ``TORCHSKY_HARNESS_REQUIRE_TOOLS=1`` a missing binary (or an
empty astrometry index directory) fails; elsewhere the checks skip.
"""

from __future__ import annotations

import pytest

import _external_tools as xt


@pytest.mark.parametrize("tool", ("swarp", "scamp", "sextractor", "psfex", "solve-field", "mProject", "mProjExec", "mAdd", "xy2sky", "sky2xy"))
def test_tool_is_installed_and_runs(tool: str) -> None:
    path = xt.need(tool)
    flag = {"solve-field": ["--version"], "xy2sky": ["-v"], "sky2xy": ["-v"]}.get(tool, ["-v"])
    proc = xt.run([path, *flag])
    text = (proc.stdout + proc.stderr).strip()
    # Every one of these prints a banner or version line; a segfaulting/ABI-broken install does not.
    assert text, f"{tool} {flag} produced no output (rc={proc.returncode})"
    assert "error while loading shared libraries" not in text and "Segmentation" not in text


def test_astrometry_net_index_files_available() -> None:
    xt.need("solve-field")
    directory = xt.index_dir()
    if not directory.is_dir():
        pytest.skip(f"ASTROMETRY_INDEX_DIR {directory} is not a directory")
    indexes = sorted(directory.glob("index-*.fits"))
    assert indexes, f"no index-*.fits under {directory}"


def test_swarp_version_is_recorded() -> None:
    path = xt.need("swarp")
    banner = (xt.run([path, "-v"]).stdout or "").strip()
    assert "SWarp" in banner or "swarp" in banner.lower()


def test_stilts_runs_and_reports_version() -> None:
    cmd = xt.stilts_command()
    if cmd is None:
        pytest.skip("stilts not on PATH (improc: /usr/local/bin/stilts)")
    proc = xt.run([*cmd, "-version"])
    text = proc.stdout + proc.stderr
    assert "STILTS version" in text, f"unexpected stilts -version output: {text[:200]!r}"
