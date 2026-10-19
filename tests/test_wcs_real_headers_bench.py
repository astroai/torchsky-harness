"""Contract for the real-data WCS parity benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks" / "fixtures" / "real_wcs"
BENCHMARK = ROOT / "benchmarks" / "bench_wcs_real_headers.py"


def test_fixture_provenance_lists_every_vendored_file() -> None:
    provenance = (FIXTURES / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "BSD-3-Clause" in provenance
    assert "astropy" in provenance.lower()
    for path in FIXTURES.iterdir():
        if path.name == "PROVENANCE.md":
            continue
        assert path.name in provenance, f"{path.name} is not described in PROVENANCE.md"


def test_real_header_benchmark_declares_claim_rows() -> None:
    pytest.importorskip("astropy")
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_wcs_real_headers", BENCHMARK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from torchsky_harness.evidence import CLAIMS

    required = set(CLAIMS["bench_wcs_real_headers"].required_rows)
    declared = set(module.FIXTURE_SPECS)
    assert required <= declared
    assert len(required) >= 10
