"""Contract for the real-sky sphere parity benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "real_sphere" / "cfht_megapipe_g_nside512.npz"
BENCHMARK = ROOT / "benchmarks" / "bench_sphere_real_map.py"


def test_real_sphere_fixture_and_provenance() -> None:
    import numpy as np

    provenance = FIXTURE.with_name("PROVENANCE.md").read_text(encoding="utf-8")
    assert "CFHT" in provenance
    assert "not redistributed" in provenance
    data = np.load(FIXTURE)
    assert int(data["nside"]) == 512
    assert data["pix"].dtype.kind == "i"
    assert data["pix"].size == data["value"].size
    assert data["pix"].size >= 50
    assert data["sample_ra"].size >= 1000


def test_real_sphere_benchmark_declares_claim_rows() -> None:
    pytest.importorskip("healpy")
    import importlib.util

    from torchsky_harness.evidence import CLAIMS

    spec = importlib.util.spec_from_file_location("bench_sphere_real_map", BENCHMARK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = set(CLAIMS["bench_sphere_real_map"].required_rows)
    source = BENCHMARK.read_text(encoding="utf-8")
    for name in required:
        if name.startswith("ud_grade_"):
            # Emitted as ``f"ud_grade_{nside}_{nside_out}"``: check the template
            # plus the fixture nside and the loop over output nsides.
            _, _, nside_in, nside_out = name.split("_")
            assert 'f"ud_grade_{nside}_{nside_out}"' in source
            assert f"nside{nside_in}" in FIXTURE.name
            assert str(nside_out) in source.split("for nside_out in (", 1)[1].split(")", 1)[0]
        else:
            assert name in source
