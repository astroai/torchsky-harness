"""Contract for the real-image map parity benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "real_map" / "cfht_g_cutout_256.npz"
BENCHMARK = ROOT / "benchmarks" / "bench_map_real_cutout.py"


def test_real_map_fixture_and_provenance() -> None:
    import numpy as np

    provenance = FIXTURE.with_name("PROVENANCE.md").read_text(encoding="utf-8")
    assert "CFHT" in provenance
    assert "not redistributed" in provenance
    data = np.load(FIXTURE)
    assert data["image"].shape == (256, 256)
    assert data["image"].dtype == np.float32
    assert "CTYPE1" in str(data["header_json"])


def test_real_map_benchmark_declares_claim_row() -> None:
    pytest.importorskip("reproject")
    from torchsky_harness.evidence import CLAIMS

    source = BENCHMARK.read_text(encoding="utf-8")
    for name in CLAIMS["bench_map_real_cutout"].required_rows:
        assert name in source
