"""Retained legacy facade timing gate; FITS performance ownership is torchfits."""

from __future__ import annotations

import os
from pathlib import Path
import statistics
import time

import pytest
import torchsky
import torchsky_harness

pytestmark = pytest.mark.performance

_FIXTURE = Path(torchsky_harness.__file__).parent / "fixtures" / "radio_mm_stub.fits"


def _median_seconds(fn, *, warmup: int = 1, iters: int = 3) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples))


def test_gate_fits_image_read_stub() -> None:
    assert _FIXTURE.is_file(), f"missing {_FIXTURE}"

    def run() -> None:
        torchsky.read(str(_FIXTURE), return_header=True)

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_READ_FITS_S", "3.0"))
    assert _median_seconds(run) < max_s
