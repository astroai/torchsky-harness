"""Out-of-process comparator oracles (PyAST, Kapteyn) with NPZ exchange.

The legacy comparators run in a separate interpreter (``scripts/oracle/wcs_oracle.py``) so they
need not share torchsky's Python/numpy. Interpreter lookup, per engine:

1. ``TORCHSKY_HARNESS_ORACLE_<ENGINE>_PYTHON`` (set by ``scripts/canfar/run_campaign.sh``);
2. the current interpreter, when the engine module imports there (local ``pip --target`` checks).

``TORCHSKY_HARNESS_REQUIRE_GATES=1`` turns an unavailable oracle into a failure through the
conftest skip-to-failure rule, so these helpers simply ``pytest.skip``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

WORKER = Path(__file__).resolve().parents[1] / "scripts" / "oracle" / "wcs_oracle.py"
_MODULE = {"pyast": "starlink.Ast", "kapteyn": "kapteyn"}


def oracle_python(engine: str) -> str | None:
    explicit = os.environ.get(f"TORCHSKY_HARNESS_ORACLE_{engine.upper()}_PYTHON")
    if explicit:
        return explicit if Path(explicit).exists() else None
    try:
        found = importlib.util.find_spec(_MODULE[engine]) is not None
    except ModuleNotFoundError:
        found = False
    return sys.executable if found else None


def need_oracle(engine: str) -> str:
    py = oracle_python(engine)
    if py is None:
        pytest.skip(f"{engine} oracle interpreter not available")
    return py


def fits_cards(header: dict) -> list[str]:
    """Render a plain header dict to 80-character FITS cards (no FITS I/O library)."""
    cards = []
    for key, value in header.items():
        if key in {"", "COMMENT", "HISTORY", "END"} or str(key).endswith("_COMMENT"):
            continue
        if isinstance(value, bool):
            text = "T" if value else "F"
            body = f"{text:>20}"
        elif isinstance(value, str):
            body = "'" + value.replace("'", "''").ljust(8) + "'"
            body = body.ljust(20)
        elif isinstance(value, int):
            body = f"{value:>20d}"
        else:
            body = f"{float(value):>20.15G}"
        key = str(key)
        cards.append((f"HIERARCH {key} = " if len(key) > 8 else f"{key:<8}= ") + body)
    cards.append("END")
    return [c[:80].ljust(80) for c in cards]


def run_wcs_oracle(engine: str, header: dict, x, y, lon, lat, repeat: int = 1) -> dict:
    """Pixel->world at (x, y) [FITS 1-based] and world->pixel at (lon, lat) [deg] in the oracle."""
    py = need_oracle(engine)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(
            src,
            cards=np.array(fits_cards(header), dtype=object),
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            lon=np.asarray(lon, dtype=np.float64),
            lat=np.asarray(lat, dtype=np.float64),
            repeat=np.int64(repeat),
        )
        done = subprocess.run(
            [py, str(WORKER), "--engine", engine, "--in", str(src), "--out", str(dst)],
            capture_output=True,
            text=True,
            timeout=1800,
            env=dict(os.environ, PYTHONNOUSERSITE=os.environ.get("PYTHONNOUSERSITE", "1")),
        )
        if done.returncode != 0:
            raise RuntimeError(f"{engine} oracle failed:\n{done.stderr[-2000:]}")
        out = np.load(dst, allow_pickle=True)
        return {k: (out[k].item() if out[k].ndim == 0 else out[k]) for k in out.files}


DRIZZLE_WORKER = WORKER.with_name("drizzle_oracle.py")


def run_drizzle_oracle(src_header: dict, dst_header: dict, image, weight, kernel: str, pixfrac: float) -> dict:
    """AstroDrizzle (drizzlepac ``do_driz``/``cdriz.tdriz``) of one image onto ``dst_header``."""
    try:
        have = importlib.util.find_spec("drizzlepac") is not None
    except ModuleNotFoundError:
        have = False
    py = os.environ.get("TORCHSKY_HARNESS_ORACLE_DRIZZLEPAC_PYTHON") or (sys.executable if have else None)
    if py is None or not Path(py).exists():
        pytest.skip("drizzlepac oracle interpreter not available")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(
            src,
            src_cards=np.array(fits_cards(src_header), dtype=object),
            dst_cards=np.array(fits_cards(dst_header), dtype=object),
            image=np.asarray(image, dtype=np.float32),
            weight=np.asarray(weight, dtype=np.float32),
            kernel=kernel,
            pixfrac=float(pixfrac),
        )
        done = subprocess.run(
            [py, str(DRIZZLE_WORKER), "--in", str(src), "--out", str(dst)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if done.returncode != 0:
            raise RuntimeError(f"drizzlepac oracle failed:\n{done.stderr[-2000:]}")
        out = np.load(dst, allow_pickle=True)
        return {k: (out[k].item() if out[k].ndim == 0 else out[k]) for k in out.files}
