"""Bounded parity of ``torchsky.wcs.WCS`` against WCSTools ``xy2sky`` / ``sky2xy``.

An independent C implementation of the FITS celestial projections (Doug Mink's WCSTools)
is run on small FITS images written with ``torchfits``; torchsky evaluates the same header.
WCSTools prints pixels to 3 decimals and degrees with ``-n 9`` digits, which sets the
tolerances: 2e-3 px and 2e-7 deg (0.7 mas). Pixel convention: FITS 1-based for the tools,
torchsky ``origin=1``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchfits

import _external_tools as xt
from _gate_helpers import f64, sep_arcsec, to_np

from torchsky.wcs import WCS

CASES = {
    "TAN": dict(ctype=("RA---TAN", "DEC--TAN"), crval=(150.1, 2.2), cd=(-1.2e-4, 3.0e-6, 2.0e-6, 1.2e-4)),
    "SIN": dict(ctype=("RA---SIN", "DEC--SIN"), crval=(10.0, -40.0), cd=(-2.0e-4, 0.0, 0.0, 2.0e-4)),
    "ARC": dict(ctype=("RA---ARC", "DEC--ARC"), crval=(300.0, 60.0), cd=(-5.0e-4, 1.0e-5, -1.0e-5, 5.0e-4)),
    "ZEA": dict(ctype=("RA---ZEA", "DEC--ZEA"), crval=(83.8, -5.4), cd=(-3.0e-4, 0.0, 0.0, 3.0e-4)),
    "STG": dict(ctype=("RA---STG", "DEC--STG"), crval=(201.3, -43.0), cd=(-3.0e-4, 0.0, 0.0, 3.0e-4)),
    "CAR": dict(ctype=("RA---CAR", "DEC--CAR"), crval=(0.0, 0.0), cd=(-1.0e-2, 0.0, 0.0, 1.0e-2)),
}
N = 512


def _header(case: dict) -> dict:
    c11, c12, c21, c22 = case["cd"]
    return {
        "NAXIS": 2, "NAXIS1": N, "NAXIS2": N,
        "CTYPE1": case["ctype"][0], "CTYPE2": case["ctype"][1],
        "CRPIX1": N / 2.0 + 0.5, "CRPIX2": N / 2.0 + 0.5,
        "CRVAL1": case["crval"][0], "CRVAL2": case["crval"][1],
        "CD1_1": c11, "CD1_2": c12, "CD2_1": c21, "CD2_2": c22,
        "RADESYS": "ICRS", "EQUINOX": 2000.0,
    }  # fmt: skip


def _write(tmp: Path, header: dict) -> Path:
    path = tmp / "wcs.fits"
    torchfits.write(str(path), torch.zeros((N, N), dtype=torch.float32), header=header, overwrite=True)
    return path


def _points(seed: int = 0, n: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(5, N - 5, n), rng.uniform(5, N - 5, n)])


@pytest.mark.parametrize("name", sorted(CASES))
def test_xy2sky_matches_torchsky(name: str, tmp_path: Path) -> None:
    exe = xt.need("xy2sky")
    header = _header(CASES[name])
    path = _write(tmp_path, header)
    xy = _points()
    wcs = WCS(header)
    ra, dec = (to_np(v) for v in wcs.pixel_to_world(f64(xy[:, 0]), f64(xy[:, 1]), origin=1))
    for i, (x, y) in enumerate(xy):
        out = xt.run([exe, "-d", "-n", "9", str(path), f"{x:.6f}", f"{y:.6f}"]).stdout
        vals = [float(v) for v in xt.FLOAT.findall(out)]
        assert len(vals) >= 2, f"unparsable xy2sky output: {out!r}"
        sep = sep_arcsec(vals[0], vals[1], ra[i], dec[i])
        assert float(sep) < 1.0e-3, f"{name}: xy2sky vs torchsky {float(sep) * 1e3:.3f} mas at ({x:.2f},{y:.2f})"


@pytest.mark.parametrize("name", sorted(CASES))
def test_sky2xy_matches_torchsky(name: str, tmp_path: Path) -> None:
    exe = xt.need("sky2xy")
    header = _header(CASES[name])
    path = _write(tmp_path, header)
    xy = _points(seed=1)
    wcs = WCS(header)
    ra, dec = (to_np(v) for v in wcs.pixel_to_world(f64(xy[:, 0]), f64(xy[:, 1]), origin=1))
    for i in range(len(xy)):
        # sky2xy takes decimal degrees with an explicit frame; -n sets output decimals.
        out = xt.run([exe, "-n", "6", str(path), f"{ra[i]:.9f}", f"{dec[i]:.9f}", "ICRS"]).stdout
        tail = out.split("->")[-1] if "->" in out else out
        vals = [float(v) for v in xt.FLOAT.findall(tail)]
        assert len(vals) >= 2, f"unparsable sky2xy output: {out!r}"
        assert abs(vals[-2] - xy[i, 0]) < 2.0e-3 and abs(vals[-1] - xy[i, 1]) < 2.0e-3, (
            f"{name}: sky2xy {vals[-2:]} vs input {xy[i].tolist()}"
        )
