"""Bounded parity of ``torchsky.wcs.WCS`` against PyAST (starlink-pyast, the AST library).

AST reads the FITS-WCS headers from card strings (``FitsChan``) in an out-of-process oracle
(``scripts/oracle/wcs_oracle.py``; no FITS I/O library on either side). Both directions are
compared on a pixel grid inside each projection's valid domain, FITS 1-based pixels
(``origin=1``). AST and torchsky both evaluate in float64 with analytic projections, so the
bound is a tiny fraction of a milliarcsecond: 1e-4 mas (0.1 micro-arcsec) on the sky and 1e-6 px
for the inverse; SIP/TPV polynomials accumulate a little more rounding and use 1e-3 mas.
"""

from __future__ import annotations

import numpy as np
import pytest

import _oracle
from _gate_helpers import f64, sep_arcsec, to_np

from torchsky.wcs import WCS

N = 512


def _diag(s: float) -> tuple[float, float, float, float]:
    return (-s, 0.0, 0.0, s)


# name -> (projection code, crval, cd, extra header cards)
CASES = {
    "TAN": ("TAN", (150.1, 2.2), (-1.2e-4, 3.0e-6, 2.0e-6, 1.2e-4), {}),
    "TAN_pole": ("TAN", (10.0, 89.5), _diag(2.0e-4), {}),
    "SIN": ("SIN", (10.0, -40.0), _diag(2.0e-4), {}),
    "ARC": ("ARC", (300.0, 60.0), (-5.0e-4, 1.0e-5, -1.0e-5, 5.0e-4), {}),
    "ZEA": ("ZEA", (83.8, -5.4), _diag(3.0e-4), {}),
    "STG": ("STG", (201.3, -43.0), _diag(3.0e-4), {}),
    "CAR": ("CAR", (0.0, 0.0), _diag(1.0e-2), {}),
    "CAR_wrap": ("CAR", (359.5, 10.0), _diag(1.0e-2), {}),
    "AIT": ("AIT", (180.0, 0.0), _diag(0.2), {}),
    "MOL": ("MOL", (180.0, 0.0), _diag(0.2), {}),
    "HPX": ("HPX", (180.0, 0.0), _diag(0.1), {}),
    "CEA": ("CEA", (30.0, 0.0), _diag(0.2), {"PV2_1": 1.0}),
    "MER": ("MER", (30.0, 0.0), _diag(0.2), {}),
    "SFL": ("SFL", (30.0, 0.0), _diag(0.2), {}),
    "ZPN": ("ZPN", (10.0, 20.0), _diag(2.0e-3), {"PV2_0": 0.0, "PV2_1": 1.0, "PV2_3": 0.01}),
    "AZP": ("AZP", (10.0, 20.0), _diag(2.0e-3), {"PV2_1": 2.0, "PV2_2": 10.0}),
    "TAN_SIP": (
        "TAN-SIP",
        (150.1, 2.2),
        (-2.5e-5, 1.0e-6, 1.0e-6, 2.5e-5),
        {
            "A_ORDER": 2, "B_ORDER": 2, "A_2_0": 1.0e-6, "A_1_1": -2.0e-6, "A_0_2": 5.0e-7,
            "B_2_0": -3.0e-7, "B_1_1": 1.5e-6, "B_0_2": 2.0e-6,
        },  # fmt: skip
    ),
    "TPV": (
        "TPV",
        (150.1, 2.2),
        (-2.5e-5, 1.0e-6, 1.0e-6, 2.5e-5),
        {
            "PV1_0": 0.0, "PV1_1": 1.0, "PV1_2": 0.0, "PV1_4": 0.01, "PV1_5": -0.02, "PV1_6": 0.005,
            "PV2_0": 0.0, "PV2_1": 1.0, "PV2_2": 0.0, "PV2_4": 0.01, "PV2_5": 0.015, "PV2_6": -0.01,
        },  # fmt: skip
    ),
}
POLY = {"TAN_SIP", "TPV"}


def header_for(name: str) -> dict:
    code, crval, cd, extra = CASES[name]
    c11, c12, c21, c22 = cd
    return {
        "NAXIS": 2, "NAXIS1": N, "NAXIS2": N, "CTYPE1": f"RA---{code}", "CTYPE2": f"DEC--{code}",
        "CRPIX1": N / 2.0 + 0.5, "CRPIX2": N / 2.0 + 0.5, "CRVAL1": crval[0], "CRVAL2": crval[1],
        "CD1_1": c11, "CD1_2": c12, "CD2_1": c21, "CD2_2": c22, "RADESYS": "ICRS", "EQUINOX": 2000.0,
        **extra,
    }  # fmt: skip


def grid(n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(8.0, N - 8.0, n)
    xx, yy = np.meshgrid(g, g)
    return xx.ravel(), yy.ravel()


@pytest.mark.parametrize("name", sorted(CASES))
def test_pyast_matches_torchsky(name: str) -> None:
    header = header_for(name)
    tol_mas = 1.0e-3 if name in POLY else 1.0e-4
    tol_px = 1.0e-5 if name in POLY else 1.0e-6
    x, y = grid()
    wcs = WCS(header)
    ra, dec = (to_np(v) for v in wcs.pixel_to_world(f64(x), f64(y), origin=1))
    ora = _oracle.run_wcs_oracle("pyast", header, x, y, ra, dec)
    sep_mas = sep_arcsec(ora["ra"], ora["dec"], ra, dec) * 1e3
    assert float(sep_mas.max()) < tol_mas, f"{name}: pixel_to_world vs AST max {float(sep_mas.max()):.3e} mas"
    px_t, py_t = (to_np(v) for v in wcs.world_to_pixel(f64(ra), f64(dec), origin=1))
    diff = np.hypot(ora["px"] - px_t, ora["py"] - py_t)
    assert float(diff.max()) < tol_px, f"{name}: world_to_pixel vs AST max {float(diff.max()):.3e} px"
