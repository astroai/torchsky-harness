"""Bounded parity of ``torchsky.wcs.WCS`` against Kapteyn (WCSLIB 4.24 under ``kapteyn.wcs``).

Kapteyn runs out-of-process in its own interpreter (``scripts/oracle/wcs_oracle.py``): the
``kapteyn`` 2.3.x sources do not build on CPython 3.13 / NumPy 2, so the campaign provisions a
dedicated conda-forge environment (``run_campaign.sh``, ``TORCHSKY_HARNESS_ORACLE_KAPTEYN_PYTHON``).
Same cases and tolerances as the PyAST gate minus SIP/TPV (WCSLIB 4.24 has neither). WCSLIB is
float64 analytic projections: 1e-4 mas on the sky, 1e-6 px on the inverse.
"""

from __future__ import annotations

import numpy as np
import pytest

import _oracle
from _gate_helpers import f64, sep_arcsec, to_np
from test_wcs_pyast_parity import CASES, POLY, grid, header_for

from torchsky.wcs import WCS


@pytest.mark.parametrize("name", sorted(set(CASES) - POLY))
def test_kapteyn_matches_torchsky(name: str) -> None:
    header = header_for(name)
    x, y = grid()
    wcs = WCS(header)
    ra, dec = (to_np(v) for v in wcs.pixel_to_world(f64(x), f64(y), origin=1))
    ora = _oracle.run_wcs_oracle("kapteyn", header, x, y, ra, dec)
    sep_mas = sep_arcsec(ora["ra"], ora["dec"], ra, dec) * 1e3
    assert float(sep_mas.max()) < 1.0e-4, f"{name}: pixel_to_world vs Kapteyn max {float(sep_mas.max()):.3e} mas"
    px_t, py_t = (to_np(v) for v in wcs.world_to_pixel(f64(ra), f64(dec), origin=1))
    diff = np.hypot(ora["px"] - px_t, ora["py"] - py_t)
    assert float(diff.max()) < 1.0e-6, f"{name}: world_to_pixel vs Kapteyn max {float(diff.max()):.3e} px"
