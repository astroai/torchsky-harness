from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchsky.wcs import (
    WCS,
)


def _make_tan_wcs(
    crval: tuple[float, float] = (150.0, 2.0),
    cdelt: float = 1.0 / 3600.0,
    shape: tuple[int, int] = (32, 32),
) -> WCS:
    ny, nx = shape
    return WCS(
        {
            "NAXIS": 2,
            "NAXIS1": nx,
            "NAXIS2": ny,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": nx / 2.0 + 0.5,
            "CRPIX2": ny / 2.0 + 0.5,
            "CRVAL1": crval[0],
            "CRVAL2": crval[1],
            "CD1_1": -cdelt,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": cdelt,
        }
    )




def test_wcs_geometry_helpers_and_astropy_parity() -> None:
    astropy_wcs = pytest.importorskip("astropy.wcs")
    scale = 0.5 / 3600.0
    angle = math.radians(25.0)
    header = {
        "NAXIS": 2,
        "NAXIS1": 40,
        "NAXIS2": 30,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 20.5,
        "CRPIX2": 15.5,
        "CRVAL1": 180.0,
        "CRVAL2": 10.0,
        "CD1_1": -scale * math.cos(angle),
        "CD1_2": -scale * math.sin(angle),
        "CD2_1": -scale * math.sin(angle),
        "CD2_2": scale * math.cos(angle),
    }
    w = WCS(header)
    aw = astropy_wcs.WCS(header)

    sx, sy = w.pixel_scales()
    assert math.isclose(sx, scale, rel_tol=1e-10)
    assert math.isclose(sy, scale, rel_tol=1e-10)
    assert math.isclose(w.pixel_area(), scale * scale, rel_tol=1e-10)
    assert math.isclose(w.position_angle(), 25.0, abs_tol=1e-8)

    fp = w.footprint(center=True)
    afp = aw.calc_footprint(center=True)
    assert fp.shape == (4, 2)
    np.testing.assert_allclose(fp.cpu().numpy(), afp, rtol=0, atol=1e-8)

    pam_deg2 = w.pixel_area_map(unit="deg2")
    pam_arcsec2 = w.pixel_area_map(unit="arcsec2")
    assert pam_deg2.shape == (30, 40)
    assert torch.allclose(pam_arcsec2, torch.full_like(pam_arcsec2, 0.25), rtol=1e-4)
