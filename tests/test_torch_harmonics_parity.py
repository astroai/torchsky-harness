"""Bounded parity of torchsky's native CAR spherical-harmonic transforms against ``torch-harmonics``.

``torchsky.sphere.sht`` ships a ``torch-harmonics`` backend for equiangular
(Clenshaw-Curtis) grids. Here the *native* torchsky backend is the subject and
``torch_harmonics.RealSHT`` / ``InverseRealSHT`` (called through torchsky's adapter, which owns
the packed-alm layout and phase conventions) is the reference: synthesis and quadrature analysis
of a band-limited field must agree to float64 round-off, and both must round-trip the input.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch_harmonics")

from _gate_helpers import device  # noqa: E402

from torchsky.sphere import alm_index, alm_size  # noqa: E402
from torchsky.sphere.sht import CARGeometry, analyze, synthesize  # noqa: E402


def _geometry(nlat: int, nlon: int) -> CARGeometry:
    return CARGeometry.from_header(
        {
            "NAXIS1": nlon, "NAXIS2": nlat, "CTYPE1": "RA---CAR", "CTYPE2": "DEC--CAR",
            "CUNIT1": "deg", "CUNIT2": "deg", "CRVAL1": 0.0, "CRVAL2": 0.0,
            "CRPIX1": 1.0, "CRPIX2": (nlat + 1.0) / 2.0,
            "CDELT1": 360.0 / nlon, "CDELT2": -180.0 / (nlat - 1),
        },
        spin_convention="iau",
    )  # fmt: skip


def _alm(lmax: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = 1.0 / (1.0 + torch.arange(alm_size(lmax)).double())
    alm = torch.complex(torch.randn(alm_size(lmax), dtype=torch.float64, generator=g), torch.randn(alm_size(lmax), dtype=torch.float64, generator=g)) * scale
    for ell in range(lmax + 1):
        alm[alm_index(ell, 0, lmax)] = alm[alm_index(ell, 0, lmax)].real.item()
    return alm


@pytest.mark.parametrize(("lmax", "nlat", "nlon"), ((8, 33, 64), (24, 97, 192), (48, 193, 384)))
def test_synthesis_and_analysis_match_torch_harmonics(lmax: int, nlat: int, nlon: int) -> None:
    dev = device()
    if dev.type == "mps":
        pytest.skip("float64 transforms are not available on MPS")
    geometry = _geometry(nlat, nlon)
    alm = _alm(lmax, seed=lmax).to(dev)
    native = synthesize(alm, geometry, lmax=lmax).values
    peer = synthesize(alm, geometry, lmax=lmax, backend="torch-harmonics").values
    scale = float(native.abs().max())
    assert float((native - peer).abs().max()) < 1e-10 * max(scale, 1.0), "synthesis"
    back_native = analyze(native, geometry, lmax=lmax).values
    back_peer = analyze(native, geometry, lmax=lmax, backend="torch-harmonics").values
    a = float(alm.abs().max())
    assert float((back_peer - back_native).abs().max()) < 1e-10 * a, "analysis native vs peer"
    # exact quadrature on the band-limited field: the input is recovered at a lmax << nlat grid
    assert float((back_peer - alm).abs().max()) < 1e-9 * a, "peer round trip"
    np.testing.assert_allclose(back_native.cpu().numpy(), alm.cpu().numpy(), rtol=0, atol=1e-9 * a)
