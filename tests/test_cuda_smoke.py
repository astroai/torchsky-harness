"""CUDA device smoke tests for the installed torchsky subject.

The CANFAR CUDA job runs these with ``-k cuda`` on an H100 host. Locally the
whole module skips when CUDA is unavailable, so the bounded suite stays green
on CPU-only machines.
"""

from __future__ import annotations

import pytest
import torch

from torchsky import WCS
from torchsky.sphere import healpix as hp
from torchsky.sphere import alm2map, map2alm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)

_DEVICE = "cuda"
_HEADER = {
    "NAXIS": 2,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 512.0,
    "CRPIX2": 512.0,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CD1_1": -0.2,
    "CD1_2": 0.0,
    "CD2_1": 0.0,
    "CD2_2": 0.2,
}


def test_cuda_wcs_roundtrip_matches_cpu() -> None:
    x = torch.linspace(-1500.0, 1500.0, 2048, dtype=torch.float64)
    y = torch.linspace(-1500.0, 1500.0, 2048, dtype=torch.float64)
    cpu = WCS(_HEADER)
    gpu = WCS(_HEADER).to(_DEVICE)
    ra_cpu, dec_cpu = cpu.pixel_to_world(x, y)
    ra_gpu, dec_gpu = gpu.pixel_to_world(x.to(_DEVICE), y.to(_DEVICE))
    assert ra_gpu.device.type == "cuda"
    # World coordinates agree to well below one micro-arcsecond.
    assert float((ra_gpu.cpu() - ra_cpu).abs().max()) < 1e-6
    assert float((dec_gpu.cpu() - dec_cpu).abs().max()) < 1e-6

    x_cpu, y_cpu = cpu.world_to_pixel(ra_cpu, dec_cpu)
    x_gpu, y_gpu = gpu.world_to_pixel(ra_gpu, dec_gpu)
    assert float((x_gpu.cpu() - x_cpu).abs().max()) < 1e-4
    assert float((y_gpu.cpu() - y_cpu).abs().max()) < 1e-4


def test_cuda_healpix_lookup_matches_cpu() -> None:
    nside = 64
    pix = torch.arange(12 * nside * nside, dtype=torch.int64)
    ra_cpu, dec_cpu = hp.pix2ang_ring(nside, pix)
    ra_gpu, dec_gpu = hp.pix2ang_ring(nside, pix.to(_DEVICE))
    assert ra_gpu.device.type == "cuda"
    # Device kernels may differ in the last few ulps; compare angles with a
    # sub-nanoarcsecond tolerance instead of exact equality.
    torch.testing.assert_close(ra_gpu.cpu(), ra_cpu, atol=1e-10, rtol=0)
    torch.testing.assert_close(dec_gpu.cpu(), dec_cpu, atol=1e-10, rtol=0)

    back = hp.ang2pix_ring(nside, ra_gpu, dec_gpu)
    matches = float((back.cpu() == pix).to(torch.float64).mean())
    # Pixel centres round-trip exactly except for a handful that sit exactly on
    # an angular boundary where the last-ulp difference can flip the face.
    assert matches > 0.999


def test_cuda_sht_smoke() -> None:
    nside, lmax = 16, 8
    values = torch.ones(12 * nside * nside, dtype=torch.float64)
    values += 0.1 * torch.arange(values.numel(), dtype=torch.float64)
    alm_cpu = map2alm(values, nside=nside, lmax=lmax)
    alm_gpu = map2alm(values.to(_DEVICE), nside=nside, lmax=lmax)
    assert alm_gpu.device.type == "cuda"
    assert torch.isfinite(alm_gpu.real).all()
    assert torch.isfinite(alm_gpu.imag).all()
    rel = float((alm_gpu.cpu() - alm_cpu).abs().max()) / max(
        float(alm_cpu.abs().max()), 1e-30
    )
    assert rel < 1e-6

    back = alm2map(alm_gpu, nside, lmax=lmax)
    assert back.device.type == "cuda"
    assert torch.isfinite(back).all()
