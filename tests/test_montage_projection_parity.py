"""Bounded parity of torchsky's conservative polygon resampler against Montage ``mProject``.

Montage reprojects with exact spherical-polygon overlap (flux-conserving); torchsky's
``conservative_polygon`` kernel computes the same overlap areas. A smooth blob is placed on a
rotated, offset input grid and both projections onto the same TAN target are compared in the
interior. Needs the ``mProject`` binary (``improc`` image). The shared tolerance (1% of the
peak) covers float32 storage and Montage's area-weight normalisation at the footprint edge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchfits

import _external_tools as xt

from torchsky.map import coadd as coadd_image_plane
from torchsky.scene import ImageObservation


def _blob(n: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    return torch.exp(-((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * 6.0**2)) + 0.2


def test_mproject_matches_conservative_polygon(tmp_path: Path) -> None:
    xt.need("mProject")
    import torchsky_harness.benchmarks.bench_map_compat as bmc

    n = 64
    target = {
        "NAXIS": 2, "NAXIS1": n, "NAXIS2": n, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": n / 2 + 0.5, "CRPIX2": n / 2 + 0.5, "CRVAL1": 200.0, "CRVAL2": 15.0,
        "CDELT1": -0.01, "CDELT2": 0.01,
    }  # fmt: skip
    theta = np.radians(7.0)
    scale = 0.0107  # slightly different pixel scale and a rotation
    src = dict(target)
    src.update(
        CRVAL1=200.0 + 0.013, CRVAL2=15.0 - 0.009, CRPIX1=n / 2 + 0.5, CRPIX2=n / 2 + 0.5,
        CD1_1=-scale * np.cos(theta), CD1_2=scale * np.sin(theta), CD2_1=scale * np.sin(theta), CD2_2=scale * np.cos(theta),
    )  # fmt: skip
    for key in ("CDELT1", "CDELT2"):
        src.pop(key)
    path = tmp_path / "src.fits"
    torchfits.write(str(path), _blob(n), header=src, overwrite=True)
    obs = [ImageObservation(str(path))]
    ours = coadd_image_plane(obs, target=target, kernel="conservative_polygon")
    data, weight, mode, _ = bmc._coadd_with_montage(obs, target_header=target)
    inner = (slice(8, n - 8), slice(8, n - 8))
    mask = (ours.weight[inner].cpu().numpy() > 0) & (np.asarray(weight)[inner] > 0)
    assert mask.sum() > 0.5 * mask.size
    a = ours.data[inner].cpu().numpy()[mask]
    b = np.asarray(data)[inner][mask]
    peak = float(np.abs(b).max())
    assert float(np.abs(a - b).max()) < 0.01 * peak, f"{mode}: max diff {np.abs(a - b).max():.4g} (peak {peak:.3g})"
