"""Bounded parity of ``torchsky.map.coadd`` drizzle kernels against AstroDrizzle (drizzlepac).

``drizzlepac.adrizzle.do_driz`` (the ``cdriz.tdriz`` C kernel, WCSLIB-based pixel mapping) runs
out-of-process on a synthetic WCS pair. Relationship to the ``drizzle`` package gate
(``test_map_drizzle_external_parity.py``): the ``drizzle`` package is the STScI
re-implementation of the same algorithm; this gate ties torchsky to the original AstroDrizzle C
kernel directly. Both are float32 accumulators, so the data agree to float32 round-off
(measured ~1e-7); the bound is 1e-4 absolute on weight and data over the common support.
Pixels where exactly one of the two has zero weight (kernel footprint boundary) are counted and
must be a vanishing fraction.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torchfits

import _oracle

pytest.importorskip("torchsky._cpp")

from torchsky.map import coadd as coadd_image_plane  # noqa: E402
from torchsky.scene import ImageObservation  # noqa: E402


def _header(*, nx=16, ny=16, crval1=180.0, crval2=0.0, rotation_deg=0.0, scale=1.0) -> dict:
    theta = math.radians(rotation_deg)
    pix = 0.01 * float(scale)
    c, s = math.cos(theta), math.sin(theta)
    return {
        "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 0.5 * (nx + 1), "CRPIX2": 0.5 * (ny + 1), "CRVAL1": crval1, "CRVAL2": crval2,
        "CD1_1": -pix * c, "CD1_2": pix * s, "CD2_1": pix * s, "CD2_2": pix * c,
    }  # fmt: skip


def _blob(n: int = 16) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    return (
        0.7 * torch.exp(-((xx - 5.0) ** 2 + (yy - 6.0) ** 2) / 6.0)
        + 1.1 * torch.exp(-((xx - 10.5) ** 2 + (yy - 9.0) ** 2) / 9.0)
        + 0.02 * xx
    )


def _cd(header: dict) -> np.ndarray:
    return np.array([[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]])


TARGETS = {
    "shift": _header(crval1=180.03),
    "rotate": _header(rotation_deg=11.0),
    "scale": _header(scale=1.08),
}


@pytest.mark.parametrize("pixfrac", [0.7, 1.0])
@pytest.mark.parametrize("kernel", ["drizzle_square", "drizzle_point"])
@pytest.mark.parametrize("case", sorted(TARGETS))
def test_astrodrizzle_matches_torchsky(tmp_path: Path, case: str, kernel: str, pixfrac: float) -> None:
    src_header, dst_header = _header(), TARGETS[case]
    image = _blob()
    path = tmp_path / "src.fits"
    torchfits.write(str(path), image, header=src_header, overwrite=True)
    ref = _oracle.run_drizzle_oracle(
        src_header, dst_header, image.numpy(), np.ones((16, 16)), kernel.split("_")[1], pixfrac
    )
    native = coadd_image_plane(
        [ImageObservation(str(path))],
        target=dict(dst_header),
        kernel=kernel,
        semantics="surface_brightness",
        pixfrac=pixfrac,
        method="weighted_mean",
        device="cpu",
    )
    w_t, d_t = native.weight.numpy(), native.data.numpy()
    # AstroDrizzle scales drops by the output/input pixel-area ratio (flux conserving); torchsky
    # "surface_brightness" does not, so divide the reference by that ratio.
    area = abs(np.linalg.det(_cd(dst_header)) / np.linalg.det(_cd(src_header)))
    w_r, d_r = ref["wht"], ref["sci"] / area
    both = (w_t > 0) & (w_r > 0)
    assert both.sum() > 100
    assert float(np.abs(w_t - w_r)[both].max()) < 1e-4, f"weight max diff {float(np.abs(w_t - w_r)[both].max()):.2e}"
    assert float(np.abs(d_t - d_r)[both].max()) < 1e-4, f"data max diff {float(np.abs(d_t - d_r)[both].max()):.2e}"
    edge = ((w_t > 0) != (w_r > 0)).sum()
    assert edge <= 0.02 * both.sum(), f"{int(edge)} boundary pixels covered by only one implementation"
