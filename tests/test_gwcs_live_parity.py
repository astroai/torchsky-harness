"""Bounded parity of ``torchsky.wcs.gwcs_live.freeze_live_gwcs`` against live ``gwcs``.

A distorted TAN detector-to-sky pipeline is built with ``gwcs`` + ``astropy.modeling``
(shift, polynomial distortion, pixel scale, TAN, native-to-celestial rotation). The
frozen torch table must reproduce the live gwcs model on independent points to the
sampler's own sub-mas validation limits, and must run on the subject's device.
"""

from __future__ import annotations

import numpy as np
import pytest

gwcs = pytest.importorskip("gwcs")
from astropy import coordinates as coord  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.modeling import models  # noqa: E402
from gwcs import coordinate_frames as cf  # noqa: E402

from _gate_helpers import device, f64, sep_arcsec, to_np  # noqa: E402

from torchsky.wcs.gwcs_live import freeze_live_gwcs  # noqa: E402

RA0, DEC0 = 150.1, 2.2
SCALE_DEG = 0.05 / 3600.0  # 50 mas / pixel
BOUNDS = ((0.0, 2047.0), (0.0, 2047.0))


def _live_gwcs():
    shift = models.Shift(-1023.5) & models.Shift(-1023.5)
    distortion = models.Mapping((0, 1, 0, 1)) | (
        models.Polynomial2D(2, c0_0=0.0, c1_0=1.0, c0_1=0.0, c2_0=2.0e-6, c1_1=-1.0e-6, c0_2=0.5e-6)
        & models.Polynomial2D(2, c0_0=0.0, c1_0=0.0, c0_1=1.0, c2_0=-0.5e-6, c1_1=1.5e-6, c0_2=1.0e-6)
    )
    scale = models.Scale(SCALE_DEG) & models.Scale(SCALE_DEG)
    celestial = models.Pix2Sky_TAN() | models.RotateNative2Celestial(RA0, DEC0, 180.0)
    model = shift | distortion | scale | celestial
    detector = cf.Frame2D(name="detector", axes_order=(0, 1), unit=(u.pix, u.pix))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg))
    return gwcs.WCS([(detector, model), (sky, None)])


def test_frozen_gwcs_matches_live_model_on_independent_points() -> None:
    live = _live_gwcs()
    frozen = freeze_live_gwcs(
        live, BOUNDS, coordinates="sky_degrees", p99_limit_mas=0.01, max_limit_mas=0.05, seed=3
    )
    assert frozen.passed
    rng = np.random.default_rng(12345)  # independent of the sampler's validation seed
    xy = rng.uniform(0.0, 2047.0, size=(5000, 2))
    ra_ref, dec_ref = live.pixel_to_world_values(xy[:, 0], xy[:, 1])
    table = frozen.transform.to(device=device()) if device().type != "mps" else frozen.transform
    out = to_np(table(f64(xy, table.table.device)))
    sep_mas = sep_arcsec(out[:, 0], out[:, 1], ra_ref, dec_ref) * 1000.0
    # Sampler gate is 0.01 mas (p99) / 0.05 mas (max); allow the max gate on fresh points.
    assert np.percentile(sep_mas, 99) < 0.02, f"p99 {np.percentile(sep_mas, 99):.4f} mas"
    assert sep_mas.max() < 0.1, f"max {sep_mas.max():.4f} mas"


def test_frozen_gwcs_inverse_roundtrip_against_live_inverse() -> None:
    live = _live_gwcs()
    frozen = freeze_live_gwcs(live, BOUNDS, coordinates="sky_degrees", seed=1)
    table = frozen.transform
    rng = np.random.default_rng(7)
    xy = rng.uniform(100.0, 1900.0, size=(500, 2))
    sky = table(f64(xy, table.table.device))
    back = to_np(table.inverse(sky))
    # Pixel round trip: 1e-3 pixel (= 0.05 mas at 50 mas/pix) bounds the table inversion.
    assert np.abs(back - xy).max() < 1.0e-3
    ra, dec = live.pixel_to_world_values(xy[:, 0], xy[:, 1])
    live_back = np.stack(live.world_to_pixel_values(ra, dec), axis=-1)
    assert np.abs(live_back - xy).max() < 1.0e-3  # the oracle itself is exact enough
