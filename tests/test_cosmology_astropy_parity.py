"""Bounded parity of ``torchsky.cosmology`` against ``astropy.cosmology``.

Distances and the Hubble function are checked over z in [1e-5, 1100] for flat,
open, closed and w0-wa models (radiation off, matching torchsky's matter +
curvature + CPL dark-energy model). Cosmic age is not part of the torchsky
surface, so it is not compared.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

astropy_cosmology = pytest.importorskip("astropy.cosmology")
tc = pytest.importorskip("torchsky.cosmology")

H0, OM0 = 67.66, 0.3111
Z = np.logspace(-5.0, np.log10(1100.0), 400)
RTOL = 1.0e-9


def _t(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64)


MODELS = {
    "flat_lcdm": (
        {},
        lambda: astropy_cosmology.FlatLambdaCDM(H0, OM0, Tcmb0=0),
    ),
    "closed": (
        {"ok0": 0.05},
        lambda: astropy_cosmology.LambdaCDM(H0, OM0, 1 - OM0 - 0.05, Tcmb0=0),
    ),
    "open": (
        {"ok0": -0.05},
        lambda: astropy_cosmology.LambdaCDM(H0, OM0, 1 - OM0 + 0.05, Tcmb0=0),
    ),
    "flat_w0wa": (
        {"w0": -0.9, "wa": 0.2},
        lambda: astropy_cosmology.Flatw0waCDM(H0, OM0, w0=-0.9, wa=0.2, Tcmb0=0),
    ),
}

QUANTITIES = (
    ("comoving_distance", "comoving_distance", True),
    ("luminosity_distance", "luminosity_distance", True),
    ("angular_diameter_distance", "angular_diameter_distance", True),
    ("transverse_comoving_distance", "comoving_transverse_distance", True),
    ("distance_modulus", "distmod", True),
    ("hubble_parameter", "H", False),
)


def _kwargs(extra: dict[str, float]) -> dict[str, torch.Tensor]:
    return {"H0": _t(H0), "om0": _t(OM0), **{k: _t(v) for k, v in extra.items()}}


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize(("ours", "theirs", "_integral"), QUANTITIES)
def test_distances_match_astropy(model: str, ours: str, theirs: str, _integral: bool) -> None:
    extra, build = MODELS[model]
    reference = getattr(build(), theirs)(Z)
    reference = np.asarray(getattr(reference, "value", reference), dtype=np.float64)
    result = getattr(tc, ours)(torch.tensor(Z, dtype=torch.float64), **_kwargs(extra))
    assert result.dtype == torch.float64
    np.testing.assert_allclose(result.numpy(), reference, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_flrw_object_and_comoving_volume_match_astropy(model: str) -> None:
    extra, build = MODELS[model]
    astropy_model = build()
    flrw = tc.FLRW(
        H0=_t(H0),
        Om0=_t(OM0),
        Ok0=_t(extra.get("ok0", 0.0)),
        w0=_t(extra.get("w0", -1.0)),
        wa=_t(extra.get("wa", 0.0)),
    )
    z = torch.tensor(Z, dtype=torch.float64)
    np.testing.assert_allclose(
        flrw.luminosity_distance(z).numpy(),
        astropy_model.luminosity_distance(Z).value,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        tc.differential_comoving_volume(z, flrw).numpy(),
        astropy_model.differential_comoving_volume(Z).value,
        rtol=RTOL,
    )


def test_distance_modulus_is_consistent_with_luminosity_distance() -> None:
    z = torch.tensor(Z, dtype=torch.float64)
    mu = tc.distance_modulus(z, H0=_t(H0), om0=_t(OM0))
    dl = tc.luminosity_distance(z, H0=_t(H0), om0=_t(OM0))
    torch.testing.assert_close(mu, 5.0 * torch.log10(dl * 1.0e6 / 10.0), rtol=0.0, atol=1e-12)


def test_python_float_parameters_keep_float64_precision() -> None:
    reference = astropy_cosmology.FlatLambdaCDM(H0, OM0, Tcmb0=0).comoving_distance(Z).value
    result = tc.comoving_distance(torch.tensor(Z, dtype=torch.float64), H0=H0, om0=OM0)
    np.testing.assert_allclose(result.numpy(), reference, rtol=RTOL, atol=0.0)


def test_gradient_wrt_om0_matches_finite_difference() -> None:
    z = torch.tensor([0.1, 0.5, 1.0, 3.0], dtype=torch.float64)
    om0 = torch.tensor(OM0, dtype=torch.float64, requires_grad=True)
    tc.luminosity_distance(z, H0=_t(H0), om0=om0).sum().backward()
    h = 1.0e-5
    plus = tc.luminosity_distance(z, H0=_t(H0), om0=_t(OM0 + h)).sum()
    minus = tc.luminosity_distance(z, H0=_t(H0), om0=_t(OM0 - h)).sum()
    assert om0.grad is not None
    assert float(om0.grad) == pytest.approx(float((plus - minus) / (2 * h)), rel=1e-6)
