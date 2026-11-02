"""Bounded parity of ``torchsky.extinction`` against ``dust_extinction``.

CCM89, O'Donnell (1994) and Fitzpatrick (1999) are compared with
``dust_extinction.parameter_averages`` over their common validity range
(0.125-3.2 micron). Calzetti (2000) is checked against the published
piecewise polynomial. G23 is not provided by the subject (see
``docs/coverage-matrix.md``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchsky.extinction as te

LAMBDA_A = np.linspace(1250.0, 32000.0, 240)
RVS = (2.5, 3.1, 4.0)
# (torchsky function, dust_extinction class name, rtol, atol in mag per A_V)
LAWS = (
    ("ccm89", "CCM89", 3.0e-4, 3.0e-4),
    ("odonnell94", "O94", 3.0e-4, 3.0e-4),
    ("fitzpatrick99", "F99", 3.0e-3, 1.0e-3),
)


def _t(x: np.ndarray | float) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("rv", RVS)
@pytest.mark.parametrize(("name", "reference_name", "rtol", "atol"), LAWS)
def test_a_lambda_matches_dust_extinction(
    name: str, reference_name: str, rtol: float, atol: float, rv: float
) -> None:
    units = pytest.importorskip("astropy.units")
    parameter_averages = pytest.importorskip("dust_extinction.parameter_averages")
    reference = getattr(parameter_averages, reference_name)(Rv=rv)(LAMBDA_A * units.AA)
    for a_v in (1.0, 0.3):
        result = getattr(te, name)(_t(LAMBDA_A), _t(a_v), rv).numpy()
        np.testing.assert_allclose(result, a_v * np.asarray(reference), rtol=rtol, atol=atol)


def test_ccm89_and_odonnell94_differ_in_the_optical_like_the_references() -> None:
    units = pytest.importorskip("astropy.units")
    parameter_averages = pytest.importorskip("dust_extinction.parameter_averages")
    optical = np.linspace(3000.0, 9000.0, 200)
    ours = te.ccm89(_t(optical), _t(1.0), 3.1) - te.odonnell94(_t(optical), _t(1.0), 3.1)
    theirs = parameter_averages.CCM89(Rv=3.1)(optical * units.AA) - parameter_averages.O94(Rv=3.1)(
        optical * units.AA
    )
    np.testing.assert_allclose(ours.numpy(), np.asarray(theirs), atol=1.0e-6)


def test_apply_extinction_matches_dust_extinction_extinguish() -> None:
    units = pytest.importorskip("astropy.units")
    parameter_averages = pytest.importorskip("dust_extinction.parameter_averages")
    lam = np.linspace(2000.0, 20000.0, 100)
    ebv, rv = 0.12, 3.1
    reference = parameter_averages.CCM89(Rv=rv).extinguish(lam * units.AA, Ebv=ebv)
    result = te.apply_extinction(torch.ones(100, dtype=torch.float64), _t(lam), _t(ebv), "ccm89", rv)
    np.testing.assert_allclose(result.numpy(), np.asarray(reference), atol=1.0e-6)


def _calzetti_k(lam_um: np.ndarray, rv: float) -> np.ndarray:
    blue = 2.659 * (-2.156 + 1.509 / lam_um - 0.198 / lam_um**2 + 0.011 / lam_um**3) + rv
    red = 2.659 * (-1.857 + 1.040 / lam_um) + rv
    return np.where(lam_um < 0.63, blue, red)


def test_calzetti_matches_published_polynomial() -> None:
    lam = np.linspace(1200.0, 22000.0, 300)
    expected = _calzetti_k(lam / 1.0e4, 4.05)
    np.testing.assert_allclose(te.calzetti_k_ebv(_t(lam), 4.05).numpy(), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        te.calzetti(_t(lam), _t(1.0), 4.05).numpy(), expected / 4.05, rtol=0.0, atol=1e-6
    )


def test_a_lambda_is_differentiable_in_extinction_amplitude() -> None:
    a_v = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    te.ccm89(_t(LAMBDA_A), a_v, 3.1).sum().backward()
    assert a_v.grad is not None
    assert float(a_v.grad) == pytest.approx(float(te.ccm89(_t(LAMBDA_A), _t(1.0), 3.1).sum()), rel=1e-12)


@pytest.mark.skip(reason="G23 is not provided by torchsky.extinction; tracked as a coverage gap, not gated")
def test_g23_parity() -> None:  # pragma: no cover
    raise AssertionError
