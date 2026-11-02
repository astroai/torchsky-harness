"""Bounded parity of ``torchsky.redshift`` / ``torchsky.log_lambda`` with astropy units.

Reference: ``astropy.units`` spectral equivalencies (``doppler_redshift``,
``doppler_relativistic``) and ``astropy.constants``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

u = pytest.importorskip("astropy.units")
const = pytest.importorskip("astropy.constants")
from astropy.stats import gaussian_fwhm_to_sigma  # noqa: E402

redshift = pytest.importorskip("torchsky.redshift")
log_lambda = pytest.importorskip("torchsky.log_lambda")

KMS = np.concatenate([-np.logspace(-2, np.log10(2.5e5), 25)[::-1], [0.0], np.logspace(-2, np.log10(2.5e5), 25)])


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def test_speed_of_light_constant_matches_astropy() -> None:
    assert redshift.C_KM_S == const.c.to_value(u.km / u.s)
    assert log_lambda.C_KM_S == const.c.to_value(u.km / u.s)


def test_velocity_to_redshift_is_astropy_relativistic_doppler() -> None:
    reference = (KMS * u.km / u.s).to(u.dimensionless_unscaled, u.doppler_redshift())
    z = redshift.convert_velocity_kind(_t(KMS), redshift.V_KIND_KMS, redshift.V_KIND_REDSHIFT)
    np.testing.assert_allclose(z.numpy(), reference.value, rtol=1e-12, atol=1e-15)


def test_redshift_to_velocity_roundtrip_matches_astropy() -> None:
    z = np.concatenate([[0.0], np.logspace(-5, np.log10(1100.0), 40)])
    reference = (z * u.dimensionless_unscaled).to(u.km / u.s, u.doppler_redshift())
    v = redshift.convert_velocity_kind(_t(z), redshift.V_KIND_REDSHIFT, redshift.V_KIND_KMS)
    np.testing.assert_allclose(v.numpy(), reference.value, rtol=1e-12, atol=1e-9)
    back = redshift.convert_velocity_kind(v, redshift.V_KIND_KMS, redshift.V_KIND_REDSHIFT)
    # beta -> 1 at z ~ 1e3 is ill-conditioned (relative error ~3e-11).
    np.testing.assert_allclose(back.numpy(), z, rtol=1e-9, atol=1e-15)


def test_beta_from_redshift_matches_astropy_velocity_over_c() -> None:
    z = np.logspace(-4, 3, 30)
    beta = redshift.beta_from_redshift(_t(z))
    v = (z * u.dimensionless_unscaled).to(u.km / u.s, u.doppler_redshift())
    np.testing.assert_allclose(beta.numpy(), (v / const.c).to_value(u.dimensionless_unscaled), rtol=1e-12)


def test_apply_redshift_matches_astropy_relativistic_frequency_shift() -> None:
    lam_rest = np.linspace(3000.0, 9000.0, 33)  # Angstrom
    flux = np.ones_like(lam_rest)
    v = 12345.0  # km/s, receding
    flux_obs, lam_obs = redshift.apply_redshift(_t(flux), _t(lam_rest), torch.tensor(v, dtype=torch.float64))
    nu_rest = (lam_rest * u.AA).to(u.Hz, u.spectral())
    nu_obs = np.array(
        [
            (v * u.km / u.s).to(u.Hz, u.doppler_relativistic(nu)).value
            for nu in nu_rest
        ]
    )
    lam_ref = (nu_obs * u.Hz).to(u.AA, u.spectral()).value
    np.testing.assert_allclose(lam_obs.numpy(), lam_ref, rtol=1e-12)
    one_plus_z = lam_ref / lam_rest
    np.testing.assert_allclose(flux_obs.numpy(), flux / one_plus_z, rtol=1e-12)


def test_f_lambda_and_tolman_dimming_exponents() -> None:
    opz = np.array([1.0, 1.5, 3.0, 11.0])
    np.testing.assert_allclose(redshift.dimming_f_lambda(_t(opz)).numpy(), opz**-1.0, rtol=1e-14)
    np.testing.assert_allclose(redshift.dimming_tolman_sb(_t(opz)).numpy(), opz**-4.0, rtol=1e-14)


def test_resolving_power_sigma_matches_astropy_fwhm_relation() -> None:
    resolving_power = np.array([100.0, 1000.0, 2700.0, 30000.0])
    sigma = log_lambda.sigma_R_kms(_t(resolving_power)).numpy()
    fwhm = const.c.to_value(u.km / u.s) / resolving_power
    # torchsky documents sigma = c / (2.355 R); the exact factor differs by 1.5e-4.
    np.testing.assert_allclose(sigma, fwhm * gaussian_fwhm_to_sigma, rtol=2.0e-4)


def test_log_lambda_convolution_widens_a_line_by_the_requested_velocity() -> None:
    n, dln = 2048, 1.0e-4
    sigma_in, sigma_kms = 20.0, 60.0
    sig_px = lambda s: s / const.c.to_value(u.km / u.s) / dln  # noqa: E731
    x = (np.arange(n) - n // 2).astype(np.float64)
    line = np.exp(-0.5 * (x / sig_px(sigma_in)) ** 2)
    out = log_lambda.convolve_log_lambda(_t(line), dln_lambda=dln, sigma_kms=sigma_kms, kernel_radius=64).numpy()
    expected = np.hypot(sigma_in, sigma_kms)
    measured = np.sqrt(np.sum(out * x**2) / np.sum(out)) * dln * const.c.to_value(u.km / u.s)
    assert measured == pytest.approx(expected, rel=2e-3)
