"""Bounded synthetic-photometry parity of ``torchsky.spectrophotometry``.

Comparators: ``speclite`` (AB magnitudes through a tabulated filter) and
``synphot`` (photon-weighted effective stimulus). A plain NumPy trapezoid of the
photon-counting definition is the dependency-free oracle that always runs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import torchsky.spectrophotometry as sp

FILTER_WAVE = np.linspace(2500.0, 10000.0, 6001)
FILTER_T = np.exp(-0.5 * ((FILTER_WAVE - 5500.0) / 400.0) ** 2)
FILTER_T[FILTER_T < 1.0e-6] = 0.0
WAVE = np.linspace(3000.0, 9500.0, 3251)
FLUX = 1.0e-17 * (WAVE / 5500.0) ** -1.5 * (1.0 + 0.3 * np.sin(WAVE / 137.0))  # erg/s/cm2/A


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def _photometry(output: str) -> torch.Tensor:
    band = sp.Passband(_t(FILTER_WAVE), _t(FILTER_T), name="g")
    result = sp.synthetic_photometry(_t(WAVE), _t(FLUX), band, output=output)
    assert bool(result.valid)
    assert float(result.coverage) == pytest.approx(1.0)
    return result.value


def test_mean_f_lambda_is_the_photon_weighted_trapezoid() -> None:
    response = np.interp(WAVE, FILTER_WAVE, FILTER_T)
    expected = np.trapezoid(FLUX * response * WAVE, WAVE) / np.trapezoid(response * WAVE, WAVE)
    assert float(_photometry("mean_f_lambda")) == pytest.approx(expected, rel=1e-9)


def test_ab_magnitude_matches_speclite() -> None:
    filters = pytest.importorskip("speclite.filters")
    units = pytest.importorskip("astropy.units")
    response = filters.FilterResponse(
        wavelength=FILTER_WAVE * units.AA,
        response=FILTER_T,
        meta={"group_name": "harness", "band_name": "g"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = response.get_ab_magnitude(
            FLUX * units.erg / units.s / units.cm**2 / units.AA, WAVE * units.AA
        )
    assert float(_photometry("ab_mag")) == pytest.approx(float(reference), abs=1.0e-5)


def test_photon_weighted_flux_and_ab_match_synphot() -> None:
    pytest.importorskip("synphot")
    units = pytest.importorskip("astropy.units")
    from synphot import Observation, SourceSpectrum, SpectralElement
    from synphot import units as synphot_units
    from synphot.models import Empirical1D

    source = SourceSpectrum(Empirical1D, points=WAVE * units.AA, lookup_table=FLUX * synphot_units.FLAM)
    band = SpectralElement(Empirical1D, points=FILTER_WAVE * units.AA, lookup_table=FILTER_T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observation = Observation(source, band, binset=WAVE * units.AA)
        flam = float(observation.effstim("flam").value)
        abmag = float(observation.effstim("abmag").value)
        jansky = float(observation.effstim("jy").value)
    assert float(_photometry("mean_f_lambda")) == pytest.approx(flam, rel=1e-6)
    # synphot converts the effective F_lambda to AB through its pivot wavelength.
    assert float(_photometry("ab_mag")) == pytest.approx(abmag, abs=2.0e-4)
    assert float(_photometry("f_nu_microjy")) == pytest.approx(jansky * 1.0e6, rel=2.0e-4)


def test_uncovered_band_is_reported_invalid_not_extrapolated() -> None:
    band = sp.Passband(_t(FILTER_WAVE), _t(FILTER_T), name="g")
    narrow = np.linspace(5400.0, 5600.0, 50)
    result = sp.synthetic_photometry(_t(narrow), _t(np.ones_like(narrow)), band)
    assert not bool(result.valid)
    assert bool(torch.isnan(result.value))


def test_synthetic_photometry_is_differentiable_in_flux() -> None:
    flux = _t(FLUX).clone().requires_grad_(True)
    band = sp.Passband(_t(FILTER_WAVE), _t(FILTER_T), name="g")
    sp.synthetic_photometry(_t(WAVE), flux, band).value.backward()
    assert flux.grad is not None
    assert bool(torch.isfinite(flux.grad).all())
    assert float(flux.grad.sum()) == pytest.approx(1.0, rel=1e-9)
