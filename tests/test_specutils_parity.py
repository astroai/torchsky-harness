"""Bounded parity of ``torchsky.spectral`` against ``specutils``.

* ``flux_conserving_rebin``      vs ``specutils.manipulation.FluxConservingResampler``
  (bin edges are taken from specutils' own ``SpectralAxis.bin_edges`` so both sides integrate
  the same piecewise-constant spectrum over the same bins).
* ``relativistic_doppler_shift`` vs ``Spectrum.shift_spectrum_to(radial_velocity=...)``
  (specutils shifts the axis with the relativistic Doppler factor; torchsky also rescales the
  flux density by 1/factor, which is checked against integrated-flux conservation).
"""

from __future__ import annotations

import numpy as np
import pytest

specutils = pytest.importorskip("specutils")
import astropy.units as u  # noqa: E402
from specutils import Spectrum  # noqa: E402
from specutils.manipulation import FluxConservingResampler  # noqa: E402

from _gate_helpers import f64, to_np  # noqa: E402

from torchsky.spectral import flux_conserving_rebin, relativistic_doppler_shift  # noqa: E402


def _spectrum(n: int = 400, seed: int = 0) -> Spectrum:
    rng = np.random.default_rng(seed)
    # non-uniform sampling stresses the edge construction
    wl = np.sort(4000.0 + 3000.0 * rng.uniform(size=n))
    wl[0], wl[-1] = 4000.0, 7000.0
    flux = (2.0 + np.sin(wl / 80.0) + 0.1 * rng.normal(size=n)) * u.Jy
    return Spectrum(spectral_axis=wl * u.AA, flux=flux)


@pytest.mark.parametrize("n_out", (97, 250, 700))
def test_flux_conserving_rebin_matches_specutils(n_out: int) -> None:
    spec = _spectrum()
    out_axis = np.linspace(4050.0, 6950.0, n_out) * u.AA
    ref = FluxConservingResampler()(spec, out_axis)
    in_edges = spec.spectral_axis.bin_edges.to_value(u.AA)
    out_edges = ref.spectral_axis.bin_edges.to_value(u.AA)
    got = flux_conserving_rebin(f64(spec.flux.value), f64(in_edges), f64(out_edges))
    ref_flux = np.asarray(ref.flux.value)
    finite = np.isfinite(ref_flux)
    inside = to_np(got.mask) & finite
    assert inside.sum() > 0.8 * n_out
    np.testing.assert_allclose(to_np(got.flux)[inside], ref_flux[inside], rtol=1e-10, atol=1e-12)


def test_flux_conserving_rebin_conserves_integral_like_specutils() -> None:
    spec = _spectrum(seed=3)
    in_edges = spec.spectral_axis.bin_edges.to_value(u.AA)
    out_edges = np.linspace(in_edges[0], in_edges[-1], 181)
    got = flux_conserving_rebin(f64(spec.flux.value), f64(in_edges), f64(out_edges))
    total_in = float(np.sum(spec.flux.value * np.diff(in_edges)))
    total_out = float(np.sum(to_np(got.flux) * np.diff(out_edges)))
    assert abs(total_out - total_in) < 1e-10 * abs(total_in)


@pytest.mark.parametrize("velocity_km_s", (-30000.0, -500.0, 0.0, 120.0, 30000.0))
def test_relativistic_doppler_axis_matches_specutils_shift(velocity_km_s: float) -> None:
    spec = Spectrum(spectral_axis=np.linspace(4000.0, 7000.0, 301) * u.AA, flux=np.ones(301) * u.Jy, radial_velocity=0 * u.km / u.s)
    edges = spec.spectral_axis.bin_edges.to_value(u.AA)
    shifted = spec.shift_spectrum_to(radial_velocity=velocity_km_s * u.km / u.s)
    assert shifted is None or shifted is spec  # specutils >= 2 shifts in place
    ref_centres = spec.spectral_axis.to_value(u.AA)
    got = relativistic_doppler_shift(f64(np.ones(301)), f64(edges), velocity_km_s * 1000.0)
    got_centres = 0.5 * (to_np(got.spectral_edges)[1:] + to_np(got.spectral_edges)[:-1])
    np.testing.assert_allclose(got_centres, ref_centres, rtol=1e-12)
    # flux density scales as 1/D so the integrated flux over the shifted bins is unchanged
    widths = np.diff(to_np(got.spectral_edges))
    np.testing.assert_allclose(np.sum(to_np(got.flux) * widths), np.sum(np.diff(edges)), rtol=1e-12)
