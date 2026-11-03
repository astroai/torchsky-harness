"""HEALPix pixel window: torchsky.sphere.pixwin vs healpy.pixwin and an independent reference.

healpy reads the HEALPix ``pixel_window_nNNNN.fits`` tables (downloaded once from healpy-data).
HEALPix documents that those tables are exact only for ``nside <= 128``; for larger nside the
temperature window is extrapolated from ``nside = 128`` with a top-hat scaling in ``l``, and the
polarization window is the temperature one times a low-``l`` factor
(https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm). Hence:

* ``nside <= 128``: torchsky T vs healpy at 1e-11 over ``l <= 4 nside`` (the table extent).
  torchsky P vs healpy carries a definitional term ~0.94e-6 (8/nside)^4 (torchsky transports the
  polarization frame between the two points of each pair; see docs/coverage-matrix.md), bounded
  by 1.2e-6 (8/nside)^4 + 1e-11.
* ``nside >= 256``: torchsky T is gated at 1e-11 against ``tests/_pixwin_reference.py`` (exact
  azimuthal integrals of Y_lm over every pixel class, NumPy only), itself validated against the
  exact healpy tables for nside <= 64. The healpy table deviation is informational: it is
  recorded and only bounded by the documented magnitude (docs/coverage-matrix.md, "pixwin").

The reference costs O(lmax^2 nside^2), so ``lmax`` is bounded per nside; the full ``3 nside - 1``
range at nside 256 and the nside 1024/2048 cases are ``slow`` (the parity suite runs them).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from _perf_record import perf_dir
from _pixwin_reference import reference_pixwin

hp = pytest.importorskip("healpy")
pixwin = pytest.importorskip("torchsky.sphere").pixwin

TIGHT = 1e-11
#: Polarization: |torchsky - healpy| <= P_DEFINITION_AT_8 * (8 / nside)^4 + TIGHT (measured 0.94e-6 at nside 8).
P_DEFINITION_AT_8 = 1.2e-6
#: Documented magnitude of the healpy (extrapolated) table error for nside 256..2048, T and P; max measured 1.28e-5.
HEALPY_EXTRAPOLATION_BOUND = 2e-5


def _healpy(nside: int, lmax: int, pol: bool = False):
    try:
        return hp.pixwin(nside, pol=pol, lmax=lmax)
    except OSError as exc:  # URLError is an OSError: the healpy-data download failed
        pytest.skip(f"healpy pixel-window table for nside={nside} unavailable: {exc}")


def _torchsky(nside: int, lmax: int, pol: bool = False):
    out = pixwin(nside, pol=pol, lmax=lmax)
    return tuple(x.numpy() for x in out) if pol else out.numpy()


def _record(row: dict) -> None:
    path = perf_dir()
    path.mkdir(parents=True, exist_ok=True)
    with (path / "pixwin_healpy_deviation.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("nside", [2, 4, 8, 16, 32, 64, 128])
def test_pixwin_temperature_matches_healpy_exact_tables(nside):
    lmax = 4 * nside
    np.testing.assert_allclose(
        _torchsky(nside, lmax), _healpy(nside, lmax), rtol=0, atol=TIGHT
    )


@pytest.mark.parametrize("nside", [2, 4, 8, 16, 32, 64, 128])
def test_pixwin_polarization_matches_healpy_up_to_frame_definition(nside):
    lmax = 4 * nside
    t_t, t_p = _torchsky(nside, lmax, pol=True)
    h_t, h_p = _healpy(nside, lmax, pol=True)
    assert np.all(t_p[:2] == 0.0)
    diff = np.abs(t_p - h_p)
    bound = P_DEFINITION_AT_8 * (8.0 / nside) ** 4 + TIGHT
    assert diff.max() <= bound, (
        f"nside={nside}: max |dP|={diff.max():.3e} > {bound:.3e}"
    )


@pytest.mark.parametrize("nside", [2, 4, 8, 16, 32, 64])
def test_reference_matches_healpy_exact_tables(nside):
    """The independent reference reproduces the exact (nside <= 128) HEALPix tables."""

    lmax = 4 * nside
    np.testing.assert_allclose(
        reference_pixwin(nside, lmax), _healpy(nside, lmax), rtol=0, atol=TIGHT
    )


REFERENCE_CASES = [
    pytest.param(256, 256, id="256-l256"),
    pytest.param(512, 128, id="512-l128"),
    pytest.param(1024, 64, id="1024-l64", marks=pytest.mark.slow),
    pytest.param(2048, 64, id="2048-l64", marks=pytest.mark.slow),
    pytest.param(256, 767, id="256-l767", marks=pytest.mark.slow),
]


@pytest.mark.parametrize(("nside", "lmax"), REFERENCE_CASES)
def test_pixwin_temperature_matches_independent_reference(nside, lmax):
    reference = reference_pixwin(nside, lmax)
    torchsky = _torchsky(nside, lmax)
    torchsky_err = float(np.abs(torchsky - reference).max())

    # Informational: the extrapolated healpy table against the same reference (l <= lmax) and
    # against torchsky over the full table range (T and P). Recorded, bounded, never tight.
    healpy = _healpy(nside, lmax)
    healpy_err = float(np.abs(healpy - reference).max())
    full = 4 * nside
    h_t, h_p = _healpy(nside, full, pol=True)
    t_t, t_p = _torchsky(nside, full, pol=True)
    row = {
        "nside": nside, "lmax_reference": lmax,
        "torchsky_vs_reference": torchsky_err, "healpy_vs_reference": healpy_err,
        "healpy_vs_torchsky_T_full": float(np.abs(h_t - t_t).max()),
        "healpy_vs_torchsky_P_full": float(np.abs(h_p - t_p).max()),
    }  # fmt: skip
    _record(row)
    print(json.dumps(row))

    assert torchsky_err <= TIGHT, (
        f"nside={nside}: torchsky vs reference {torchsky_err:.3e}"
    )
    for key in (
        "healpy_vs_reference",
        "healpy_vs_torchsky_T_full",
        "healpy_vs_torchsky_P_full",
    ):
        assert row[key] <= HEALPY_EXTRAPOLATION_BOUND, (
            f"healpy table deviation grew beyond the documented bound: {row}"
        )
