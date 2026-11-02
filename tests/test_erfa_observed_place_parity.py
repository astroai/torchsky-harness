"""Bounded parity of the observed-place chain against pyerfa (SOFA).

Covers ``torchsky.wcs.observed_altaz`` (Earth orientation, IAU 2006/2000A CIO
chain, stellar aberration, Saemundsson refraction), the ICRS to CIRS/TETE frame
transforms and ``propagate_physical`` (``starpv``/``pvstar`` space-motion model).
Reference: ``erfa.atco13`` / ``erfa.c2i06a`` / ``erfa.pnm06a`` / ``erfa.pmsafe``.
Inputs are pinned (no IERS download): DUT1, TT-UTC and polar motion are explicit.
"""

from __future__ import annotations

import numpy as np
import pytest

erfa = pytest.importorskip("erfa")

from _gate_helpers import device, f64, sep_arcsec, to_np  # noqa: E402

from torchsky.wcs import CatalogState, EarthOrientation, Site, observed_altaz, transform  # noqa: E402

UTC_MJD = 60000.0
DUT1 = 0.184
TT_MINUS_UTC = 69.184  # 37 s leap seconds + 32.184 s, matches erfa.dat at MJD 60000
XP, YP = 0.12, 0.34  # arcsec
SITE = dict(lon_deg=-155.5763, lat_deg=19.8260, height_m=4205.0)
JD0 = 2400000.5


def _sources(n: int = 200, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.degrees(np.arcsin(rng.uniform(-0.95, 0.95, n)))
    return ra, dec


def _atco13(ra, dec, *, pressure=0.0, temperature=0.0, xp=XP, yp=YP):
    r = np.radians
    aob, zob, *_ = erfa.atco13(
        r(ra), r(dec), 0.0, 0.0, 0.0, 0.0,
        JD0, UTC_MJD, DUT1, r(SITE["lon_deg"]), r(SITE["lat_deg"]), SITE["height_m"],
        r(xp / 3600.0), r(yp / 3600.0), pressure, temperature, 0.0, 0.55,
    )  # fmt: skip
    return np.degrees(aob), 90.0 - np.degrees(zob)


def _our_altaz(ra, dec, **kw):
    dev = device()
    state = CatalogState(lon=f64(ra, dev), lat=f64(dec, dev))
    eop = EarthOrientation(dut1_s=DUT1, tt_minus_utc_s=TT_MINUS_UTC, xp_arcsec=XP, yp_arcsec=YP)
    site = Site(**SITE, pressure_mbar=kw.pop("pressure_mbar", None), temperature_c=kw.pop("temperature_c", None))
    az, alt = observed_altaz(state, obstime_mjd_utc=UTC_MJD, site=site, eop=eop, **kw)
    return to_np(az), to_np(alt)


def test_observed_altaz_matches_erfa_atco13_no_refraction() -> None:
    ra, dec = _sources()
    az, alt = _our_altaz(ra, dec, pn="iau2006", aberration=True)
    az_ref, alt_ref = _atco13(ra, dec)
    keep = alt_ref > 5.0
    assert keep.sum() > 20
    sep = sep_arcsec(az[keep], alt[keep], az_ref[keep], alt_ref[keep])
    # torchsky documents agreement "to about 1 arcsec" (observed_altaz docstring). The residual is
    # the Keplerian Earth velocity (~540 m/s off ERFA epv00 => <= 0.37") plus the diurnal
    # aberration and solar light deflection that atco13 adds; measured max 0.56", median 0.47".
    assert sep.max() < 1.0, f"max sep {sep.max():.4f} arcsec"
    assert np.median(sep) < 0.6, f"median sep {np.median(sep):.4f} arcsec"


def test_observed_altaz_refraction_matches_erfa_refco_model() -> None:
    ra, dec = _sources(seed=11)
    az0, alt0 = _our_altaz(ra, dec, pn="iau2006", aberration=True)
    az1, alt1 = _our_altaz(ra, dec, pn="iau2006", aberration=True, refraction=True, pressure_mbar=620.0, temperature_c=2.0)
    az_ref, alt_ref = _atco13(ra, dec, pressure=620.0, temperature=2.0)
    az_ref0, alt_ref0 = _atco13(ra, dec)
    keep = alt_ref0 > 20.0
    # Compare the refraction *shift* itself so the unrefracted offset cancels.
    ours = alt1[keep] - alt0[keep]
    ref = alt_ref[keep] - alt_ref0[keep]
    assert np.all(ours > 0.0) and np.all(ref > 0.0)
    # Saemundsson (1986) is quoted accurate to 0.1 arcmin (6") against rigorous refraction; the
    # ERFA two-term refco model is the reference here, so the tolerance is that published accuracy.
    diff = np.abs(ours - ref) * 3600.0
    assert diff.max() < 6.0, f"max refraction-shift difference {diff.max():.2f} arcsec"


@pytest.mark.parametrize(("frame", "reference"), (("CIRS", "c2i06a"), ("TETE", "pnm06a")))
def test_icrs_to_cirs_tete_matches_erfa(frame: str, reference: str) -> None:
    ra, dec = _sources(n=64, seed=5)
    dev = device()
    eop = EarthOrientation(dut1_s=DUT1, tt_minus_utc_s=TT_MINUS_UTC)
    lon, lat = transform(f64(ra, dev), f64(dec, dev), from_frame="ICRS", to_frame=frame, obstime_mjd_utc=UTC_MJD, eop=eop)
    tt = UTC_MJD + TT_MINUS_UTC / 86400.0
    rot = getattr(erfa, reference)(JD0, tt)
    xyz = np.stack(
        [np.cos(np.radians(dec)) * np.cos(np.radians(ra)), np.cos(np.radians(dec)) * np.sin(np.radians(ra)), np.sin(np.radians(dec))],
        axis=-1,
    )
    out = xyz @ rot.T
    ref_lon = np.degrees(np.arctan2(out[:, 1], out[:, 0])) % 360.0
    ref_lat = np.degrees(np.arcsin(out[:, 2]))
    sep = sep_arcsec(to_np(lon), to_np(lat), ref_lon, ref_lat)
    assert sep.max() < 1.0e-3, f"{frame}: max sep {sep.max():.3e} arcsec"


def test_propagate_space_motion_matches_erfa_starpm() -> None:
    from torchsky.wcs import propagate_space_motion

    rng = np.random.default_rng(9)
    n = 50
    ra = rng.uniform(0, 360, n)
    dec = rng.uniform(-80, 80, n)
    pmra = rng.uniform(-300, 300, n)  # mas/yr (pm_ra * cos dec)
    pmdec = rng.uniform(-300, 300, n)
    plx = rng.uniform(5.0, 200.0, n)  # mas
    rv = rng.uniform(-80, 80, n)  # km/s
    dt_yr = 25.0
    r = np.radians
    ref_ra, ref_dec, *_ = erfa.starpm(
        r(ra), r(dec), r(pmra / 3.6e6) / np.cos(r(dec)), r(pmdec / 3.6e6), plx / 1000.0, rv,
        2451545.0, 0.0, 2451545.0, dt_yr * 365.25,
    )  # fmt: skip
    ref_ra, ref_dec = np.degrees(ref_ra) % 360.0, np.degrees(ref_dec)
    dev = device()
    out = propagate_space_motion(
        f64(ra, dev), f64(dec, dev), f64(pmra, dev), f64(pmdec, dev), f64(plx, dev), f64(rv, dev),
        source_epoch_jyear=2000.0, target_epoch_jyear=2000.0 + dt_yr,
    )  # fmt: skip
    sep = sep_arcsec(to_np(out[0]), to_np(out[1]), ref_ra, ref_dec)
    # erfa.pvu advances by 365.25-day years (Julian years), as torchsky does.
    assert sep.max() < 1.0e-3, f"max sep {sep.max():.3e} arcsec"
