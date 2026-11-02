"""Bounded parity of torchsky's opt-in interchange extras against their packages.

* ``catalog_state_from_tle``   vs ``sgp4`` (published SGP4 verification vector, satellite 00005)
* ``ephemeris.get_body``       vs ``jplephem`` via astropy and ERFA ``epv00`` (DE440s kernel)
* ``astropy_interop``          vs ``astropy.coordinates.SkyCoord`` (round trip, frame conversion)

The DE440s kernel is read from ``TORCHSKY_HARNESS_DE_KERNEL`` or downloaded once from
NAIF into ``~/.cache/torchsky-harness``. The small DE440s kernel carries planet-centre segments
only for Mercury, Venus, Earth and Moon (Mars and outer planets are barycentre-only there, so
``get_body`` raises a documented ``ValueError``; the full DE440 is needed for those); if neither works the ephemeris test skips (and
fails when gates are required).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import urllib.request

import numpy as np
import pytest

from _gate_helpers import sep_arcsec, to_np  # noqa: E402

DE440S_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"

# Vallado et al. SGP4 verification case, satellite 00005 (tcppver.out, tsince = 0).
TLE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
TLE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
TLE_R0_KM = np.array([7022.46529266, -1400.08296755, 0.03995155])


def test_tle_state_matches_published_sgp4_vector_and_package() -> None:
    sgp4_api = pytest.importorskip("sgp4.api")
    from torchsky.wcs.tle import catalog_state_from_tle

    sat = sgp4_api.Satrec.twoline2rv(TLE1, TLE2)
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(days=sat.jdsatepoch + sat.jdsatepochF - 2451544.5)
    # microsecond rounding of the datetime epoch moves a LEO satellite by < 8 mm
    state = catalog_state_from_tle(TLE1, TLE2, epoch.replace(tzinfo=None))
    assert state.frame == "TEME"
    lon, lat = to_np(state.lon), to_np(state.lat)
    direction = TLE_R0_KM / np.linalg.norm(TLE_R0_KM)
    ref_lon = np.degrees(np.arctan2(direction[1], direction[0])) % 360.0
    ref_lat = np.degrees(np.arcsin(direction[2]))
    sep = sep_arcsec(lon, lat, ref_lon, ref_lat)
    # published vector is quoted to 1e-8 km; 8 mm / 7000 km = 0.24 arcsec worst case
    assert float(sep.max()) < 0.3, f"separation from published SGP4 vector {float(sep.max()):.4f} arcsec"


def test_tle_state_matches_sgp4_at_later_times() -> None:
    sgp4_api = pytest.importorskip("sgp4.api")
    from torchsky.wcs.tle import catalog_state_from_tle

    sat = sgp4_api.Satrec.twoline2rv(TLE1, TLE2)
    for hours in (1.0, 6.5, 24.0):
        when = datetime(2000, 6, 28, 12, 0, 0) + timedelta(hours=hours)
        jd, fr = sgp4_api.jday(when.year, when.month, when.day, when.hour, when.minute, when.second)
        err, r, _ = sat.sgp4(jd, fr)
        assert err == 0
        state = catalog_state_from_tle(TLE1, TLE2, when)
        unit = np.asarray(r) / np.linalg.norm(r)
        sep = sep_arcsec(
            to_np(state.lon), to_np(state.lat),
            np.degrees(np.arctan2(unit[1], unit[0])) % 360.0, np.degrees(np.arcsin(unit[2])),
        )  # fmt: skip
        assert float(sep.max()) < 1.0e-6


def _kernel() -> str:
    env = os.environ.get("TORCHSKY_HARNESS_DE_KERNEL")
    if env and Path(env).is_file():
        return env
    cache = Path.home() / ".cache" / "torchsky-harness" / "de440s.bsp"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part")
        try:
            urllib.request.urlretrieve(DE440S_URL, tmp)  # noqa: S310 - fixed NAIF URL
            tmp.replace(cache)
        except Exception as exc:  # network unavailable
            pytest.skip(f"DE440s kernel unavailable: {exc}")
    return str(cache)


@pytest.mark.parametrize("body", ("earth", "moon", "sun", "venus", "mercury"))
def test_get_body_matches_jplephem_via_astropy(body: str) -> None:
    pytest.importorskip("jplephem")
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    from astropy.time import Time

    from torchsky.wcs.ephemeris import get_body

    kernel = _kernel()
    for tt_mjd in (51544.5, 55000.25, 60000.0, 64000.75):
        state = get_body(body, tt_mjd, path=kernel)
        with solar_system_ephemeris.set(kernel):
            # astropy interprets Time(scale="tdb") as the SPK barycentric dynamical time;
            # torchsky feeds tt_mjd straight into the SPK (TT-TDB < 1.7 ms => < 0.1 km for Earth).
            xyz = get_body_barycentric(body, Time(tt_mjd, format="mjd", scale="tdb")).xyz.to_value("km")
        xyz = np.asarray(xyz, dtype=np.float64)
        unit = xyz / np.linalg.norm(xyz)
        sep = sep_arcsec(
            to_np(state.lon), to_np(state.lat),
            np.degrees(np.arctan2(unit[1], unit[0])) % 360.0, np.degrees(np.arcsin(unit[2])),
        )  # fmt: skip
        # 1.7 ms of Earth motion (30 km/s) is 51 m; Moon/Earth-orbit speeds give <= ~0.1 mas at 1 AU
        # except for the Sun (SSB offset ~1e6 km, moves ~1e-2 km/s) which stays below 1 mas.
        assert float(sep.max()) < 1.0e-3, f"{body} @ MJD {tt_mjd}: {float(sep.max()) * 1e3:.4f} mas"


def test_get_body_earth_matches_erfa_epv00() -> None:
    pytest.importorskip("jplephem")
    erfa = pytest.importorskip("erfa")
    from torchsky.wcs.ephemeris import get_body

    kernel = _kernel()
    for tt_mjd in (51544.5, 55000.25, 60000.0):
        state = get_body("earth", tt_mjd, path=kernel)
        _, pb = erfa.epv00(2400000.5, tt_mjd)
        unit = pb[0] / np.linalg.norm(pb[0])
        sep = sep_arcsec(
            to_np(state.lon), to_np(state.lat),
            np.degrees(np.arctan2(unit[1], unit[0])) % 360.0, np.degrees(np.arcsin(unit[2])),
        )  # fmt: skip
        # epv00 is a truncated series: its barycentric position is ~8e-7 AU (~120 km) from DE440 at
        # these epochs (measured with jplephem directly), i.e. ~0.17" at 1 AU. The bound is 1e-6 rad.
        assert float(sep.max()) < 0.25, f"{float(sep.max()) * 1e3:.2f} mas"


def test_astropy_interop_roundtrip_and_frame_conversion() -> None:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    from torchsky.wcs.astropy_interop import catalog_state_from_astropy, catalog_state_to_astropy

    rng = np.random.default_rng(5)
    n = 40
    sky = SkyCoord(
        ra=rng.uniform(0, 360, n) * u.deg, dec=rng.uniform(-80, 80, n) * u.deg,
        pm_ra_cosdec=rng.uniform(-50, 50, n) * u.mas / u.yr, pm_dec=rng.uniform(-50, 50, n) * u.mas / u.yr,
        distance=rng.uniform(50, 500, n) * u.pc, radial_velocity=rng.uniform(-30, 30, n) * u.km / u.s,
        obstime=Time(2015.5, format="jyear", scale="tcb"), frame="icrs",
    )  # fmt: skip
    state = catalog_state_from_astropy(sky)
    back = catalog_state_to_astropy(state)
    assert float(sky.separation(back).to_value(u.mas).max()) < 1.0e-6
    np.testing.assert_allclose(back.pm_ra_cosdec.to_value(u.mas / u.yr), sky.pm_ra_cosdec.to_value(u.mas / u.yr), rtol=1e-12)
    np.testing.assert_allclose(back.distance.to_value(u.pc), sky.distance.to_value(u.pc), rtol=1e-12)
    # Galactic direction from the torchsky frame engine == astropy
    gal = state.to_frame("GALACTIC")
    ref = sky.galactic
    sep = sep_arcsec(to_np(gal.lon), to_np(gal.lat), ref.l.deg, ref.b.deg)
    assert float(sep.max()) < 5.0e-3, f"galactic: {float(sep.max()):.5f} arcsec"
