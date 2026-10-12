import numpy as np
import torch
from astropy.time import Time


from torchsky.wcs.temporal import (
    TemporalWCS,
    convert_time_scale,
)
















def test_temporal_utc_leap_day_fraction_matches_astropy_time() -> None:
    mjd_utc = torch.tensor(
        [57203.0, 57203.5, 57203.99999, 57753.0, 57753.5, 57753.99999],
        dtype=torch.float64,
    )
    astropy_utc = Time(mjd_utc.numpy(), format="mjd", scale="utc")

    tai = convert_time_scale(mjd_utc, "UTC", "TAI")
    tt = convert_time_scale(mjd_utc, "UTC", "TT")
    utc_back = convert_time_scale(tai, "TAI", "UTC")

    np.testing.assert_allclose(tai.numpy(), astropy_utc.tai.mjd, atol=2e-12)
    np.testing.assert_allclose(tt.numpy(), astropy_utc.tt.mjd, atol=2e-12)
    np.testing.assert_allclose(utc_back.numpy(), mjd_utc.numpy(), atol=2e-12)


def test_temporal_scale_conversion_matches_astropy_time() -> None:
    mjd_utc = torch.tensor([57204.0, 57754.0, 60000.25], dtype=torch.float64)
    astropy_utc = Time(mjd_utc.numpy(), format="mjd", scale="utc")

    tai = convert_time_scale(mjd_utc, "UTC", "TAI")
    tt = convert_time_scale(mjd_utc, "UTC", "TT")
    utc_back = convert_time_scale(tai, "TAI", "UTC")

    np.testing.assert_allclose(tai.numpy(), astropy_utc.tai.mjd, atol=2e-12)
    np.testing.assert_allclose(tt.numpy(), astropy_utc.tt.mjd, atol=2e-12)
    np.testing.assert_allclose(utc_back.numpy(), mjd_utc.numpy(), atol=2e-12)




def test_temporal_ephemeris_corrections() -> None:
    # Test topocenter (no corrections applied)
    twcs_topo = TemporalWCS({"MJDREF": 60000.0, "TREFPOS": "TOPOCENTER"})
    mjd = torch.tensor([60000.0])
    ra = torch.tensor([120.0])
    dec = torch.tensor([30.0])
    torch.testing.assert_close(twcs_topo.apply_corrections(mjd, ra, dec), mjd)

    # Test BARYCENTER (should apply light travel time corrections)
    twcs_bary = TemporalWCS(
        {
            "MJDREF": 60000.0,
            "TREFPOS": "BARYCENTER",
            "OBSGEO-X": 6378137.0,
            "OBSGEO-Y": 0.0,
            "OBSGEO-Z": 0.0,
        }
    )
    assert twcs_bary.requires_ephemeris
    mjd_corr = twcs_bary.apply_corrections(mjd, ra, dec)

    delta_days = (mjd_corr - mjd).item()
    assert abs(delta_days) < 0.01

    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u
    from astropy.time import Time

    loc = EarthLocation.from_geocentric(6378137.0 * u.m, 0.0 * u.m, 0.0 * u.m)
    t = Time(60000.0, format="mjd", scale="utc", location=loc)
    sc = SkyCoord(ra=120.0 * u.deg, dec=30.0 * u.deg)
    ltt = t.light_travel_time(sc, "barycentric")
    expected_mjd = 60000.0 + ltt.to(u.day).value

    # Keplerian approximation is extremely fast and matches astropy to within 15 seconds
    np.testing.assert_allclose(mjd_corr.item(), expected_mjd, atol=15.0 / 86400.0)




def test_temporal_reference_position_corrections_match_astropy_grid() -> None:
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u

    location = EarthLocation.from_geocentric(
        6378137.0 * u.m,
        0.0 * u.m,
        0.0 * u.m,
    )
    header = {
        "MJDREF": 60000.0,
        "TREFPOS": "BARYCENTER",
        "OBSGEO-X": 6378137.0,
        "OBSGEO-Y": 0.0,
        "OBSGEO-Z": 0.0,
    }
    twcs = TemporalWCS(header)

    mjd_values = torch.tensor([58000.0, 59000.0, 60000.0, 61000.0], dtype=torch.float64)
    ra_values = torch.tensor([0.0, 90.0, 180.0, 270.0], dtype=torch.float64)
    dec_values = torch.tensor([-60.0, 0.0, 45.0, 30.0], dtype=torch.float64)

    corrected = twcs.apply_corrections(mjd_values, ra_values, dec_values)
    expected = []
    for mjd, ra, dec in zip(
        mjd_values.tolist(), ra_values.tolist(), dec_values.tolist()
    ):
        time = Time(mjd, format="mjd", scale="utc", location=location)
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        light_time = time.light_travel_time(coord, "barycentric").to(u.day).value
        expected.append(mjd + light_time)

    np.testing.assert_allclose(
        corrected.numpy(),
        np.asarray(expected, dtype=np.float64),
        atol=5.0 / 86400.0,
    )


def test_temporal_geodetic_observatory_location_matches_astropy_grid() -> None:
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u

    lon = -70.403
    lat = -24.625
    height = 2635.0
    location = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height * u.m)
    twcs = TemporalWCS(
        {
            "MJDREF": 60000.0,
            "TREFPOS": "HELIOCENTER",
            "OBSGEO-L": lon,
            "OBSGEO-B": lat,
            "OBSGEO-H": height,
        }
    )

    mjd_values = torch.tensor([59000.0, 60000.0, 61000.0], dtype=torch.float64)
    ra_values = torch.tensor([15.0, 180.0, 315.0], dtype=torch.float64)
    dec_values = torch.tensor([-20.0, 15.0, 65.0], dtype=torch.float64)

    corrected = twcs.apply_corrections(mjd_values, ra_values, dec_values)
    expected = []
    for mjd, ra, dec in zip(
        mjd_values.tolist(), ra_values.tolist(), dec_values.tolist()
    ):
        time = Time(mjd, format="mjd", scale="utc", location=location)
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        light_time = time.light_travel_time(coord, "heliocentric").to(u.day).value
        expected.append(mjd + light_time)

    np.testing.assert_allclose(
        corrected.numpy(),
        np.asarray(expected, dtype=np.float64),
        atol=5.0 / 86400.0,
    )
