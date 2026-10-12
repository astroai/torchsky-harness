import numpy as np
import torch
import pytest

from torchsky.sphere import compat
from torchsky.sphere import healpix as hp














@pytest.mark.parametrize("nest", [False, True])
def test_compat_ang2pix_pix2ang_match_hpgeom(nest: bool) -> None:
    hpgeom = pytest.importorskip("hpgeom")
    nside = 32
    lon_np = np.array([0.25, 17.5, 92.25, 181.0, 359.1], dtype=np.float64)
    lat_np = np.array([-63.0, -12.5, 0.75, 34.0, 71.25], dtype=np.float64)
    lon = torch.from_numpy(lon_np)
    lat = torch.from_numpy(lat_np)

    got_pix = compat.ang2pix(nside, lon, lat, nest=nest, lonlat=True)
    expected_pix = torch.as_tensor(
        hpgeom.angle_to_pixel(
            nside, lon_np, lat_np, nest=nest, lonlat=True, degrees=True
        ),
        dtype=torch.int64,
    )
    torch.testing.assert_close(got_pix, expected_pix)

    got_lon, got_lat = compat.pix2ang(nside, got_pix, nest=nest, lonlat=True)
    expected_lon_np, expected_lat_np = hpgeom.pixel_to_angle(
        nside, expected_pix.numpy(), nest=nest, lonlat=True, degrees=True
    )
    torch.testing.assert_close(
        got_lon, torch.as_tensor(expected_lon_np), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        got_lat, torch.as_tensor(expected_lat_np), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("nest", [False, True])
def test_compat_ang2pix_matches_astropy_healpix(nest: bool) -> None:
    pytest.importorskip("astropy_healpix")
    units = pytest.importorskip("astropy.units")
    from astropy_healpix import HEALPix

    nside = 16
    lon_np = np.array([3.0, 45.5, 127.25, 250.0, 315.75], dtype=np.float64)
    lat_np = np.array([-50.0, -5.5, 22.0, 48.75, 80.0], dtype=np.float64)
    got = compat.ang2pix(
        nside,
        torch.from_numpy(lon_np),
        torch.from_numpy(lat_np),
        nest=nest,
        lonlat=True,
    )
    order = "nested" if nest else "ring"
    expected = HEALPix(nside=nside, order=order).lonlat_to_healpix(
        lon_np * units.deg,
        lat_np * units.deg,
    )
    torch.testing.assert_close(got, torch.as_tensor(expected, dtype=torch.int64))


































def test_compat_pixel_range_helpers_match_hpgeom() -> None:
    hpgeom = pytest.importorskip("hpgeom")
    ranges = np.array([[0, 3], [7, 8], [20, 24]], dtype=np.int64)

    got_exclusive = compat.pixel_ranges_to_pixels(ranges, inclusive=False)
    expected_exclusive = torch.as_tensor(
        hpgeom.pixel_ranges_to_pixels(ranges, inclusive=False), dtype=torch.int64
    )
    torch.testing.assert_close(got_exclusive, expected_exclusive)

    got_inclusive = compat.pixel_ranges_to_pixels(ranges, inclusive=True)
    expected_inclusive = torch.as_tensor(
        hpgeom.pixel_ranges_to_pixels(ranges, inclusive=True), dtype=torch.int64
    )
    torch.testing.assert_close(got_inclusive, expected_inclusive)

    upgraded = compat.upgrade_pixel_ranges(2, ranges, 16)
    expected_upgraded = torch.as_tensor(
        hpgeom.upgrade_pixel_ranges(2, ranges, 16), dtype=torch.int64
    )
    torch.testing.assert_close(upgraded, expected_upgraded)


@pytest.mark.parametrize("nest", [False, True])
def test_compat_upgrade_pixels_match_hpgeom(nest: bool) -> None:
    hpgeom = pytest.importorskip("hpgeom")
    if not hasattr(hpgeom, "upgrade_pixels"):
        pytest.skip("hpgeom does not expose upgrade_pixels")

    pixels_np = np.array([2, 5, 18], dtype=np.int64)
    pixels = torch.from_numpy(pixels_np)
    got = compat.upgrade_pixels(4, pixels, 16, nest=nest)
    expected = torch.as_tensor(
        hpgeom.upgrade_pixels(4, pixels_np, 16, nest=nest),
        dtype=torch.int64,
    )
    if not nest:
        expected = torch.unique(expected, sorted=True)
    torch.testing.assert_close(got, expected)




def test_compat_query_circle_pixel_ranges_match_hpgeom_expansion() -> None:
    hpgeom = pytest.importorskip("hpgeom")
    nside = 32
    lon = 42.0
    lat = -12.5
    radius = 3.0

    got_ranges = compat.query_circle(
        nside,
        lon,
        lat,
        radius,
        nest=True,
        inclusive=False,
        degrees=True,
        return_pixel_ranges=True,
    )
    got_pixels = compat.pixel_ranges_to_pixels(got_ranges, inclusive=False)
    expected_pixels = torch.as_tensor(
        hpgeom.query_circle(
            nside,
            lon,
            lat,
            radius,
            nest=True,
            inclusive=False,
            degrees=True,
        ),
        dtype=torch.int64,
    )

    torch.testing.assert_close(
        torch.sort(got_pixels).values,
        torch.sort(expected_pixels).values,
    )


def test_compat_fit_remove_monopole_matches_healpy() -> None:
    healpy = pytest.importorskip("healpy")
    nside = 8
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    _, lat = hp.pix2ang(nside, pix, nest=False, lonlat=True)
    m = (1.5 + 0.01 * lat).to(torch.float64)
    m[5] = hp.UNSEEN

    got = compat.fit_monopole(m, nest=False, bad=hp.UNSEEN, gal_cut=10.0)
    exp = torch.tensor(
        float(healpy.fit_monopole(m.numpy(), nest=False, bad=hp.UNSEEN, gal_cut=10.0)),
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)

    got_removed, got_fit = compat.remove_monopole(
        m, nest=False, bad=hp.UNSEEN, gal_cut=10.0, fitval=True, copy=True
    )
    exp_removed, exp_fit = healpy.remove_monopole(
        m.numpy(),
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=10.0,
        fitval=True,
        copy=True,
        verbose=False,
    )
    torch.testing.assert_close(
        got_fit, torch.tensor(float(exp_fit), dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        got_removed,
        torch.as_tensor(exp_removed, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_compat_fit_remove_dipole_matches_healpy() -> None:
    healpy = pytest.importorskip("healpy")
    nside = 8
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    x, y, z = hp.pix2vec(nside, pix, nest=False)
    m = (1.1 + 0.2 * x - 0.05 * y + 0.08 * z).to(torch.float64)
    m[7] = hp.UNSEEN

    got_mono, got_dip = compat.fit_dipole(m, nest=False, bad=hp.UNSEEN, gal_cut=12.0)
    exp_mono, exp_dip = healpy.fit_dipole(
        m.numpy(), nest=False, bad=hp.UNSEEN, gal_cut=12.0
    )
    torch.testing.assert_close(
        got_mono,
        torch.tensor(float(exp_mono), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        got_dip, torch.as_tensor(exp_dip, dtype=torch.float64), atol=1e-12, rtol=0.0
    )

    got_removed, got_mono2, got_dip2 = compat.remove_dipole(
        m,
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=12.0,
        fitval=True,
        copy=True,
    )
    exp_removed, exp_mono2, exp_dip2 = healpy.remove_dipole(
        m.numpy(),
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=12.0,
        fitval=True,
        copy=True,
        verbose=False,
    )
    exp_removed_t = torch.as_tensor(exp_removed.filled(hp.UNSEEN), dtype=torch.float64)
    torch.testing.assert_close(got_removed, exp_removed_t, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        got_mono2,
        torch.tensor(float(exp_mono2), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        got_dip2, torch.as_tensor(exp_dip2, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
