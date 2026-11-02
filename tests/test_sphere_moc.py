
import numpy as np
import pytest
import torch
from torchsky.sphere import healpix as hp
from torchsky.sphere import MOC, nest2uniq








































































def test_mocpy_contains_union_area_and_polygon_parity() -> None:
    mocpy = pytest.importorskip("mocpy")
    import astropy.units as u

    lon = torch.tensor([10.0, 12.0, 11.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float64)
    ts_poly = MOC.from_polygon(lon, lat, max_order=6)

    mocpy_cls = mocpy.MOC
    if hasattr(mocpy_cls, "from_polygon"):
        ref = mocpy_cls.from_polygon(
            lon.numpy() * u.deg, lat.numpy() * u.deg, max_depth=6
        )
    else:
        pytest.skip("mocpy.MOC.from_polygon unavailable")

    qlon = torch.tensor([11.0, 50.0], dtype=torch.float64)
    qlat = torch.tensor([0.4, 40.0], dtype=torch.float64)
    got = ts_poly.contains(qlon, qlat).cpu().numpy()
    if hasattr(ref, "contains_lonlat"):
        exp = np.asarray(ref.contains_lonlat(qlon.numpy() * u.deg, qlat.numpy() * u.deg))
    elif hasattr(ref, "contains"):
        exp = np.asarray(ref.contains(qlon.numpy() * u.deg, qlat.numpy() * u.deg))
    else:
        pytest.skip("mocpy contains API unavailable")
    np.testing.assert_array_equal(got.reshape(-1), np.asarray(exp).reshape(-1))

    a = MOC.from_circle(0.0, 0.0, 2.0, max_order=5)
    b = MOC.from_circle(3.0, 0.0, 2.0, max_order=5)
    u_ts = a.union(b)
    i_ts = a.intersection(b)
    four_pi = 4.0 * float(np.pi)
    if hasattr(mocpy_cls, "from_cone"):
        ra = mocpy_cls.from_cone(
            lon=0.0 * u.deg, lat=0.0 * u.deg, radius=2.0 * u.deg, max_depth=5
        )
        rb = mocpy_cls.from_cone(
            lon=3.0 * u.deg, lat=0.0 * u.deg, radius=2.0 * u.deg, max_depth=5
        )
        uref = ra.union(rb)
        iref = ra.intersection(rb)
        if hasattr(uref, "sky_fraction"):
            assert abs(u_ts.area / four_pi - float(uref.sky_fraction)) < 5e-3
        if hasattr(iref, "sky_fraction"):
            assert abs(i_ts.area / four_pi - float(iref.sky_fraction)) < 5e-3


def test_mocpy_fits_roundtrip_optional(tmp_path) -> None:
    mocpy = pytest.importorskip("mocpy")
    moc = MOC.from_circle(20.0, -10.0, 1.5, max_order=6)
    path = tmp_path / "moc.fits"
    moc.write_fits(str(path), overwrite=True)
    if not hasattr(mocpy.MOC, "from_fits"):
        pytest.skip("mocpy.MOC.from_fits unavailable")
    loaded = mocpy.MOC.from_fits(str(path))
    assert loaded is not None
    back = MOC.read_fits(str(path))
    torch.testing.assert_close(back.uniq.sort().values, moc.uniq.sort().values)




def _leaf_circle_moc(
    lon_deg: float,
    lat_deg: float,
    radius_deg: float,
    max_order: int,
    *,
    inclusive: bool = True,
) -> MOC:
    nside = 1 << max_order
    pix = hp.query_circle(
        nside,
        lon_deg,
        lat_deg,
        radius_deg,
        nest=True,
        degrees=True,
        inclusive=inclusive,
    )
    return MOC(nest2uniq(nside, pix))
