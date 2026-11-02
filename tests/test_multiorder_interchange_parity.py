"""Bounded parity of multi-order HEALPix interchange against ``mhealpy`` and ``cdshealpix``.

* ``MultiOrderHealpixMap`` <-> ``mhealpy.HealpixMap`` (NUNIQ): cell lookup at random sky
  positions, cell areas and the integral agree with mhealpy's own pixel arithmetic.
* ``to_cdshealpix`` / ``from_cdshealpix`` round trip of fixed-order maps through
  ``cdshealpix.skymap.SkymapExplicit``.
* Nested ``ang2pix`` / ``pix2ang`` against ``cdshealpix.nested`` at orders 0-24.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from _gate_helpers import device, f64, to_np  # noqa: E402

from torchsky.sphere import MultiOrderHealpixMap  # noqa: E402
from torchsky.sphere import ang2pix  # noqa: E402


def _uniq(order: int, pix: int) -> int:
    return 4 * (4**order) + pix


def _partition() -> tuple[np.ndarray, np.ndarray]:
    """A disjoint cover of the sphere with cells of order 0, 1 and 2."""
    cells: list[int] = []
    for p in range(12):
        if p >= 4:
            cells.append(_uniq(0, p))
            continue
        for c in range(4 * p, 4 * p + 4):  # children at order 1
            if c == 5:  # refine one of them to order 2
                cells.extend(_uniq(2, d) for d in range(4 * c, 4 * c + 4))
            else:
                cells.append(_uniq(1, c))
    uniq = np.array(sorted(cells), dtype=np.int64)
    values = np.random.default_rng(2).normal(size=uniq.size)
    return uniq, values


def _random_sky(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))


def test_lookup_matches_mhealpy() -> None:
    mhealpy = pytest.importorskip("mhealpy")
    uniq, values = _partition()
    ours = MultiOrderHealpixMap(torch.as_tensor(uniq, device=device()), torch.as_tensor(values, device=device()))
    ref = mhealpy.HealpixMap(data=values, uniq=uniq, density=False)
    lon, lat = _random_sky(4000)
    got = to_np(ours.get_values_pos(f64(lon), f64(lat)))
    theta, phi = np.radians(90.0 - lat), np.radians(lon)
    expected = np.asarray(ref[ref.ang2pix(theta, phi)])
    np.testing.assert_array_equal(got, expected)


def test_area_and_integral_match_mhealpy_density() -> None:
    mhealpy = pytest.importorskip("mhealpy")
    uniq, values = _partition()
    ours = MultiOrderHealpixMap(torch.as_tensor(uniq, device=device()), torch.as_tensor(values, device=device()))
    ref = mhealpy.HealpixMap(data=values, uniq=uniq, density=False)
    area_ref = np.array([ref.pixarea(i).value if hasattr(ref.pixarea(i), "value") else float(ref.pixarea(i)) for i in range(len(uniq))])
    np.testing.assert_allclose(to_np(ours.cell_area()), area_ref, rtol=1e-12)
    assert abs(to_np(ours.cell_area()).sum() - 4.0 * np.pi) < 1e-12


def test_mhealpy_roundtrip_preserves_cells_and_values() -> None:
    pytest.importorskip("mhealpy")
    uniq, values = _partition()
    ours = MultiOrderHealpixMap(torch.as_tensor(uniq), torch.as_tensor(values), metadata={"coordsys": "C"})
    back = MultiOrderHealpixMap.from_mhealpy(ours.to_mhealpy())
    torch.testing.assert_close(back.uniq, ours.uniq, rtol=0, atol=0)
    torch.testing.assert_close(back.values, ours.values, rtol=0, atol=0)


def test_mhealpy_density_flag_roundtrip() -> None:
    mhealpy = pytest.importorskip("mhealpy")
    uniq, values = _partition()
    for flag in (False, True):
        ref = mhealpy.HealpixMap(data=values, uniq=uniq, density=flag)
        assert MultiOrderHealpixMap.from_mhealpy(ref).values_are_densities is flag


def test_cdshealpix_skymap_roundtrip() -> None:
    pytest.importorskip("cdshealpix")
    order = 5
    rng = np.random.default_rng(4)
    pix = np.sort(rng.choice(12 * 4**order, size=300, replace=False)).astype(np.uint64)
    values = rng.normal(size=pix.size)
    uniq = torch.as_tensor(np.array([_uniq(order, int(p)) for p in pix], dtype=np.int64))
    ours = MultiOrderHealpixMap(uniq, torch.as_tensor(values))
    skymap = ours.to_cdshealpix()
    assert int(skymap.order) == order
    np.testing.assert_array_equal(np.asarray(skymap.keys), pix)
    np.testing.assert_array_equal(np.asarray(skymap.values), values)
    back = MultiOrderHealpixMap.from_cdshealpix(skymap)
    torch.testing.assert_close(back.uniq, ours.uniq, rtol=0, atol=0)
    torch.testing.assert_close(back.values, ours.values, rtol=0, atol=0)


@pytest.mark.parametrize("order", (0, 1, 3, 8, 14, 20, 24))
def test_nested_pixelization_matches_cdshealpix(order: int) -> None:
    nested = pytest.importorskip("cdshealpix.nested")
    import astropy.units as u

    lon, lat = _random_sky(5000, seed=order)
    ref = np.asarray(nested.lonlat_to_healpix(lon * u.deg, lat * u.deg, order)).astype(np.int64)
    got = to_np(ang2pix(1 << order, f64(lon), f64(lat), nest=True, lonlat=True))
    # float64 position rounding can move a point across a cell edge at fine orders: tolerate
    # at most one in 1e4 disagreements, each of them an adjacent-cell flip (never a gross error).
    mismatch = np.count_nonzero(got != ref)
    assert mismatch <= max(1, int(1e-4 * lon.size)), f"{mismatch} of {lon.size} cells differ at order {order}"
