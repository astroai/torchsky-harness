from pathlib import Path

import numpy as np
import torch
import pytest

from torchsky.sphere.geom import (
    SphericalPolygon,
    spherical_polygon_contains,
)

try:
    from spherical_geometry.polygon import SphericalPolygon as SGP  # type: ignore
except Exception:  # pragma: no cover - optional comparator
    SGP = None

try:
    from spherical_geometry.tests.helpers import ROOT_DIR as SG_ROOT_DIR  # type: ignore
except Exception:  # pragma: no cover - optional comparator fixtures
    SG_ROOT_DIR = None


def _has_cpp_geom() -> bool:
    try:
        from torchsky import cpp as _cpp_mod  # type: ignore
    except Exception:
        return False
    return hasattr(_cpp_mod, "_sphere_planar_point_in_polygon_cpu") and hasattr(
        _cpp_mod, "_sphere_arc_intersections_matrix_cpu"
    )


def _concave_polygon() -> tuple[torch.Tensor, torch.Tensor]:
    # Concave L-shaped polygon near the equator.
    lon = torch.tensor([-20.0, 20.0, 20.0, 0.0, 0.0, -20.0], dtype=torch.float64)
    lat = torch.tensor([-20.0, -20.0, 0.0, 0.0, 20.0, 20.0], dtype=torch.float64)
    return lon, lat


def _strip_closed(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if lon.size >= 4 and np.isclose(lon[0], lon[-1]) and np.isclose(lat[0], lat[-1]):
        return lon[:-1], lat[:-1]
    return lon, lat


def _xy_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    lon = np.remainder(np.degrees(np.arctan2(y, x)), 360.0)
    lat = np.degrees(np.arcsin(np.clip(z / np.linalg.norm(xyz, axis=1), -1.0, 1.0)))
    return lon.astype(np.float64), lat.astype(np.float64)


def _load_difficult_pairs(
    max_pairs: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if SG_ROOT_DIR is None:
        return []
    path = Path(SG_ROOT_DIR) / "difficult_intersections.txt"
    if not path.exists():
        return []

    raw = path.read_bytes().splitlines()
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(0, len(raw), 4):
        if len(pairs) >= max_pairs or i + 3 >= len(raw):
            break
        a_points = np.frombuffer(
            bytes.fromhex(raw[i].decode("ascii")), dtype="<f8"
        ).reshape((-1, 3))
        a_inside = np.frombuffer(
            bytes.fromhex(raw[i + 1].decode("ascii")), dtype="<f8"
        ).reshape((-1, 3))
        b_points = np.frombuffer(
            bytes.fromhex(raw[i + 2].decode("ascii")), dtype="<f8"
        ).reshape((-1, 3))
        b_inside = np.frombuffer(
            bytes.fromhex(raw[i + 3].decode("ascii")), dtype="<f8"
        ).reshape((-1, 3))
        pairs.append((a_points, a_inside, b_points, b_inside))
    return pairs








def _geom_accel_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


























@pytest.mark.skipif(SGP is None, reason="spherical-geometry not available")
def test_spherical_polygon_contains_matches_spherical_geometry() -> None:
    lon_poly = np.array(
        [106.0, 114.0, 130.0, 114.0, 106.0, 98.0, 82.0, 98.0, 106.0], dtype=np.float64
    )
    lat_poly = np.array(
        [-20.0, -15.0, -20.0, -25.0, -32.0, -25.0, -20.0, -15.0, -20.0],
        dtype=np.float64,
    )
    rng = np.random.default_rng(123)
    q_lon = rng.uniform(70.0, 140.0, size=2000)
    q_lat = rng.uniform(-40.0, -8.0, size=2000)

    if hasattr(SGP, "from_lonlat"):
        sg_poly = SGP.from_lonlat(lon_poly, lat_poly, degrees=True)
        sg_contains = np.asarray(
            [
                sg_poly.contains_lonlat(float(lo), float(la), degrees=True)
                for lo, la in zip(q_lon, q_lat, strict=False)
            ],
            dtype=bool,
        )
    elif hasattr(SGP, "from_radec"):
        sg_poly = SGP.from_radec(lon_poly, lat_poly, degrees=True)
        sg_contains = np.asarray(
            [
                sg_poly.contains_radec(float(lo), float(la), degrees=True)
                for lo, la in zip(q_lon, q_lat, strict=False)
            ],
            dtype=bool,
        )
    else:
        pytest.skip("unsupported spherical-geometry API")

    tf_contains = (
        spherical_polygon_contains(
            torch.from_numpy(q_lon),
            torch.from_numpy(q_lat),
            torch.from_numpy(lon_poly),
            torch.from_numpy(lat_poly),
            inclusive=False,
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(tf_contains, sg_contains)


@pytest.mark.skipif(
    SGP is None or SG_ROOT_DIR is None,
    reason="spherical-geometry fixtures not available",
)
def test_spherical_polygon_intersects_matches_upstream_difficult_pairs() -> None:
    difficult_pairs = _load_difficult_pairs(max_pairs=6)
    if not difficult_pairs:
        pytest.skip("upstream difficult_intersections fixtures unavailable")

    for a_points, a_inside, b_points, b_inside in difficult_pairs:
        sg_a = SGP(a_points, a_inside)
        sg_b = SGP(b_points, b_inside)

        lon_a, lat_a = _xy_to_lonlat(a_points)
        lon_b, lat_b = _xy_to_lonlat(b_points)
        inside_a_lon, inside_a_lat = _xy_to_lonlat(a_inside[:1])
        inside_b_lon, inside_b_lat = _xy_to_lonlat(b_inside[:1])
        lon_a, lat_a = _strip_closed(lon_a, lat_a)
        lon_b, lat_b = _strip_closed(lon_b, lat_b)

        tf_a = SphericalPolygon(
            torch.from_numpy(lon_a),
            torch.from_numpy(lat_a),
            inside_lon_deg=float(inside_a_lon[0]),
            inside_lat_deg=float(inside_a_lat[0]),
        )
        tf_b = SphericalPolygon(
            torch.from_numpy(lon_b),
            torch.from_numpy(lat_b),
            inside_lon_deg=float(inside_b_lon[0]),
            inside_lat_deg=float(inside_b_lat[0]),
        )

        assert tf_a.intersects(tf_b) == bool(sg_a.intersects_poly(sg_b))


@pytest.mark.skipif(
    SGP is None or SG_ROOT_DIR is None,
    reason="spherical-geometry fixtures not available",
)
@pytest.mark.performance
def test_upstream_polygon_replay_gate_small() -> None:

    from torchsky_harness.benchmarks.replays.replay_upstream_spherical_geometry_polygons import (
        ReplaySettings,
        run_replay,
    )

    payload = run_replay(
        ReplaySettings(
            n_points=4_000,
            runs=1,
            seed=123,
            intersection_nside=256,
            max_difficult_pairs=1,
            max_contains_mismatches_nonboundary=0,
            max_pair_intersection_mismatches=0,
            max_difficult_nonempty_mismatches=0,
            max_area_rel_error=1e-10,
            difficult_area_nsides=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
            max_difficult_area_rel_error_final=0.12,
            max_difficult_area_convergence_rel=0.10,
            allow_nonrelease_distributions=True,
        )
    )
    assert payload["failures"] == []
