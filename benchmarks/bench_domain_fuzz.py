#!/usr/bin/env python3
"""Randomized differential domain fuzz for WCS and generalized HEALPix.

The release gates compare a fixed set of valid-domain samples. This evaluator
searches for the opposite failure mode: a subject that returns finite values
where the independent oracle says the projection domain is invalid, or that
systematically loses valid coverage.

All randomness is seeded and all parameters are recorded in the JSON report.
Rows use ``status: OK|FAIL`` so the evidence claim evaluator can consume them.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECTIONS = (
    "TAN", "SIN", "ARC", "ZEA", "STG", "ZPN", "AZP", "SZP", "AIR", "AIT",
    "MOL", "HPX", "XPH", "CEA", "MER", "CAR", "SFL", "PAR", "COP", "COE",
    "COD", "COO", "BON", "PCO", "NCP",
)
# Angular tolerance for points both implementations call finite. Domain leaks
# are strict; this tolerance only absorbs formula/roundoff differences.
TOL_DEG = 2.0e-5
MIN_COVERAGE = 0.98


def _header(projection: str, rng: np.random.Generator, size: int = 64) -> dict:
    scale = float(rng.uniform(0.001, 0.08))
    rot = float(rng.uniform(0.0, 360.0))
    c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    header = {
        "NAXIS": 2, "NAXIS1": size, "NAXIS2": size,
        "CRPIX1": size / 2.0, "CRPIX2": size / 2.0,
        "CRVAL1": float(rng.uniform(0.0, 360.0)),
        "CRVAL2": float(rng.uniform(-75.0, 75.0)),
        "CD1_1": -scale * c, "CD1_2": scale * s,
        "CD2_1": -scale * s, "CD2_2": -scale * c,
        "CTYPE1": f"RA---{projection}", "CTYPE2": f"DEC--{projection}",
        "CUNIT1": "deg", "CUNIT2": "deg",
    }
    if projection == "HPX":
        header["PV2_1"] = float(rng.integers(2, 7))
        header["PV2_2"] = float(rng.integers(2, 6))
    elif projection == "AIR":
        # WCSLIB's Airy projection requires the theta_b parameter; without it
        # astropy and a native default can legitimately differ, which is not a
        # bug in the subject.
        header["PV2_1"] = float(rng.uniform(30.0, 120.0))
    elif projection == "NCP":
        # NCP is singular on the equator; use a valid oblique reference.
        header["CRVAL2"] = 45.0 * float(rng.choice([-1.0, 1.0]))
        header["CD1_1"] = -0.01
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.01
    elif projection in {"ZPN", "COP", "COE", "COD", "COO", "BON"}:
        header["PV2_1"] = 1.0
    return header


def _points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    # A mix of uniform sky and deliberately out-of-domain / polar points.
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    k = n // 4
    ra[:k] = rng.uniform(0.0, 360.0, k)
    dec[:k] = rng.uniform(80.0, 90.0, k) * rng.choice([-1.0, 1.0], k)
    return np.mod(ra, 360.0), dec


def _pixels(rng: np.random.Generator, n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    # Mostly inside the image, plus a thin band outside every edge to probe the
    # projection domain without wandering arbitrarily far from the footprint.
    x = rng.uniform(-0.25 * size, 1.25 * size, n)
    y = rng.uniform(-0.25 * size, 1.25 * size, n)
    return x, y


def _wcs_row(projection: str, rng: np.random.Generator) -> dict:
    import torch
    from astropy.wcs import WCS as AstropyWCS
    from torchsky.wcs import WCS as TorchWCS

    header = _header(projection, rng)
    x, y = _pixels(rng, 3000, int(header["NAXIS1"]))
    aw = AstropyWCS(header)
    tw = TorchWCS(header)
    ara, adec = aw.all_pix2world(x, y, 0)
    tra, tdec = tw.pixel_to_world(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
    )
    tra = tra.detach().cpu().numpy()
    tdec = tdec.detach().cpu().numpy()
    finite_a = np.isfinite(ara) & np.isfinite(adec)
    finite_t = np.isfinite(tra) & np.isfinite(tdec)
    invented = int(np.count_nonzero(finite_t & ~finite_a))
    coverage = 1.0 if not finite_a.any() else float(np.count_nonzero(finite_t & finite_a) / finite_a.sum())
    both = finite_a & finite_t
    dra = (tra[both] - ara[both] + 180.0) % 360.0 - 180.0
    ddec = tdec[both] - adec[both]
    max_err = float(np.max(np.hypot(dra, ddec))) if both.any() else float("inf")
    status = "OK"
    reasons = []
    if invented:
        reasons.append(f"{invented} invented finite points")
    if coverage < MIN_COVERAGE:
        reasons.append(f"coverage {coverage:.4f} < {MIN_COVERAGE}")
    if max_err > TOL_DEG:
        reasons.append(f"max error {max_err:.3e} deg > {TOL_DEG:.1e}")
    if reasons:
        status = "FAIL"
    return {
        "case": f"wcs_domain_{projection}",
        "status": status,
        "projection": projection,
        "header": header,
        "n_points": int(len(x)),
        "invented_finite": invented,
        "coverage": round(coverage, 6),
        "max_error_deg": max_err,
        "reasons": reasons,
    }


def _hpx_geometry_row(rng: np.random.Generator) -> dict:
    import torch
    from astropy.wcs import WCS as AstropyWCS
    from torchsky.wcs import WCS as TorchWCS

    h = float(rng.integers(2, 8))
    k = float(rng.integers(1, 6))
    header = {
        "NAXIS": 2, "NAXIS1": 256, "NAXIS2": 256,
        "CRPIX1": 128.0, "CRPIX2": 128.0,
        "CRVAL1": 0.0, "CRVAL2": 0.0,
        "CD1_1": -0.01, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 0.01,
        "CTYPE1": "RA---HPX", "CTYPE2": "DEC--HPX",
        "PV2_1": h, "PV2_2": k,
    }
    rng2 = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    ra, dec = _points(rng2, 2000)
    aw = AstropyWCS(header)
    tw = TorchWCS(header)
    ax, ay = aw.all_world2pix(ra, dec, 0)
    tx, ty = tw.world_to_pixel(
        torch.tensor(ra, dtype=torch.float64),
        torch.tensor(dec, dtype=torch.float64),
    )
    tx = tx.detach().cpu().numpy()
    ty = ty.detach().cpu().numpy()
    finite_a = np.isfinite(ax) & np.isfinite(ay)
    finite_t = np.isfinite(tx) & np.isfinite(ty)
    invented = int(np.count_nonzero(finite_t & ~finite_a))
    coverage = 1.0 if not finite_a.any() else float(np.count_nonzero(finite_t & finite_a) / finite_a.sum())
    both = finite_a & finite_t
    max_pix = float(np.max(np.hypot(tx[both] - ax[both], ty[both] - ay[both]))) if both.any() else float("inf")
    reasons = []
    if invented:
        reasons.append(f"{invented} invented finite points")
    if coverage < MIN_COVERAGE:
        reasons.append(f"coverage {coverage:.4f} < {MIN_COVERAGE}")
    if max_pix > 0.01:
        reasons.append(f"max pixel diff {max_pix:.3e} > 1e-2")
    return {
        "case": "hpx_generalized_hk",
        "status": "FAIL" if reasons else "OK",
        "h": h,
        "k": k,
        "n_points": int(len(ra)),
        "invented_finite": invented,
        "coverage": round(coverage, 6),
        "max_pixel_diff": max_pix,
        "reasons": reasons,
    }


def _tpv_forward_row(rng: np.random.Generator) -> dict:
    import torch
    from astropy.wcs import WCS as AstropyWCS
    from torchsky.wcs import WCS as TorchWCS

    scale = float(rng.uniform(1e-5, 1e-3))
    header = {
        "NAXIS": 2, "NAXIS1": 128, "NAXIS2": 128,
        "CRPIX1": 64.0, "CRPIX2": 64.0,
        "CRVAL1": 150.0, "CRVAL2": 2.0,
        "CD1_1": -scale, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": scale,
        "CTYPE1": "RA---TPV", "CTYPE2": "DEC--TPV",
        "PV1_0": 0.0, "PV1_1": 1.0,
        "PV1_4": float(rng.uniform(-5e-4, 5e-4)),
        "PV1_5": float(rng.uniform(-5e-4, 5e-4)),
        "PV1_6": float(rng.uniform(-5e-4, 5e-4)),
        "PV2_0": 0.0, "PV2_1": 1.0, "PV2_2": 0.0,
        "PV2_4": float(rng.uniform(-5e-4, 5e-4)),
        "PV2_5": float(rng.uniform(-5e-4, 5e-4)),
        "PV2_6": float(rng.uniform(-5e-4, 5e-4)),
    }
    aw = AstropyWCS(header)
    tw = TorchWCS(header)
    rng2 = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    xy = rng2.uniform(-64.0, 64.0, size=(2, 3000))
    ax, ay = aw.all_pix2world(xy[0], xy[1], 0)
    tx, ty = tw.pixel_to_world(
        torch.tensor(xy[0], dtype=torch.float64),
        torch.tensor(xy[1], dtype=torch.float64),
    )
    tx = tx.detach().cpu().numpy()
    ty = ty.detach().cpu().numpy()
    finite_a = np.isfinite(ax) & np.isfinite(ay)
    finite_t = np.isfinite(tx) & np.isfinite(ty)
    invented = int(np.count_nonzero(finite_t & ~finite_a))
    coverage = 1.0 if not finite_a.any() else float(np.count_nonzero(finite_t & finite_a) / finite_a.sum())
    both = finite_a & finite_t
    dra = (tx[both] - ax[both] + 180.0) % 360.0 - 180.0 if both.any() else np.array([])
    ddec = ty[both] - ay[both] if both.any() else np.array([])
    max_err = float(np.max(np.hypot(dra, ddec))) if both.any() else float("inf")
    reasons = []
    if invented:
        reasons.append(f"{invented} invented finite points")
    if coverage < MIN_COVERAGE:
        reasons.append(f"coverage {coverage:.4f} < {MIN_COVERAGE}")
    if max_err > TOL_DEG:
        reasons.append(f"max error {max_err:.3e} deg > {TOL_DEG:.1e}")
    return {
        "case": "tpv_forward_random_polynomial",
        "status": "FAIL" if reasons else "OK",
        "header": header,
        "invented_finite": invented,
        "coverage": round(coverage, 6),
        "max_error_deg": max_err,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark_results/domain_fuzz.json"))
    args = parser.parse_args(argv)

    import torch
    import torchsky

    rng = np.random.default_rng(args.seed)
    rows = [_wcs_row(name, rng) for name in PROJECTIONS]
    rows.append(_hpx_geometry_row(rng))
    rows.append(_tpv_forward_row(rng))
    failures = [row["case"] for row in rows if row["status"] != "OK"]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
    sidecar = args.json_out.with_suffix(args.json_out.suffix + ".provenance.json")
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "failures": failures,
        "subject": {
            "torchsky_version": torchsky.__version__,
            "torchsky_import_path": str(Path(torchsky.__file__).resolve()),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "python": sys.version,
        },
    }, indent=2) + "\n")
    for row in rows:
        print(f"{row['case']}: {row['status']} {row.get('reasons', [])}")
    print(f"json: {args.json_out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
