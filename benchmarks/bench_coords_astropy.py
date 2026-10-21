#!/usr/bin/env python3
"""Differential parity for torchsky celestial frame conversions vs Astropy.

The observed-place/ERFA path has a separate benchmark. This one covers the
geometric frame graph (ICRS, FK5, Galactic, mean ecliptic of J2000), where a
missing ICRS frame bias is easy to hide inside a sub-arcsecond convention.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _wrap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--tolerance-asec", type=float, default=1.0e-6)
    parser.add_argument(
        "--json-out", type=Path, default=Path("benchmark_results/coords_astropy.json")
    )
    args = parser.parse_args(argv)

    import astropy.units as u
    import torch
    from astropy.coordinates import (
        BarycentricMeanEcliptic,
        FK5,
        Galactic,
        SkyCoord,
    )

    import torchsky
    from torchsky.wcs.frames import convert_celestial

    rng = np.random.default_rng(args.seed)
    ra = rng.uniform(0.0, 360.0, args.n_points)
    dec = rng.uniform(-85.0, 85.0, args.n_points)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    cases = (
        ("icrs_to_galactic", Galactic(), "GALACTIC"),
        ("icrs_to_fk5_j2000", FK5(equinox="J2000"), "FK5"),
        (
            "icrs_to_barycentric_mean_ecliptic_j2000",
            BarycentricMeanEcliptic(equinox="J2000"),
            "ECLIPTIC",
        ),
    )
    rows = []
    for case, frame, torch_frame in cases:
        expected = coords.transform_to(frame)
        lon, lat = convert_celestial(
            torch.tensor(ra), torch.tensor(dec),
            from_frame="ICRS", to_frame=torch_frame,
        )
        lon = lon.numpy()
        lat = lat.numpy()
        err = np.hypot(_wrap(lon, expected.spherical.lon.deg), lat - expected.spherical.lat.deg)
        max_err_asec = float(np.max(err) * 3600.0)
        back_lon, back_lat = convert_celestial(
            torch.tensor(lon), torch.tensor(lat),
            from_frame=torch_frame, to_frame="ICRS",
        )
        back_err = np.hypot(
            _wrap(back_lon.numpy(), ra), back_lat.numpy() - dec
        )
        max_back_asec = float(np.max(back_err) * 3600.0)
        reasons = []
        if max_err_asec > args.tolerance_asec:
            reasons.append(
                f"max astropy separation {max_err_asec:.3e} asec > "
                f"{args.tolerance_asec:.1e}"
            )
        if max_back_asec > args.tolerance_asec:
            reasons.append(f"round-trip {max_back_asec:.3e} asec")
        rows.append(
            {
                "case": case,
                "status": "FAIL" if reasons else "OK",
                "n_points": int(args.n_points),
                "max_astropy_separation_arcsec": max_err_asec,
                "max_roundtrip_arcsec": max_back_asec,
                "tolerance_arcsec": args.tolerance_asec,
                "reasons": reasons,
            }
        )

    failures = [row["case"] for row in rows if row["status"] != "OK"]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
    sidecar = args.json_out.with_suffix(args.json_out.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed,
                "failures": failures,
                "subject": {
                    "torchsky_version": torchsky.__version__,
                    "torchsky_import_path": str(Path(torchsky.__file__).resolve()),
                    "torch_version": torch.__version__,
                },
            },
            indent=2,
        )
        + "\n"
    )
    for row in rows:
        print(
            f"{row['case']}: {row['status']} "
            f"max={row['max_astropy_separation_arcsec']:.3e} asec"
        )
    print(f"json: {args.json_out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
