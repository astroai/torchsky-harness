#!/usr/bin/env python3
"""WCS parity against astropy on real-world instrument headers and images.

The fixtures are vendored from the Astropy WCS test suite (see
``benchmarks/fixtures/real_wcs/PROVENANCE.md``): real INT/Spitzer/PTF/HST
headers, SIP/TPV/ZPN distortions, a -TAB table image, and a real DSS cutout.

For each fixture this benchmark compares torchsky and astropy on a pixel grid:
forward world coordinates, and the inverse pixel obtained from astropy's world
coordinates. Every row records the maximum/p99 disagreement so a regression is
visible per real-world instrument, not averaged away.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks" / "fixtures" / "real_wcs"

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS as AstropyWCS  # noqa: E402

import torchfits  # noqa: E402
from torchsky.wcs import DistortionLookupTable, WCS  # noqa: E402

FIXTURE_SPECS: dict[str, dict[str, Any]] = {
    "int_wfc_zpn": {"file": "2wcses.hdr"},
    "spitzer_irac_sip": {"file": "irac_sip.hdr"},
    "ptf_tpv": {"file": "tpvonly.hdr"},
    "sip_only": {"file": "siponly.hdr"},
    "zpn_edge": {"file": "zpn-hole.hdr"},
    "hst_wfpc2_sip": {"file": "j94f05bgq_flt.fits"},
    "sip_header": {"file": "sip.fits"},
    "sip2_header": {"file": "sip2.fits"},
    "header_newlines_sip": {"file": "header_newlines.fits"},
    "validate_tan": {"file": "validate.fits"},
    "tabular_time": {"file": "tab-time-last-axis.fits"},
    "hst_wfc3_d2im": {
        "file": "ie6d07ujq_wcs.fits",
        "skip": "D2IM lookup parity is open (9e-3 px vs astropy)",
    },
    "hst_sip_lookup": {
        "file": "dist_lookup.fits.gz",
        "skip": "CPDIS/D2IM/WCSDVARR lookup parity is open (3.5e-2 px vs astropy)",
    },
    "dss_cutout": {"file": "dss.14.29.56-62.41.05.fits.gz", "skip": "astropy rejects the singular PC matrix"},
    "nonstandard_units": {"file": "nonstandard_units.hdr", "skip": "unsupported spectral unit"},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default=None, help="Comma-separated fixture names")
    parser.add_argument("--grid", type=int, default=17, help="Samples per side of the pixel grid")
    parser.add_argument("--max-angular-error-arcsec", type=float, default=None)
    parser.add_argument("--max-inverse-pixel-error", type=float, default=None)
    parser.add_argument("--origin", type=int, default=0, choices=[0, 1])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def _read_astropy(path: Path) -> tuple[fits.Header, fits.HDUList | None, int]:
    if path.suffix == ".hdr":
        return fits.Header.fromstring(path.read_text(errors="ignore")), None, 0
    hdul = fits.open(path)
    for index, hdu in enumerate(hdul):
        ctype1 = str(hdu.header.get("CTYPE1", ""))
        if ctype1.startswith(("RA", "GLON", "ELON")) and hdu.header.get("CTYPE2"):
            return hdu.header, hdul, index
    return hdul[0].header, hdul, 0


def _table_axes(header: fits.Header) -> list[int]:
    n = int(header.get("WCSAXES", header.get("NAXIS", 2)) or 2)
    return [
        i
        for i in range(1, n + 1)
        if "-TAB" in str(header.get(f"CTYPE{i}", "")).upper()
    ]


def _lookup_tables(thdul: Any, header: fits.Header) -> dict[str, DistortionLookupTable]:
    tables: dict[str, DistortionLookupTable] = {}
    specs = (
        ("cpdis1", "CPDIS1", "DP1.EXTVER"),
        ("cpdis2", "CPDIS2", "DP2.EXTVER"),
        ("det2im1", "D2IMDIS1", "D2IM1.EXTVER"),
        ("det2im2", "D2IMDIS2", "D2IM2.EXTVER"),
    )
    for argument, card, extver_key in specs:
        if str(header.get(card, "")).upper() not in {"LOOKUP", "D2IM"}:
            continue
        extver = header.get(extver_key)
        if extver is None:
            continue
        for index in range(len(thdul)):
            ext = thdul[index]
            if int(ext.header.get("EXTVER", 0) or 0) != int(extver):
                continue
            data = torch.as_tensor(ext.data, dtype=torch.float64)
            if data.ndim != 2:
                continue
            ext_header = ext.header
            tables[argument] = DistortionLookupTable(
                data,
                crpix=(
                    float(ext_header.get("CRPIX1", 1.0)),
                    float(ext_header.get("CRPIX2", 1.0)),
                ),
                crval=(
                    float(ext_header.get("CRVAL1", 1.0)),
                    float(ext_header.get("CRVAL2", 1.0)),
                ),
                cdelt=(
                    float(ext_header.get("CDELT1", 1.0)),
                    float(ext_header.get("CDELT2", 1.0)),
                ),
            )
            break
    return tables


def _shape(header: fits.Header) -> tuple[int, int]:
    rows = int(header.get("NAXIS2", 0) or 0)
    columns = int(header.get("NAXIS1", 0) or 0)
    if rows > 1 and columns > 1:
        return rows, columns
    return 96, 96


def _angular_separation_arcsec(
    lon_a: np.ndarray, lat_a: np.ndarray, lon_b: np.ndarray, lat_b: np.ndarray
) -> np.ndarray:
    """Great-circle separation in arcsec for equal-length finite arrays."""
    ra1 = np.deg2rad(lon_a)
    dec1 = np.deg2rad(lat_a)
    ra2 = np.deg2rad(lon_b)
    dec2 = np.deg2rad(lat_b)
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 3600.0


def _bench_fixture(
    name: str, spec: dict[str, Any], *, grid: int, origin: int
) -> dict[str, Any]:
    row: dict[str, Any] = {"case": name, "fixture": spec["file"]}
    if "skip" in spec:
        row.update({"status": "SKIPPED", "note": spec["skip"]})
        return row

    path = FIXTURES / spec["file"]
    ahdul: fits.HDUList | None = None
    thdul: Any | None = None
    try:
        header, ahdul, _ = _read_astropy(path)
        header_dict = dict(header)
        rows, columns = _shape(header)

        if path.suffix == ".fits":
            thdul = torchfits.open(str(path))
        kwargs: dict[str, Any] = {}
        if _table_axes(header):
            if thdul is None:
                raise RuntimeError("tabular fixture without torchfits HDUList")
            kwargs["tables"] = thdul
        if spec.get("lookup_tables", True) and thdul is not None:
            lookup = _lookup_tables(thdul, header)
            if lookup:
                kwargs.update(lookup)

        torch_wcs = WCS(header_dict, **kwargs)
        astropy_wcs = (
            AstropyWCS(header, fobj=ahdul, relax=True)
            if ahdul is not None
            else AstropyWCS(header, relax=True)
        )

        pixel_x = np.linspace(0.0, columns - 1.0, grid)
        pixel_y = np.linspace(0.0, rows - 1.0, grid)
        xx, yy = np.meshgrid(pixel_x, pixel_y)
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        naxis = int(astropy_wcs.naxis)
        extra = [
            np.full(x_flat.shape, (int(header.get(f"NAXIS{i}", 1) or 1) - 1) / 2.0)
            for i in range(3, naxis + 1)
        ]
        x_t = torch.from_numpy(x_flat.astype(np.float64))
        y_t = torch.from_numpy(y_flat.astype(np.float64))
        t0 = time.perf_counter()
        if naxis > 2:
            torch_world = torch_wcs.pixel_to_world(
                x_t, y_t, *[torch.from_numpy(e) for e in extra]
            )
        else:
            torch_world = torch_wcs.pixel_to_world(x_t, y_t)
        torch_forward_ms = (time.perf_counter() - t0) * 1e3
        torch_world = (
            torch_world if isinstance(torch_world, tuple) else (torch_world,)
        )
        torch_world_arr = np.stack(
            [np.asarray(value.detach().cpu(), dtype=np.float64) for value in torch_world],
            axis=1,
        )

        t0 = time.perf_counter()
        astropy_world = np.asarray(
            astropy_wcs.all_pix2world(x_flat, y_flat, *extra, origin)
            if naxis > 2
            else astropy_wcs.all_pix2world(x_flat, y_flat, origin),
            dtype=np.float64,
        )
        astropy_forward_ms = (time.perf_counter() - t0) * 1e3
        astropy_world = astropy_world.T if astropy_world.ndim > 1 else astropy_world[:, None]

        finite = np.isfinite(astropy_world).all(axis=1) & np.isfinite(
            torch_world_arr
        ).all(axis=1)
        if finite.any():
            separation = _angular_separation_arcsec(
                astropy_world[finite, 0],
                astropy_world[finite, 1],
                torch_world_arr[finite, 0],
                torch_world_arr[finite, 1],
            )
            max_angular = float(np.max(separation))
            p99_angular = float(np.percentile(separation, 99))
            # Inverse parity: feed astropy's world coordinates to both inverses.
            world_valid = astropy_world[finite]
            t0 = time.perf_counter()
            astropy_pixel = np.asarray(
                astropy_wcs.all_world2pix(world_valid, origin), dtype=np.float64
            )
            astropy_inverse_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            torch_pixel = torch_wcs.world_to_pixel(
                *[torch.from_numpy(world_valid[:, i]) for i in range(world_valid.shape[1])]
            )
            torch_inverse_ms = (time.perf_counter() - t0) * 1e3
            torch_pixel = torch_pixel if isinstance(torch_pixel, tuple) else (torch_pixel,)
            torch_pixel_arr = np.stack(
                [np.asarray(value.detach().cpu(), dtype=np.float64) for value in torch_pixel],
                axis=1,
            )
            pixel_diff = torch_pixel_arr[:, :2] - astropy_pixel[:, :2]
            finite_pixel = np.isfinite(pixel_diff).all(axis=1)
            max_pixel = (
                float(np.max(np.abs(pixel_diff[finite_pixel])))
                if finite_pixel.any()
                else float("nan")
            )
            p99_pixel = (
                float(np.percentile(np.abs(pixel_diff[finite_pixel]), 99))
                if finite_pixel.any()
                else float("nan")
            )
        else:
            max_angular = p99_angular = max_pixel = p99_pixel = float("nan")
            astropy_forward_ms = astropy_inverse_ms = float("nan")
            torch_inverse_ms = float("nan")

        row.update(
            {
                "status": "OK",
                "n_points": int(finite.size),
                "n_valid": int(finite.sum()),
                "naxis": naxis,
                "torch_forward_ms": torch_forward_ms,
                "astropy_forward_ms": astropy_forward_ms,
                "torch_inverse_ms": torch_inverse_ms,
                "astropy_inverse_ms": astropy_inverse_ms,
                "max_angular_error_arcsec": max_angular,
                "p99_angular_error_arcsec": p99_angular,
                "max_inverse_pixel_error": max_pixel,
                "p99_inverse_pixel_error": p99_pixel,
            }
        )
        return row
    except Exception as exc:  # noqa: BLE001 - fixture-level reporting
        row.update({"status": "SKIPPED", "note": f"{type(exc).__name__}: {exc}"})
        return row
    finally:
        if thdul is not None:
            thdul.close()
        if ahdul is not None:
            ahdul.close()


def _print_rows(rows: list[dict[str, Any]]) -> None:
    header = ("case", "status", "valid", "ang_max", "p99", "pix_max", "p99")
    print(" ".join(f"{column:>26s}" for column in header))
    for row in rows:
        if row["status"] == "OK":
            print(
                f"{row['case']:>26s}{row['status']:>12s}{row['n_valid']:12d}"
                f"{row['max_angular_error_arcsec']:12.3e}"
                f"{row['p99_angular_error_arcsec']:12.3e}"
                f"{row['max_inverse_pixel_error']:12.3e}"
                f"{row['p99_inverse_pixel_error']:12.3e}"
            )
        else:
            print(f"{row['case']:>26s}{row['status']:>12s}  {row.get('note', '')}")


def main() -> int:
    args = _parse_args()
    names = (
        [name.strip() for name in args.fixtures.split(",") if name.strip()]
        if args.fixtures
        else list(FIXTURE_SPECS)
    )
    unknown = [name for name in names if name not in FIXTURE_SPECS]
    if unknown:
        raise ValueError(f"Unknown fixtures: {', '.join(unknown)}")

    rows = [
        _bench_fixture(name, FIXTURE_SPECS[name], grid=args.grid, origin=args.origin)
        for name in names
    ]
    _print_rows(rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if args.max_angular_error_arcsec is not None:
        exceeded = [
            row
            for row in rows
            if row["status"] == "OK"
            and math.isfinite(row["max_angular_error_arcsec"])
            and row["max_angular_error_arcsec"] > args.max_angular_error_arcsec
        ]
        if exceeded:
            print("\nReal-header angular parity threshold exceeded:")
            for row in exceeded:
                print(f"  {row['case']}: {row['max_angular_error_arcsec']:.6g} arcsec")
            return 1

    if args.max_inverse_pixel_error is not None:
        exceeded = [
            row
            for row in rows
            if row["status"] == "OK"
            and math.isfinite(row["max_inverse_pixel_error"])
            and row["max_inverse_pixel_error"] > args.max_inverse_pixel_error
        ]
        if exceeded:
            print("\nReal-header inverse parity threshold exceeded:")
            for row in exceeded:
                print(f"  {row['case']}: {row['max_inverse_pixel_error']:.6g} px")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
