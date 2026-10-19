#!/usr/bin/env python3
"""Sphere/HEALPix parity on a real-sky map derived from CFHT MegaPipe.

The fixture (``benchmarks/fixtures/real_sphere``) contains per-pixel means from
the actual 2 GB CFHT MegaPipe g-band mosaic and a sample of real sky
positions, derived by ``scripts/derive_real_sphere_fixture.py``. This benchmark
compares torchsky against healpy on that real map for the operations a survey
pipeline uses: pixel lookup, interpolation, ud_grade, SHT analysis/synthesis,
power spectra, and smoothing.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "real_sphere" / "cfht_megapipe_g_nside512.npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=64)
    parser.add_argument("--fwhm-deg", type=float, default=1.0)
    parser.add_argument("--max-relative-error", type=float, default=None)
    parser.add_argument("--max-ud-grade-absolute-error", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _relative(error: float, reference: float) -> float:
    scale = max(abs(float(reference)), 1.0e-30)
    return abs(float(error)) / scale


def main() -> int:
    args = _parse_args()
    try:
        import healpy as hp
        import torch
        import torchsky.sphere as sphere
    except ImportError as exc:  # pragma: no cover - benchmark environment
        raise SystemExit(f"bench_sphere_real_map requires healpy and torchsky: {exc}")

    data = np.load(FIXTURE)
    nside = int(data["nside"])
    npix = hp.nside2npix(nside)
    values = np.zeros(npix, dtype=np.float64)
    values[data["pix"]] = data["value"]
    nnz = int((values != 0.0).sum())
    map_t = torch.from_numpy(values)
    rows: list[dict[str, Any]] = []

    # 1. Real position -> HEALPix pixel.
    ra = data["sample_ra"]
    dec = data["sample_dec"]
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    torch_pix = sphere.ang2pix(
        nside,
        torch.from_numpy(theta.astype(np.float64)),
        torch.from_numpy(phi.astype(np.float64)),
    ).numpy()
    healpy_pix = hp.ang2pix(nside, theta, phi)
    mismatches = int((torch_pix != healpy_pix).sum())
    rows.append(
        {
            "case": "ang2pix_real",
            "n_points": int(ra.size),
            "mismatches": mismatches,
            "status": "OK" if mismatches == 0 else "FAIL",
        }
    )

    # 2. Real-position interpolation on the real map.
    t0 = time.perf_counter()
    torch_interp = sphere.get_interp_val(
        map_t, torch.from_numpy(theta), torch.from_numpy(phi)
    ).numpy()
    torch_interp_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    healpy_interp = hp.get_interp_val(values, theta, phi)
    healpy_interp_ms = (time.perf_counter() - t0) * 1e3
    interp_abs = float(np.max(np.abs(torch_interp - healpy_interp)))
    rows.append(
        {
            "case": "interp_real",
            "n_points": int(ra.size),
            "max_abs_diff": interp_abs,
            "max_rel_diff": _relative(interp_abs, np.max(np.abs(healpy_interp))),
            "torch_ms": torch_interp_ms,
            "healpy_ms": healpy_interp_ms,
            "status": "OK",
        }
    )

    # 3. ud_grade of the real map.
    for nside_out in (256, 128):
        torch_map = sphere.ud_grade(map_t, nside_out).numpy()
        healpy_map = hp.ud_grade(values, nside_out)
        abs_diff = float(np.max(np.abs(torch_map - healpy_map)))
        rows.append(
            {
                "case": f"ud_grade_{nside}_{nside_out}",
                "n_points": int(healpy_map.size),
                "max_abs_diff": abs_diff,
                "max_rel_diff": _relative(abs_diff, np.max(np.abs(healpy_map))),
                "status": "OK",
            }
        )

    # 4. Spherical harmonic transforms on the real map.
    lmax = int(args.lmax)
    t0 = time.perf_counter()
    torch_alm = sphere.map2alm(map_t, nside=nside, lmax=lmax).numpy()
    torch_map2alm_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    healpy_alm = hp.map2alm(values, lmax=lmax)
    healpy_map2alm_ms = (time.perf_counter() - t0) * 1e3
    alm_abs = float(np.max(np.abs(torch_alm - healpy_alm)))
    rows.append(
        {
            "case": "map2alm_real",
            "n_points": int(healpy_alm.size),
            "max_abs_diff": alm_abs,
            "max_rel_diff": _relative(alm_abs, np.max(np.abs(healpy_alm))),
            "torch_ms": torch_map2alm_ms,
            "healpy_ms": healpy_map2alm_ms,
            "status": "OK",
        }
    )

    t0 = time.perf_counter()
    torch_map = sphere.alm2map(
        torch.from_numpy(healpy_alm), nside, lmax=lmax
    ).numpy()
    torch_alm2map_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    healpy_map = hp.alm2map(healpy_alm, nside, lmax=lmax)
    healpy_alm2map_ms = (time.perf_counter() - t0) * 1e3
    map_abs = float(np.max(np.abs(torch_map - healpy_map)))
    rows.append(
        {
            "case": "alm2map_real",
            "n_points": int(healpy_map.size),
            "max_abs_diff": map_abs,
            "max_rel_diff": _relative(map_abs, np.max(np.abs(healpy_map))),
            "torch_ms": torch_alm2map_ms,
            "healpy_ms": healpy_alm2map_ms,
            "status": "OK",
        }
    )

    torch_cl = sphere.alm2cl(torch.from_numpy(healpy_alm), lmax=lmax).numpy()
    healpy_cl = hp.alm2cl(healpy_alm, lmax=lmax)
    cl_abs = float(np.max(np.abs(torch_cl - healpy_cl)))
    rows.append(
        {
            "case": "alm2cl_real",
            "n_points": int(healpy_cl.size),
            "max_abs_diff": cl_abs,
            "max_rel_diff": _relative(cl_abs, np.max(np.abs(healpy_cl))),
            "status": "OK",
        }
    )

    # 5. Smoothing the real map.
    fwhm_rad = math.radians(float(args.fwhm_deg))
    t0 = time.perf_counter()
    torch_smooth = sphere.smoothmap(
        map_t, nside=nside, lmax=lmax, fwhm_rad=fwhm_rad
    ).numpy()
    torch_smooth_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    healpy_smooth = hp.smoothing(values, fwhm=fwhm_rad, lmax=lmax)
    healpy_smooth_ms = (time.perf_counter() - t0) * 1e3
    smooth_abs = float(np.max(np.abs(torch_smooth - healpy_smooth)))
    rows.append(
        {
            "case": "smooth_real",
            "n_points": int(healpy_smooth.size),
            "max_abs_diff": smooth_abs,
            "max_rel_diff": _relative(smooth_abs, np.max(np.abs(healpy_smooth))),
            "torch_ms": torch_smooth_ms,
            "healpy_ms": healpy_smooth_ms,
            "status": "OK",
        }
    )

    for row in rows:
        row["nside"] = nside
        row["fixture_nonzero_pixels"] = nnz

    print(f"{'case':22s} {'status':8s} {'abs':>12s} {'rel':>12s}")
    for row in rows:
        if "max_abs_diff" in row:
            print(
                f"{row['case']:22s} {row['status']:8s} "
                f"{row['max_abs_diff']:12.3e} {row['max_rel_diff']:12.3e}"
            )
        else:
            print(f"{row['case']:22s} {row['status']:8s} mismatches={row['mismatches']}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    failures: list[str] = []
    ug_abs = max(
        (row["max_abs_diff"] for row in rows if row["case"].startswith("ud_grade")),
        default=0.0,
    )
    if (
        args.max_ud_grade_absolute_error is not None
        and ug_abs > args.max_ud_grade_absolute_error
    ):
        failures.append(f"ud_grade max abs diff {ug_abs:.3e}")
    if args.max_relative_error is not None:
        scaled = [
            row
            for row in rows
            if "max_rel_diff" in row and not row["case"].startswith("ud_grade")
        ]
        if scaled:
            worst = max(scaled, key=lambda row: row["max_rel_diff"])
            if worst["max_rel_diff"] > args.max_relative_error:
                failures.append(
                    f"{worst['case']} max rel diff {worst['max_rel_diff']:.3e}"
                )
    if any(row["status"] != "OK" for row in rows):
        failures.append("non-OK row")
    if failures:
        for failure in failures:
            print(f"Real-sphere threshold exceeded: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
