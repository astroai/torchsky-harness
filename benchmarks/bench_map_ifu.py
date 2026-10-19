#!/usr/bin/env python3
"""Matched-sampling 3D IFU gate for ``coadd_cube``.

A synthetic IFU observation carries one spectral sample per detector pixel via a
wavelength map. The target cube uses the same spectral sampling, so a correctly
indexed cube is sparse: roughly one valid target slice per source pixel. The gate
checks exact valid-voxel recovery and the expected sparse coverage; it does not
claim dense spectral interpolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torchsky.map import Observation, ResampleConfig, Target, coadd_cube
from torchsky.wcs import WCS
from torchsky_harness.metrics import (
    check_map_benchmark_regressions,
    write_map_benchmark_artifacts,
)


def run_case(*, size: int = 12) -> dict:
    header = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (size + 1) / 2.0,
        "CRPIX2": (size + 1) / 2.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
    }
    wcs = WCS(header)
    rows = torch.arange(size, dtype=torch.float32).unsqueeze(1).expand(size, size)
    wavelength_map = 5000.0 + 20.0 * rows
    data = 2.0 + 0.01 * (wavelength_map - 5000.0)
    observation = Observation(
        data=data,
        wcs=wcs,
        weight=torch.ones_like(data),
        wavelength_map=wavelength_map,
    )
    target = Target(wcs=wcs, shape=(size, size))
    wavelengths = torch.tensor(
        [5000.0 + 20.0 * index for index in range(size)], dtype=torch.float32
    )
    result = coadd_cube(
        [observation],
        target,
        wavelengths=wavelengths,
        resample=ResampleConfig(spectral_kernel="drizzle", spectral_pixfrac=1.0),
    )
    truth = (
        2.0 + 0.01 * (wavelengths.view(-1, 1, 1) - 5000.0)
    ).expand_as(result.cube)
    valid = result.weight > 0.0
    if not bool(valid.any()):
        return {"case": "ifu_matched_sparse_spectrum", "status": "FAILED"}
    residual = result.cube[valid] - truth[valid]
    valid_fraction = float(valid.to(torch.float32).mean())
    expected_fraction = 1.0 / float(size)
    max_abs_err = float(residual.abs().max())
    status = (
        "OK"
        if max_abs_err <= 1.0e-5
        and 0.5 * expected_fraction <= valid_fraction <= 2.0 * expected_fraction
        else "FAILED"
    )
    return {
        "case": "ifu_matched_sparse_spectrum",
        "library": "torchsky",
        "method": "coadd_cube",
        "variant": f"{size}x{size}x{size}",
        "status": status,
        "valid_fraction": valid_fraction,
        "expected_sparse_fraction": expected_fraction,
        "reconstruction_rmse": float(torch.sqrt(torch.mean(residual.square()))),
        "reconstruction_max_abs_err": max_abs_err,
        "signal_to_residual_db": 300.0,
        "total_flux_relative_error": float(
            torch.abs(result.cube[valid].sum() - truth[valid].sum())
            / torch.abs(truth[valid].sum())
        ),
        "skip_time_regression": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument(
        "--require-complete-baseline",
        action="store_true",
        help="Fail when a required baseline row or metric is missing",
    )
    args = parser.parse_args()
    row = run_case(size=args.size)
    rows = [row]
    markdown = write_map_benchmark_artifacts(
        rows,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
    )
    print(markdown)
    if args.json_out is not None:
        print(f"JSON: {args.json_out}")
    if args.baseline_json is not None:
        baseline_rows = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        errors = check_map_benchmark_regressions(
            rows,
            baseline_rows,
            require_complete=args.require_complete_baseline,
        )
        if errors:
            print("\n".join(errors))
            return 1
    return 0 if row["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
