#!/usr/bin/env python3
"""Diagnostic: 3D IFU linear-spectrum recovery through ``coadd_cube``.

This is intentionally not a release gate. It measures whether the native IFU
cube path recovers a known linear spectrum from a 3D wavelength map; the current
result is recorded in the release baseline and blocks any IFU accuracy claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torchsky.map import Observation, ResampleConfig, Target, coadd_cube
from torchsky.wcs import WCS


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
        [5020.0, 5060.0, 5100.0, 5140.0, 5180.0], dtype=torch.float32
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
        return {"case": "ifu_linear_spectrum", "status": "FAILED", "valid_pixels": 0}
    residual = result.cube[valid] - truth[valid]
    return {
        "case": "ifu_linear_spectrum",
        "status": "DIAGNOSTIC",
        "valid_fraction": float(valid.to(torch.float32).mean()),
        "reconstruction_rmse": float(torch.sqrt(torch.mean(residual.square()))),
        "reconstruction_max_abs_err": float(residual.abs().max()),
        "total_flux_relative_error": float(
            torch.abs(result.cube[valid].sum() - truth[valid].sum())
            / torch.abs(truth[valid].sum())
        ),
        "sample_output": [float(value) for value in result.cube[:, size // 2, size // 2]],
        "sample_truth": [float(value) for value in truth[:, 0, 0]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    row = run_case(size=args.size)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([row], indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
