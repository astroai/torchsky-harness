#!/usr/bin/env python3
"""IFU spectral-axis gate: line localization and interpolation on the target axis.

A synthetic long-slit IFU (one slice per detector column, one spectral sample per
detector row) is assembled into a cube whose target wavelength axis is given
explicitly. The gate checks

* every monochromatic detector row lands in the target bin that contains its
  wavelength, for an irregular (non-uniform) target axis and all spectral
  kernels;
* a constant spectrum is normalized exactly on an irregular axis;
* a smooth spectrum is recovered on a regular axis within its interior
  interpolation tolerance;
* the recorded error envelope of the current index-space interpolation on an
  irregular axis, which is a documented limitation rather than a dense
  interpolation claim.

It does not claim flux-conserving spectral redistribution or physical
wavelength-space interpolation for strongly irregular axes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from torchsky.map import Observation, ResampleConfig, Target, coadd_cube
from torchsky.wcs import IFUSliceWCS, WCS
from torchsky_harness.metrics import (
    check_map_benchmark_regressions,
    write_map_benchmark_artifacts,
)

N_SLICE = 12
N_WAVE = 400
WAVE_START = 5000.0
WAVE_STEP = 2.0
TARGET_SIZE = 16
TARGET_CDELT_ARCSEC = 2.0
CENTRAL_PIXEL = (TARGET_SIZE // 2, TARGET_SIZE // 2)

IRREGULAR_AXIS = (5000.0, 5010.0, 5040.0, 5060.0, 5100.0, 5150.0, 5220.0)
REGULAR_AXIS = (5000.0, 5020.0, 5040.0, 5060.0, 5080.0, 5100.0)
IRREGULAR_ENVELOPE = 0.10
REGULAR_TOLERANCE = 2.5e-3
KERNELS = ("nearest", "drizzle", "linear")


def _ifu() -> IFUSliceWCS:
    rows = torch.arange(N_WAVE, dtype=torch.float64)
    columns = torch.arange(N_SLICE, dtype=torch.float64)
    ra_map = 150.0 + (columns - (N_SLICE - 1) / 2.0) * TARGET_CDELT_ARCSEC / 3600.0
    ra_map = ra_map.unsqueeze(0).expand(N_WAVE, N_SLICE).clone()
    dec_map = torch.full((N_WAVE, N_SLICE), 2.0, dtype=torch.float64)
    wave_map = (WAVE_START + WAVE_STEP * rows)
    wave_map = wave_map.unsqueeze(1).expand(N_WAVE, N_SLICE).clone()
    slice_map = columns.to(torch.int64).unsqueeze(0).expand(N_WAVE, N_SLICE).clone()
    return IFUSliceWCS(
        ra_map=ra_map,
        dec_map=dec_map,
        wavelength_map=wave_map,
        slice_id_map=slice_map,
    )


def _target() -> Target:
    header = {
        "NAXIS": 2,
        "NAXIS1": TARGET_SIZE,
        "NAXIS2": TARGET_SIZE,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (TARGET_SIZE + 1) / 2.0,
        "CRPIX2": (TARGET_SIZE + 1) / 2.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -TARGET_CDELT_ARCSEC / 3600.0,
        "CDELT2": TARGET_CDELT_ARCSEC / 3600.0,
    }
    return Target(wcs=WCS(header), shape=(TARGET_SIZE, TARGET_SIZE))


def _cube(
    spectrum,
    axis: torch.Tensor,
    *,
    spectral_kernel: str,
):
    wave_map = WAVE_START + WAVE_STEP * torch.arange(N_WAVE, dtype=torch.float64)
    data = spectrum(wave_map)
    data = data.unsqueeze(1).expand(N_WAVE, N_SLICE).clone().to(torch.float32)
    observation = Observation(
        data=data,
        ifu_wcs=_ifu(),
        weight=torch.ones_like(data),
    )
    return coadd_cube(
        [observation],
        _target(),
        wavelengths=axis,
        resample=ResampleConfig(spectral_kernel=spectral_kernel),
    )


def _interior_mask(weight: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask[1:-1, 2:-2, 2:-2] = True
    return mask & (weight > 0.0)


def _constant_spectrum(wave_map: torch.Tensor) -> torch.Tensor:
    return torch.full_like(wave_map, 5.0)


def _sloped_spectrum(wave_map: torch.Tensor) -> torch.Tensor:
    return 2.0 + 0.01 * (wave_map - WAVE_START)


def _curved_spectrum(wave_map: torch.Tensor) -> torch.Tensor:
    return _sloped_spectrum(wave_map) + 2.0e-5 * (wave_map - WAVE_START) ** 2


def _max_relative_error(values: torch.Tensor, truth: torch.Tensor) -> float:
    return float(((values - truth).abs() / truth.abs()).max())


def _residual_db(relative_error: float, signal: float) -> float:
    residual = max(relative_error * signal, 1.0e-300)
    return 20.0 * math.log10(signal / residual)


def _row(*, case: str, variant: str, status: str, **metrics: object) -> dict:
    return {
        "case": case,
        "library": "torchsky",
        "method": "coadd_cube",
        "variant": variant,
        "status": status,
        "skip_time_regression": True,
        **metrics,
    }


def run_line_localization_case() -> dict:
    """Each monochromatic row must land in the target bin holding its wavelength."""
    axis = torch.tensor(IRREGULAR_AXIS, dtype=torch.float64)
    column, row = CENTRAL_PIXEL
    kernels: dict[str, dict] = {}
    failures = 0
    for kernel in KERNELS:
        residuals: list[float] = []
        for index, wavelength in enumerate(IRREGULAR_AXIS):
            source_row = int(round((wavelength - WAVE_START) / WAVE_STEP))
            line_wave = WAVE_START + WAVE_STEP * source_row

            def spectrum(wave_map: torch.Tensor, line_wave: float = line_wave):
                data = torch.zeros_like(wave_map)
                data[(wave_map - line_wave).abs() < 1.0e-6] = 7.0
                return data

            result = _cube(spectrum, axis, spectral_kernel=kernel)
            profile = result.cube[:, column, row]
            expected = torch.zeros_like(profile)
            expected[index] = profile[index]
            residuals.append(float((profile - expected).abs().max()))
            if not bool(profile[index] > 0.0):
                failures += 1
        max_residual = max(residuals)
        kernels[kernel] = {
            "line_count": len(IRREGULAR_AXIS),
            "max_spurious_value": max_residual,
            "spurious_bin_count": sum(1 for value in residuals if value > 0.0),
        }
        if max_residual > 0.0:
            failures += 1
    status = "OK" if failures == 0 else "FAILED"
    return _row(
        case="ifu_nonuniform_axis_line_localization",
        variant=f"{N_SLICE} slices x {N_WAVE} samples",
        status=status,
        kernels=kernels,
        max_spurious_value=max(
            float(entry["max_spurious_value"]) for entry in kernels.values()
        )
        if kernels
        else 0.0,
        reconstruction_rmse=0.0,
        signal_to_residual_db=300.0,
        total_flux_relative_error=0.0,
    )


def run_constant_spectrum_case() -> dict:
    """A constant spectrum must be exact on an irregular axis, including edges."""
    axis = torch.tensor(IRREGULAR_AXIS, dtype=torch.float64)
    per_kernel: dict[str, float] = {}
    for kernel in KERNELS:
        result = _cube(_constant_spectrum, axis, spectral_kernel=kernel)
        valid = result.weight > 0.0
        truth = torch.full_like(result.cube[valid], 5.0)
        per_kernel[kernel] = _max_relative_error(result.cube[valid], truth)
    max_error = max(per_kernel.values())
    status = "OK" if max_error <= 1.0e-5 else "FAILED"
    return _row(
        case="ifu_nonuniform_axis_constant_spectrum",
        variant=f"edge-inclusive, {len(IRREGULAR_AXIS)} channels",
        status=status,
        per_kernel_max_relative_error=per_kernel,
        max_relative_error=max_error,
        reconstruction_rmse=max_error * 5.0,
        signal_to_residual_db=_residual_db(max_error, 5.0),
        total_flux_relative_error=max_error,
    )


def run_regular_axis_spectrum_case() -> dict:
    """A smooth spectrum on a regular axis must interpolate to its interior tolerance."""
    axis = torch.tensor(REGULAR_AXIS, dtype=torch.float64)
    per_case: dict[str, dict[str, float]] = {}
    max_error = 0.0
    for name, spectrum in (("sloped", _sloped_spectrum), ("curved", _curved_spectrum)):
        truth = spectrum(axis).to(torch.float32).view(-1, 1, 1)
        for kernel in ("drizzle", "linear"):
            result = _cube(spectrum, axis, spectral_kernel=kernel)
            interior = _interior_mask(result.weight)
            if not bool(interior.any()):
                return _row(
                    case="ifu_regular_axis_spectrum_interpolation",
                    variant="regular axis",
                    status="FAILED",
                    failure_reason="no interior voxels with spectral coverage",
                )
            error = _max_relative_error(
                result.cube[interior], truth.expand_as(result.cube)[interior]
            )
            per_case[f"{name}:{kernel}"] = {
                "interior_max_relative_error": error,
                "interior_voxel_count": int(interior.sum()),
            }
            max_error = max(max_error, error)
    status = "OK" if max_error <= REGULAR_TOLERANCE else "FAILED"
    return _row(
        case="ifu_regular_axis_spectrum_interpolation",
        variant=f"regular axis, {len(REGULAR_AXIS)} channels",
        status=status,
        per_case=per_case,
        max_relative_error=max_error,
        declared_tolerance=REGULAR_TOLERANCE,
        reconstruction_rmse=max_error * 2.0,
        signal_to_residual_db=_residual_db(max_error, 2.0),
        total_flux_relative_error=max_error,
    )


def run_irregular_envelope_case() -> dict:
    """Record the current index-space interpolation envelope on an irregular axis."""
    axis = torch.tensor(IRREGULAR_AXIS, dtype=torch.float64)
    truth = _sloped_spectrum(axis).to(torch.float32).view(-1, 1, 1)
    interior_error = 0.0
    edge_error = 0.0
    for kernel in ("drizzle", "linear"):
        result = _cube(_sloped_spectrum, axis, spectral_kernel=kernel)
        interior = _interior_mask(result.weight)
        if not bool(interior.any()):
            return _row(
                case="ifu_nonuniform_axis_interpolation_envelope",
                variant="irregular axis",
                status="FAILED",
                failure_reason="no interior voxels with spectral coverage",
            )
        expanded = truth.expand_as(result.cube)
        interior_error = max(
            interior_error,
            _max_relative_error(result.cube[interior], expanded[interior]),
        )
        valid = result.weight > 0.0
        edge_error = max(
            edge_error, _max_relative_error(result.cube[valid], expanded[valid])
        )
    status = "OK" if interior_error <= IRREGULAR_ENVELOPE else "FAILED"
    return _row(
        case="ifu_nonuniform_axis_interpolation_envelope",
        variant=f"irregular axis, {len(IRREGULAR_AXIS)} channels",
        status=status,
        interior_max_relative_error=interior_error,
        all_voxel_max_relative_error=edge_error,
        declared_envelope=IRREGULAR_ENVELOPE,
        reconstruction_rmse=interior_error * 2.0,
        signal_to_residual_db=_residual_db(interior_error, 2.0),
        total_flux_relative_error=interior_error,
    )


def run_case() -> list[dict]:
    return [
        run_line_localization_case(),
        run_constant_spectrum_case(),
        run_regular_axis_spectrum_case(),
        run_irregular_envelope_case(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument(
        "--require-complete-baseline",
        action="store_true",
        help="Fail when a required baseline row or metric is missing",
    )
    args = parser.parse_args()
    rows = run_case()
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
    return 0 if all(row["status"] == "OK" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
