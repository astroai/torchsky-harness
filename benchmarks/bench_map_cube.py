#!/usr/bin/env python3
"""Synthetic spectral-cube gate for torchsky cube coadds.

This is a bounded correctness evaluation: a known spatial-spectral truth cube is
split into per-band 2D images, each with an explicit spectral interval, and the
same observations are recombined with :func:`torchsky.map.coadd_cube`. It checks
voxel reconstruction and integrated band flux, not real-data calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import torch
import torchfits

from torchsky.map import coadd_cube
from torchsky.scene import ImageObservation
from torchsky.spectral import SpectralAxis
from torchsky_harness.metrics import (
    check_map_benchmark_regressions,
    write_map_benchmark_artifacts,
)


def _header(size: int) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (size + 1) / 2.0,
        "CRPIX2": (size + 1) / 2.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def _truth_image(size: int, *, background: float, amplitude: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float64) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sigma = max(size / 6.0, 1.0)
    return background + amplitude * torch.exp(
        -(xx.square() + yy.square()) / (2.0 * sigma * sigma)
    )


def run_case(*, size: int, bands: int, runs: int) -> dict:
    header = _header(size)
    centers = [5000.0 + 200.0 * index for index in range(bands)]
    half_width = 100.0
    edges = torch.tensor(
        [centers[0] - half_width]
        + [0.5 * (centers[i] + centers[i + 1]) for i in range(bands - 1)]
        + [centers[-1] + half_width],
        dtype=torch.float64,
    )
    axis = SpectralAxis(
        values=torch.tensor(centers, dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    truth = torch.stack(
        [
            _truth_image(size, background=1.0 + 0.5 * index, amplitude=2.0)
            for index in range(bands)
        ]
    )

    timings: list[float] = []
    result_data: torch.Tensor | None = None
    with tempfile.TemporaryDirectory(prefix="torchsky_cube_gate_") as tmp:
        paths = []
        for index in range(bands):
            path = Path(tmp) / f"band_{index}.fits"
            torchfits.write(
                str(path),
                truth[index].to(torch.float32),
                header=header,
                overwrite=True,
            )
            paths.append(path)
        observations = [
            ImageObservation(
                str(path),
                weight=1.0,
                weight_is_inverse_variance=True,
                wavelength=centers[index],
                spectral_min=centers[index] - half_width,
                spectral_max=centers[index] + half_width,
                band_name=f"band_{index}",
            )
            for index, path in enumerate(paths)
        ]
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            result = coadd_cube(
                observations,
                spectral_axis=axis,
                spectral_bin_edges=edges,
                footprint_order=6,
            )
            timings.append((time.perf_counter() - start) * 1.0e3)
            # Path observations follow the default compute device (CUDA when present);
            # the truth cube is built on the host, so compare on the host.
            result_data = result.data.detach().cpu().to(torch.float64)

    assert result_data is not None
    residual = result_data - truth
    band_flux_truth = truth.sum(dim=(1, 2))
    band_flux_output = result_data.sum(dim=(1, 2))
    band_flux_rel = torch.abs(band_flux_output - band_flux_truth) / torch.clamp(
        torch.abs(band_flux_truth), min=1.0e-12
    )
    time_ms = float(sorted(timings)[len(timings) // 2])
    status = "OK"
    if float(band_flux_rel.max()) > 1.0e-3 or float(residual.abs().max()) > 1.0e-5:
        status = "FAILED"
    signal_rms = float(torch.sqrt(torch.mean(truth.square())))
    residual_rms = float(torch.sqrt(torch.mean(residual.square())))
    signal_to_residual_db = 20.0 * math.log10(max(signal_rms, 1e-12) / max(residual_rms, 1e-12))
    return {
        "case": "cube_constant_bands",
        "library": "torchsky",
        "method": "coadd_cube",
        "variant": f"{size}x{size}x{bands}",
        "status": status,
        "n_bands": bands,
        "size": size,
        "time_ms": time_ms,
        "reconstruction_rmse": float(torch.sqrt(torch.mean(residual.square()))),
        "reconstruction_max_abs_err": float(residual.abs().max()),
        "total_flux_relative_error": float(
            torch.abs(result_data.sum() - truth.sum()) / torch.abs(truth.sum())
        ),
        "band_flux_relative_error_max": float(band_flux_rel.max()),
        "band_flux_relative_error": [float(value) for value in band_flux_rel],
        "skip_time_regression": True,
        "signal_rms": signal_rms,
        "signal_to_residual_db": signal_to_residual_db,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument(
        "--require-complete-baseline",
        action="store_true",
        help="Fail when a required baseline row or metric is missing",
    )
    args = parser.parse_args()
    if args.size < 4 or args.bands < 2:
        parser.error("--size must be >= 4 and --bands must be >= 2")

    row = run_case(size=args.size, bands=args.bands, runs=args.runs)
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
