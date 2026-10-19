#!/usr/bin/env python3
"""Synthetic native-response gate for the linear native star profile.

Injects a known pixel-integrated Gaussian source on a native detector stamp and
fits its flux plus constant background at the true centroid. This bounds the
native response and weighted linear solve; it is not a PSF, crowding, or
real-detector calibration claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from torchsky.map.operators.epsf import (
    GaussianPixelResponse,
    native_pixel_stamp,
    profile_native_star,
)
from torchsky_harness.metrics import (
    check_map_benchmark_regressions,
    write_map_benchmark_artifacts,
)


def run_case(*, size: int = 32, runs: int = 1) -> dict:
    torch.manual_seed(20261018)
    stamp_slice = slice(size // 4, 3 * size // 4)
    truth_flux = 100.0
    truth_background = 5.0
    sigma_pix = 1.5
    true_centroid = torch.tensor(
        [size / 2.0 + 0.3, size / 2.0 - 0.3], dtype=torch.float64
    )
    response = GaussianPixelResponse(torch.tensor(sigma_pix, dtype=torch.float64))
    empty = torch.zeros((size, size), dtype=torch.float64)
    variance = torch.ones((size, size), dtype=torch.float64)
    geometry = native_pixel_stamp(
        empty, variance, x_slice=stamp_slice, y_slice=stamp_slice
    )
    mean = truth_flux * response(geometry.pixel_xy - true_centroid) + truth_background
    noisy = mean + torch.randn_like(mean)
    image = torch.zeros((size, size), dtype=torch.float64)
    image[stamp_slice, stamp_slice] = noisy

    timings: list[float] = []
    fit = None
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        stamp = native_pixel_stamp(
            image, variance, x_slice=stamp_slice, y_slice=stamp_slice
        )
        fit = profile_native_star(stamp, response, true_centroid)
        timings.append((time.perf_counter() - start) * 1.0e3)
    assert fit is not None

    recovered_flux = float(fit.flux)
    recovered_background = float(fit.background_coefficients[0])
    flux_rel = abs(recovered_flux - truth_flux) / truth_flux
    background_abs = abs(recovered_background - truth_background)
    time_ms = float(sorted(timings)[len(timings) // 2])
    status = "OK" if flux_rel <= 0.05 and background_abs <= 0.2 else "FAILED"
    return {
        "case": "native_gaussian_star",
        "library": "torchsky",
        "method": "profile_native_star",
        "variant": f"{size}x{size}",
        "status": status,
        "time_ms": time_ms,
        "skip_time_regression": True,
        "true_flux": truth_flux,
        "recovered_flux": recovered_flux,
        "flux_relative_error": flux_rel,
        "true_background": truth_background,
        "recovered_background": recovered_background,
        "background_absolute_error": background_abs,
        "chi2": float(fit.chi2),
        "degrees_of_freedom": int(fit.degrees_of_freedom),
        "retained_response": float(fit.retained_response),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
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
    if args.size < 12 or args.size % 4 != 0:
        parser.error("--size must be a multiple of 4 and >= 12")

    row = run_case(size=args.size, runs=args.runs)
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
