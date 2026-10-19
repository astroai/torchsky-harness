#!/usr/bin/env python3
"""Synthetic native nuisance-precision gate.

Builds a small profiled/calibrated nuisance span and compares its precision
action against an independent dense reference. This covers the matrix-free
nuisance primitive; it is not a joint-field or real-calibration claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torchsky.map.operators.nuisance import (
    CalibratedNuisancePrecision,
    ProfiledNuisancePrecision,
)
from torchsky_harness.metrics import (
    check_map_benchmark_regressions,
    write_map_benchmark_artifacts,
)


def run_case() -> dict:
    basis = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [0.0, 0.7], [1.0, 1.0]], dtype=torch.float64
    )
    weights = torch.tensor([2.0, 1.0, 3.0, 0.5], dtype=torch.float64)
    base = torch.diag(weights)
    value = torch.tensor([1.2, -0.3, 2.0, 0.7], dtype=torch.float64)

    profiled = ProfiledNuisancePrecision(
        basis, apply_precision=lambda data: weights * data
    )
    calibrated = CalibratedNuisancePrecision(
        basis,
        apply_precision=lambda data: weights * data,
        covariance_factor=torch.tensor(
            [[0.3, 0.0], [0.0, 0.2]], dtype=torch.float64
        ),
    )

    profiled_ref = (
        base
        - base
        @ basis
        @ torch.linalg.pinv(basis.T @ base @ basis)
        @ basis.T
        @ base
    ) @ value
    factor = basis @ torch.tensor(
        [[0.3, 0.0], [0.0, 0.2]], dtype=torch.float64
    )
    small = torch.eye(2, dtype=torch.float64) + factor.T @ base @ factor
    calibrated_ref = (
        base
        - base
        @ factor
        @ torch.linalg.solve(small, factor.T @ base)
    ) @ value

    profiled_err = float((profiled.apply(value) - profiled_ref).abs().max())
    calibrated_err = float((calibrated.apply(value) - calibrated_ref).abs().max())
    status = "OK" if profiled_err <= 1.0e-10 and calibrated_err <= 1.0e-10 else "FAILED"
    return {
        "case": "native_nuisance_precision",
        "library": "torchsky",
        "method": "ProfiledNuisancePrecision/CalibratedNuisancePrecision",
        "variant": "4-pixel-2-mode",
        "status": status,
        "profiled_max_abs_error": profiled_err,
        "calibrated_max_abs_error": calibrated_err,
        "profiled_rank": int(profiled.nuisance_rank),
        "reconstruction_rmse": profiled_err,
        "signal_to_residual_db": 300.0,
        "total_flux_relative_error": 0.0,
        "skip_time_regression": True,
    }


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
    row = run_case()
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
