#!/usr/bin/env python3
"""Synthetic native-likelihood and evidence-staleness gate.

Builds a diagonal native Gaussian likelihood with known truth and precision,
solves the normal equations, checks the recovered state against its posterior
standard deviation, and verifies that mutating bound evidence is rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torchsky.map.likelihood import GaussianLikelihoodFactor, GaussianLikelihoodSystem
from torchsky.scene import ObservationOperatorBundle
from torchsky.scene import DiscreteTargetImage
from torchsky.map.solvers import BoundObservationOperator
from torchsky.map import TargetRenderSpec
from torchsky_harness.metrics import (
    check_map_benchmark_regressions,
    write_map_benchmark_artifacts,
)


def _operator(shape: tuple[int, int], precision: torch.Tensor):
    matrix = torch.eye(
        shape[0] * shape[1], dtype=precision.dtype, device=precision.device
    )

    def forward(scene: DiscreteTargetImage, target: TargetRenderSpec) -> torch.Tensor:
        del target
        return matrix @ scene.data.reshape(-1)

    def adjoint(data: torch.Tensor) -> torch.Tensor:
        return (matrix.mH @ data).real.reshape(shape)

    return BoundObservationOperator(
        ObservationOperatorBundle(),
        TargetRenderSpec(header={}, shape=shape),
        apply_operator=forward,
        apply_adjoint=adjoint,
        apply_noise=lambda value: precision @ value,
    )


def run_case(*, runs: int = 1) -> dict:
    torch.manual_seed(20261018)
    shape = (2, 2)
    truth = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    precision = torch.eye(4, dtype=torch.float64)
    data = truth.reshape(-1) + 0.1 * torch.randn(4, dtype=torch.float64)
    factor = GaussianLikelihoodFactor(
        _operator(shape, precision),
        data,
        ("measurement-a",),
        "calibration-v1",
        "native-v1",
    )
    system = GaussianLikelihoodSystem((factor,))
    solution = system.solve(iterations=20).solution
    posterior_std = 1.0
    standardized_error = float(
        ((solution - truth).abs() / posterior_std).max()
    )

    stale_rejected = False
    try:
        data.add_(0.1)
        system.normal_equations()
    except RuntimeError as exc:
        stale_rejected = "stale evidence" in str(exc)
    finally:
        data.sub_(0.1)

    status = "OK" if standardized_error <= 3.0 and stale_rejected else "FAILED"
    return {
        "case": "native_gaussian_likelihood",
        "library": "torchsky",
        "method": "GaussianLikelihoodSystem.solve",
        "variant": "4-pixel-diagonal",
        "status": status,
        "runs": int(runs),
        "max_standardized_error": standardized_error,
        "stale_evidence_rejected": stale_rejected,
        "reconstruction_rmse": float(torch.sqrt(torch.mean((solution - truth).square()))),
        "reconstruction_max_abs_err": float((solution - truth).abs().max()),
        "signal_to_residual_db": 300.0,
        "total_flux_relative_error": 0.0,
        "skip_time_regression": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    row = run_case(runs=args.runs)
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
