#!/usr/bin/env python3
"""Benchmark installed-subject PCG on controlled positive-definite systems."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch
from torch import Tensor

from torchsky.map.solvers import conjugate_gradient
from torchsky_harness.subject import identify_subject


@dataclass(frozen=True)
class Profile:
    sizes: dict[str, int]
    iteration_budgets: tuple[int, ...]
    runs: int


@dataclass(frozen=True)
class System:
    truth: Tensor
    rhs: Tensor
    operator: Callable[[Tensor], Tensor]
    preconditioner: Callable[[Tensor], Tensor] | None
    condition_bound: float


PROFILES = {
    "quick": Profile(
        sizes={
            "diagonal": 10_000,
            "dense": 128,
            "sparse": 10_000,
            "image": 128,
        },
        iteration_budgets=(8, 32),
        runs=1,
    ),
    "full": Profile(
        sizes={
            "diagonal": 1_000_000,
            "dense": 512,
            "sparse": 1_000_000,
            "image": 512,
        },
        iteration_budgets=(8, 32, 64, 128),
        runs=3,
    ),
}

CASES = ("diagonal", "dense", "sparse", "image")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_system(
    case: str,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> System:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if case == "diagonal":
        diagonal = torch.linspace(1.0, 8.0, size, dtype=dtype, device=device)
        truth = torch.sin(torch.linspace(0.0, 8.0, size, dtype=dtype, device=device))

        def operator(value: Tensor) -> Tensor:
            return diagonal * value

        def preconditioner(value: Tensor) -> Tensor:
            return value / diagonal

        return System(truth, operator(truth), operator, preconditioner, 8.0)
    if case == "dense":
        raw = torch.randn(size, size, dtype=dtype, generator=generator)
        orthogonal, _ = torch.linalg.qr(raw)
        eigenvalues = torch.linspace(1.0, 4.0, size, dtype=dtype)
        matrix = ((orthogonal * eigenvalues) @ orthogonal.T).to(device=device)
        truth = torch.cos(torch.linspace(0.0, 6.0, size, dtype=dtype, device=device))
        inverse_diagonal = torch.reciprocal(torch.diagonal(matrix))

        def operator(value: Tensor) -> Tensor:
            return matrix @ value

        def preconditioner(value: Tensor) -> Tensor:
            return inverse_diagonal * value

        return System(truth, operator(truth), operator, preconditioner, 4.0)
    if case == "sparse":
        truth = torch.sin(torch.linspace(0.0, 20.0, size, dtype=dtype, device=device))

        def operator(value: Tensor) -> Tensor:
            output = 3.0 * value.clone()
            output[1:] -= value[:-1]
            output[:-1] -= value[1:]
            return output

        def preconditioner(value: Tensor) -> Tensor:
            return value / 3.0

        return System(truth, operator(truth), operator, preconditioner, 5.0)
    if case == "image":
        axis = torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        truth = torch.exp(-4.0 * (xx.square() + yy.square())) + 0.1 * xx

        def operator(value: Tensor) -> Tensor:
            neighbours = (
                torch.roll(value, 1, 0)
                + torch.roll(value, -1, 0)
                + torch.roll(value, 1, 1)
                + torch.roll(value, -1, 1)
            )
            return 5.0 * value - neighbours

        def preconditioner(value: Tensor) -> Tensor:
            return value / 5.0

        return System(truth, operator(truth), operator, preconditioner, 9.0)
    raise ValueError(f"unknown case {case!r}")


def _counter(diagnostics: object, *names: str) -> int | None:
    for name in names:
        value = getattr(diagnostics, name, None)
        if value is not None:
            return int(value)
    return None


def _run_once(
    system: System,
    *,
    max_iterations: int,
    rtol: float,
    record_residual_history: bool,
    device: torch.device,
) -> tuple[float, Any, dict[str, int]]:
    counts = {"operator": 0, "preconditioner": 0}

    def operator(value: Tensor) -> Tensor:
        counts["operator"] += 1
        return system.operator(value)

    def preconditioner(value: Tensor) -> Tensor:
        counts["preconditioner"] += 1
        assert system.preconditioner is not None
        return system.preconditioner(value)

    kwargs: dict[str, Any] = {
        "operator_is_hpd": True,
        "max_iterations": max_iterations,
        "rtol": rtol,
        "record_residual_history": record_residual_history,
    }
    if system.preconditioner is not None:
        kwargs["preconditioner"] = preconditioner
    _synchronize(device)
    started = time.perf_counter()
    result = conjugate_gradient(operator, system.rhs, **kwargs)
    _synchronize(device)
    return time.perf_counter() - started, result, counts


def run_case(
    case: str,
    *,
    size: int,
    max_iterations: int,
    runs: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1.0e-6,
    seed: int = 20261023,
    record_residual_history: bool = False,
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    system = _make_system(case, size, device=resolved_device, dtype=dtype, seed=seed)
    _run_once(
        system,
        max_iterations=max_iterations,
        rtol=rtol,
        record_residual_history=record_residual_history,
        device=resolved_device,
    )
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    timings: list[float] = []
    result = None
    observed_counts: dict[str, int] = {}
    for _ in range(runs):
        elapsed, result, observed_counts = _run_once(
            system,
            max_iterations=max_iterations,
            rtol=rtol,
            record_residual_history=record_residual_history,
            device=resolved_device,
        )
        timings.append(elapsed)
    assert result is not None and result.diagnostics is not None
    diagnostics = result.diagnostics
    true_residual = system.rhs - system.operator(result.solution)
    rhs_norm = torch.linalg.vector_norm(system.rhs)
    relative_residual = float(
        (torch.linalg.vector_norm(true_residual) / torch.clamp_min(rhs_norm, torch.finfo(dtype).tiny)).item()
    )
    truth_norm = torch.linalg.vector_norm(system.truth)
    relative_solution_error = float(
        (torch.linalg.vector_norm(result.solution - system.truth) / torch.clamp_min(truth_norm, torch.finfo(dtype).tiny)).item()
    )
    breakdown = getattr(diagnostics, "breakdown", None)
    termination = getattr(diagnostics, "termination", None)
    correctness_limit = max(5.0e-3, 10.0 * rtol * system.condition_bound)
    return {
        "case": case,
        "size": size,
        "state_elements": int(system.truth.numel()),
        "condition_bound": system.condition_bound,
        "max_iterations": max_iterations,
        "runs": runs,
        "device": str(resolved_device),
        "dtype": str(dtype).removeprefix("torch."),
        "wall_time_seconds_median": statistics.median(timings),
        "wall_time_seconds_min": min(timings),
        "iterations": int(diagnostics.iterations),
        "operator_applications": _counter(diagnostics, "operator_applications"),
        "preconditioner_applications": _counter(diagnostics, "preconditioner_applications"),
        "observed_operator_applications": observed_counts["operator"],
        "observed_preconditioner_applications": observed_counts["preconditioner"],
        "convergence_checks": _counter(diagnostics, "convergence_checks", "stopping_checks"),
        "validity_checks": _counter(diagnostics, "validity_checks", "breakdown_checks"),
        "true_residual_recomputations": _counter(
            diagnostics, "true_residual_recomputations", "residual_recomputations"
        ),
        "host_scalar_extractions": _counter(
            diagnostics, "host_scalar_extractions", "scalar_extractions"
        ),
        "termination_reason": getattr(termination, "value", str(termination)),
        "converged": bool(diagnostics.converged),
        "breakdown": None if breakdown is None else getattr(breakdown.reason, "value", str(breakdown.reason)),
        "final_true_relative_residual": relative_residual,
        "relative_solution_error": relative_solution_error,
        "correctness_limit": correctness_limit,
        "known_solution_correct": math.isfinite(relative_solution_error)
        and relative_solution_error <= correctness_limit,
        "residual_history_enabled": record_residual_history,
        "residual_history_length": len(result.residual_norm_history),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.type == "cuda"
            else None
        ),
    }


def run_benchmark(
    *,
    profile: str = "quick",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    cases: tuple[str, ...] = CASES,
    iteration_budgets: tuple[int, ...] | None = None,
    runs: int | None = None,
    rtol: float = 1.0e-6,
    seed: int = 20261023,
    record_residual_history: bool = False,
    include_subject: bool = True,
) -> dict[str, Any]:
    selected = PROFILES[profile]
    resolved_device = torch.device(device)
    budgets = iteration_budgets or selected.iteration_budgets
    effective_runs = selected.runs if runs is None else runs
    rows = [
        run_case(
            case,
            size=selected.sizes[case],
            max_iterations=budget,
            runs=effective_runs,
            device=resolved_device,
            dtype=dtype,
            rtol=rtol,
            seed=seed,
            record_residual_history=record_residual_history,
        )
        for case in cases
        for budget in budgets
    ]
    return {
        "benchmark": "solver_pcg",
        "profile": profile,
        "device": str(resolved_device),
        "dtype": str(dtype).removeprefix("torch."),
        "subject": identify_subject() if include_subject else None,
        "solver_signature": str(inspect.signature(conjugate_gradient)),
        "config": {
            "cases": list(cases),
            "sizes": {case: selected.sizes[case] for case in cases},
            "iteration_budgets": list(budgets),
            "runs": effective_runs,
            "rtol": rtol,
            "seed": seed,
            "residual_history_enabled": record_residual_history,
        },
        "rows": rows,
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _parse_csv(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("at least one value is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--iteration-budgets", default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--record-residual-history", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    cases = _parse_csv(args.cases)
    unknown = set(cases) - set(CASES)
    if unknown:
        parser.error(f"unknown cases: {', '.join(sorted(unknown))}")
    budgets = (
        None
        if args.iteration_budgets is None
        else tuple(int(value) for value in _parse_csv(args.iteration_budgets))
    )
    if args.runs is not None and args.runs < 1:
        parser.error("runs must be positive")
    if budgets is not None and any(value < 1 for value in budgets):
        parser.error("iteration budgets must be positive")
    if not math.isfinite(args.rtol) or args.rtol <= 0.0:
        parser.error("rtol must be finite and positive")
    artifact = run_benchmark(
        profile=args.profile,
        device=_resolve_device(args.device),
        dtype=getattr(torch, args.dtype),
        cases=cases,
        iteration_budgets=budgets,
        runs=args.runs,
        rtol=args.rtol,
        seed=args.seed,
        record_residual_history=args.record_residual_history,
    )
    payload = json.dumps(artifact, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
