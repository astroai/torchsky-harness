#!/usr/bin/env python3
"""Benchmark current per-plane and batched triplane feature sampling."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Literal, Sequence

import torch
import torch.nn.functional as F

import torchsky
from torchsky.map.field.hashgrid import sample_feature_plane
from torchsky_harness.subject import identify_subject


_PLANE_NAMES = {
    2: ("xy", "xl", "yl"),
    3: ("xy", "xl", "yl", "xz", "yz", "zl"),
}
_QUICK_CASES = (
    (2, 4, (16,), 1_000, "contiguous", "forward"),
    (3, 8, (32,), 1_000, "broadcast", "forward_backward"),
    (2, 8, (16, 32, 64), 10_000, "broadcast", "forward"),
    (3, 4, (16, 32, 64), 1_000, "contiguous", "forward_backward"),
)


@dataclass(frozen=True)
class TriplaneCase:
    spatial_dim: Literal[2, 3]
    features: int
    resolutions: tuple[int, ...]
    points: int
    coordinate_layout: Literal["contiguous", "broadcast"]
    execution: Literal["forward", "forward_backward"]

    @property
    def plane_count(self) -> int:
        return len(_PLANE_NAMES[self.spatial_dim])


def profile_cases(profile: str) -> tuple[TriplaneCase, ...]:
    if profile == "quick":
        raw = _QUICK_CASES
    elif profile == "full":
        references = (
            (8, (64,), 10_000),
            *((width, (64,), 10_000) for width in (4, 8, 16, 32)),
            *((8, (resolution,), 10_000) for resolution in (16, 32, 64, 128, 256)),
            *((8, (64,), points) for points in (1_000, 10_000, 100_000, 1_000_000)),
            (8, (16, 32, 64), 10_000),
            (8, (64, 128, 256), 10_000),
        )
        raw = tuple(
            (spatial_dim, features, resolutions, points, layout, execution)
            for spatial_dim in (2, 3)
            for layout in ("contiguous", "broadcast")
            for execution in ("forward", "forward_backward")
            for features, resolutions, points in references
        )
    else:
        raise ValueError(f"unknown profile {profile!r}")
    return tuple(
        dict.fromkeys(
            TriplaneCase(
                spatial_dim=spatial_dim,
                features=features,
                resolutions=resolutions,
                points=points,
                coordinate_layout=layout,
                execution=execution,
            )
            for spatial_dim, features, resolutions, points, layout, execution in raw
        )
    )


def _point_shape(points: int) -> tuple[int, int]:
    rows = math.isqrt(points)
    while rows > 1 and points % rows:
        rows -= 1
    return rows, points // rows


def _coordinate_values(
    case: TriplaneCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> dict[str, torch.Tensor]:
    rows, columns = _point_shape(case.points)
    names = ("x", "y", "lam", "z")[: case.spatial_dim + 1]
    coordinates: dict[str, torch.Tensor] = {}
    for index, name in enumerate(names):
        generator = torch.Generator(device=device).manual_seed(9100 + index)
        if case.coordinate_layout == "contiguous":
            value = 1.4 * torch.rand(
                (rows, columns), generator=generator, device=device, dtype=dtype
            ) - 0.2
            anchors = value.reshape(-1)
            anchors[:4] = anchors.new_tensor((-0.25, 0.0, 1.0, 1.25))
        elif index % 2:
            base = 1.4 * torch.rand(
                (1, columns), generator=generator, device=device, dtype=dtype
            ) - 0.2
            base[0, : min(4, columns)] = base.new_tensor(
                (-0.25, 0.0, 1.0, 1.25)[: min(4, columns)]
            )
            value = base.expand(rows, columns)
        else:
            base = 1.4 * torch.rand(
                (rows, 1), generator=generator, device=device, dtype=dtype
            ) - 0.2
            base[: min(4, rows), 0] = base.new_tensor(
                (-0.25, 0.0, 1.0, 1.25)[: min(4, rows)]
            )
            value = base.expand(rows, columns)
        coordinates[name] = value.detach().requires_grad_(requires_grad)
    return coordinates


def _planes(
    case: TriplaneCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    generator = torch.Generator(device=device).manual_seed(12031)
    return tuple(
        tuple(
            torch.randn(
                (case.features, resolution, resolution),
                generator=generator,
                device=device,
                dtype=dtype,
            ).requires_grad_(requires_grad)
            for _ in range(case.plane_count)
        )
        for resolution in case.resolutions
    )


def _coordinate_pairs(
    coordinates: dict[str, torch.Tensor], spatial_dim: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    pairs = (
        (coordinates["x"], coordinates["y"]),
        (coordinates["x"], coordinates["lam"]),
        (coordinates["y"], coordinates["lam"]),
    )
    if spatial_dim == 3:
        pairs += (
            (coordinates["x"], coordinates["z"]),
            (coordinates["y"], coordinates["z"]),
            (coordinates["z"], coordinates["lam"]),
        )
    return pairs


def _per_plane_sample(
    planes: tuple[tuple[torch.Tensor, ...], ...],
    coordinates: dict[str, torch.Tensor],
    spatial_dim: int,
) -> torch.Tensor:
    pairs = _coordinate_pairs(coordinates, spatial_dim)
    return torch.cat(
        [
            sample_feature_plane(plane, coord_u, coord_v)
            for scale in planes
            for plane, (coord_u, coord_v) in zip(scale, pairs, strict=True)
        ],
        dim=-1,
    )


def _batched_sample(
    planes: tuple[tuple[torch.Tensor, ...], ...],
    coordinates: dict[str, torch.Tensor],
    spatial_dim: int,
) -> torch.Tensor:
    pairs = _coordinate_pairs(coordinates, spatial_dim)
    outputs = []
    for scale in planes:
        plane_batch = torch.stack(scale, dim=0)
        grids = [
            torch.stack(
                (2.0 * coord_u.reshape(-1) - 1.0, 2.0 * coord_v.reshape(-1) - 1.0),
                dim=-1,
            )
            for coord_u, coord_v in pairs
        ]
        grid_batch = torch.stack(grids, dim=0).unsqueeze(-2).to(
            device=plane_batch.device, dtype=plane_batch.dtype
        )
        sampled = F.grid_sample(
            plane_batch,
            grid_batch,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        outputs.append(sampled.squeeze(-1).permute(2, 0, 1).flatten(start_dim=1))
    output = torch.cat(outputs, dim=-1)
    return output.reshape(*coordinates["x"].shape, -1)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _candidate_capability(device: torch.device, dtype: torch.dtype) -> tuple[bool, str | None]:
    case = TriplaneCase(2, 2, (4,), 4, "contiguous", "forward_backward")
    try:
        planes = _planes(case, device=device, dtype=dtype, requires_grad=True)
        coordinates = _coordinate_values(
            case, device=device, dtype=dtype, requires_grad=True
        )
        _batched_sample(planes, coordinates, 2).sum().backward()
        _synchronize(device)
    except (RuntimeError, NotImplementedError) as exc:
        return False, f"batched grid_sample unavailable: {type(exc).__name__}: {exc}"
    return True, None


def _flatten_planes(
    planes: tuple[tuple[torch.Tensor, ...], ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(plane for scale in planes for plane in scale)


def _parity(
    case: TriplaneCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    def clone_inputs():
        planes = _planes(case, device=device, dtype=dtype, requires_grad=True)
        coordinates = _coordinate_values(
            case, device=device, dtype=dtype, requires_grad=True
        )
        return planes, coordinates

    baseline_planes, baseline_coordinates = clone_inputs()
    candidate_planes, candidate_coordinates = clone_inputs()
    baseline = _per_plane_sample(baseline_planes, baseline_coordinates, case.spatial_dim)
    candidate = _batched_sample(candidate_planes, candidate_coordinates, case.spatial_dim)
    generator = torch.Generator(device=device).manual_seed(441)
    cotangent = torch.randn(
        baseline.shape, generator=generator, device=device, dtype=dtype
    )
    baseline_inputs = _flatten_planes(baseline_planes) + tuple(
        baseline_coordinates.values()
    )
    candidate_inputs = _flatten_planes(candidate_planes) + tuple(
        candidate_coordinates.values()
    )
    baseline_gradients = torch.autograd.grad(baseline, baseline_inputs, cotangent)
    candidate_gradients = torch.autograd.grad(candidate, candidate_inputs, cotangent)
    plane_gradient_errors = {}
    plane_total = len(_flatten_planes(baseline_planes))
    plane_names = _PLANE_NAMES[case.spatial_dim]
    for index, (left, right) in enumerate(
        zip(
            baseline_gradients[:plane_total],
            candidate_gradients[:plane_total],
            strict=True,
        )
    ):
        scale_index, plane_index = divmod(index, case.plane_count)
        plane_gradient_errors[f"s{scale_index}_{plane_names[plane_index]}"] = float(
            torch.max(torch.abs(left - right)).item()
        )
    coordinate_gradient_errors = {
        name: float(torch.max(torch.abs(left - right)).item())
        for name, left, right in zip(
            baseline_coordinates,
            baseline_gradients[plane_total:],
            candidate_gradients[plane_total:],
            strict=True,
        )
    }
    return {
        "forward_max_abs_error": float(torch.max(torch.abs(baseline - candidate)).item()),
        "plane_gradient_max_abs_error": plane_gradient_errors,
        "coordinate_gradient_max_abs_error": coordinate_gradient_errors,
        "gradient_max_abs_error": max(
            (*plane_gradient_errors.values(), *coordinate_gradient_errors.values()),
            default=0.0,
        ),
    }


def _invoke(
    sampler: Callable[
        [tuple[tuple[torch.Tensor, ...], ...], dict[str, torch.Tensor], int],
        torch.Tensor,
    ],
    planes: tuple[tuple[torch.Tensor, ...], ...],
    coordinates: dict[str, torch.Tensor],
    case: TriplaneCase,
    cotangent: torch.Tensor,
) -> None:
    if case.execution == "forward":
        with torch.no_grad():
            sampler(planes, coordinates, case.spatial_dim)
        return
    for value in (*_flatten_planes(planes), *coordinates.values()):
        value.grad = None
    sampler(planes, coordinates, case.spatial_dim).backward(cotangent)


def _cuda_kernel_count(
    invoke: Callable[[], None], device: torch.device
) -> tuple[int | None, str | None]:
    if device.type != "cuda":
        return None, "CUDA profiler is unavailable on CPU"
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as trace:
            invoke()
            _synchronize(device)
        count = sum(
            1 for event in trace.events() if "cuda" in str(event.device_type).lower()
        )
        return count, None
    except (RuntimeError, NotImplementedError) as exc:
        return None, f"CUDA profiler unavailable: {type(exc).__name__}: {exc}"


def _measure(
    case: TriplaneCase,
    *,
    implementation: str,
    sampler: Callable[
        [tuple[tuple[torch.Tensor, ...], ...], dict[str, torch.Tensor], int],
        torch.Tensor,
    ],
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    requires_grad = case.execution == "forward_backward"
    planes = _planes(case, device=device, dtype=dtype, requires_grad=requires_grad)
    coordinates = _coordinate_values(
        case, device=device, dtype=dtype, requires_grad=requires_grad
    )
    cotangent = torch.randn(
        (*coordinates["x"].shape, case.features * case.plane_count * len(case.resolutions)),
        generator=torch.Generator(device=device).manual_seed(661),
        device=device,
        dtype=dtype,
    )
    def invoke() -> None:
        _invoke(sampler, planes, coordinates, case, cotangent)

    for _ in range(warmup):
        invoke()
    _synchronize(device)
    allocated_before = None
    if device.type == "cuda":
        allocated_before = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(runs):
        _synchronize(device)
        started = time.perf_counter()
        invoke()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    peak_allocation = None
    peak_reason = "peak tensor allocation is only reported by CUDA"
    if device.type == "cuda" and allocated_before is not None:
        peak_allocation = max(
            int(torch.cuda.max_memory_allocated(device)) - allocated_before, 0
        )
        peak_reason = None
    kernel_count, kernel_reason = _cuda_kernel_count(invoke, device)
    median = statistics.median(samples)
    return {
        "implementation": implementation,
        "status": "OK",
        **asdict(case),
        "resolutions": list(case.resolutions),
        "scales": len(case.resolutions),
        "planes_per_scale": case.plane_count,
        "total_plane_samples": case.plane_count * len(case.resolutions),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "warmup": warmup,
        "runs": runs,
        "wall_seconds_median": median,
        "wall_seconds_min": min(samples),
        "wall_seconds_samples": samples,
        "points_per_second": case.points / median,
        "kernel_count": kernel_count,
        "kernel_count_unavailable_reason": kernel_reason,
        "peak_allocation_bytes": peak_allocation,
        "peak_allocation_unavailable_reason": peak_reason,
    }


def run_case(
    case: TriplaneCase,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    warmup: int = 1,
    runs: int = 2,
) -> list[dict[str, Any]]:
    target = torch.device(device)
    available, reason = _candidate_capability(target, dtype)
    parity = _parity(case, device=target, dtype=dtype) if available else None
    rows = [
        _measure(
            case,
            implementation="current_per_plane",
            sampler=_per_plane_sample,
            device=target,
            dtype=dtype,
            warmup=warmup,
            runs=runs,
        )
    ]
    rows[0].update(
        {
            "candidate_capability": "torch_batched_grid_sample",
            "candidate_available": available,
            "candidate_unavailable_reason": reason,
            "parity_reference": True,
            "forward_max_abs_error": 0.0,
            "plane_gradient_max_abs_error": {},
            "coordinate_gradient_max_abs_error": {},
            "gradient_max_abs_error": 0.0,
        }
    )
    if available:
        candidate = _measure(
            case,
            implementation="batched_per_scale_candidate",
            sampler=_batched_sample,
            device=target,
            dtype=dtype,
            warmup=warmup,
            runs=runs,
        )
        candidate.update(parity or {})
        candidate.update(
            {
                "candidate_capability": "torch_batched_grid_sample",
                "candidate_available": True,
                "candidate_unavailable_reason": None,
                "parity_reference": False,
            }
        )
    else:
        candidate = {
            "implementation": "batched_per_scale_candidate",
            "status": "SKIPPED",
            **asdict(case),
            "resolutions": list(case.resolutions),
            "scales": len(case.resolutions),
            "planes_per_scale": case.plane_count,
            "total_plane_samples": case.plane_count * len(case.resolutions),
            "device": str(target),
            "dtype": str(dtype).removeprefix("torch."),
            "candidate_capability": "torch_batched_grid_sample",
            "candidate_available": False,
            "candidate_unavailable_reason": reason,
        }
    rows.append(candidate)
    return rows


def run_suite(
    *,
    profile: str,
    devices: Sequence[str],
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    rows = []
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            rows.append(
                {
                    "implementation": "all",
                    "status": "SKIPPED",
                    "device": "cuda",
                    "reason": "torch.cuda.is_available() is false",
                }
            )
            continue
        for case in profile_cases(profile):
            rows.extend(
                run_case(case, device=device, warmup=warmup, runs=runs)
            )
    return {
        "schema": "torchsky.field_triplane_sampling/v1",
        "profile": profile,
        "subject": identify_subject(),
        "torch_version": torch.__version__,
        "torchsky_version": torchsky.__version__,
        "configuration": {
            "devices": list(devices),
            "warmup": warmup,
            "runs": runs,
        },
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    warmup = args.warmup if args.warmup is not None else (1 if args.profile == "quick" else 3)
    runs = args.runs if args.runs is not None else (3 if args.profile == "quick" else 10)
    if warmup < 0 or runs <= 0:
        parser.error("warmup must be non-negative and runs must be positive")
    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    if not devices or any(device not in {"cpu", "cuda"} for device in devices):
        parser.error("devices must contain cpu and/or cuda")
    payload = run_suite(
        profile=args.profile, devices=devices, warmup=warmup, runs=runs
    )
    output = json.dumps(payload, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
