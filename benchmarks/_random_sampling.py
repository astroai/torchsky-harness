from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np
import torch


Draw = Callable[[torch.Generator], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class RandomSamplingCase:
    subject: str
    n_cells: int
    n_samples: int
    device: torch.device
    one_shot: Draw
    contains: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    categories: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    expected_probabilities: torch.Tensor
    ordering: str | None = None
    weighted: bool | None = None
    prepared_factory: Callable[[], Any] | None = None
    prepared_api: str | None = None
    prepared_unavailable_reason: str | None = None
    all_sky: bool = False


def sampling_sizes(profile: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if profile == "off":
        return (), ()
    if profile == "quick":
        return (1_000,), (1_000,)
    if profile == "full":
        return (1_000, 100_000, 1_000_000), (1_000, 100_000)
    raise ValueError(f"unknown sampling profile: {profile!r}")


def discover_prepared_factory(
    subject: Any,
    *,
    weights: torch.Tensor | str | None = None,
    device: torch.device,
) -> tuple[Callable[[], Any] | None, str | None, str | None]:
    for name in ("prepare_random_sampler", "prepare_sampler", "prepare_randoms"):
        method = getattr(subject, name, None)
        if not callable(method):
            continue
        kwargs = _supported_kwargs(method, weights=weights, device=device)
        return lambda method=method, kwargs=kwargs: method(**kwargs), name, None
    return (
        None,
        None,
        "subject exposes no prepare_random_sampler, prepare_sampler, or prepare_randoms API",
    )


def benchmark_random_sampling_case(
    case: RandomSamplingCase,
    *,
    runs: int,
    seed: int,
) -> dict[str, Any]:
    run_count = max(1, int(runs))
    cold_generator = _generator(case.device, seed)
    _sync(case.device)
    t0 = time.perf_counter()
    cold_lon, cold_lat = case.one_shot(cold_generator)
    _sync(case.device)
    cold_s = time.perf_counter() - t0

    repeated_samples: list[float] = []
    for index in range(run_count):
        generator = _generator(case.device, seed + index + 1)
        _sync(case.device)
        t0 = time.perf_counter()
        case.one_shot(generator)
        _sync(case.device)
        repeated_samples.append(time.perf_counter() - t0)
    one_shot_repeated_s = float(np.median(repeated_samples))

    row: dict[str, Any] = {
        "operation": "random_sampling",
        "library": "torchsky",
        "subject": case.subject,
        "n_cells": int(case.n_cells),
        "n_samples": int(case.n_samples),
        "device": str(case.device),
        "allocation_device": str(cold_lon.device),
        "ordering": case.ordering,
        "weighted": case.weighted,
        "one_shot_available": True,
        "one_shot_cold_s": float(cold_s),
        "one_shot_repeated_s": one_shot_repeated_s,
        "one_shot_samples_per_s": _rate(case.n_samples, one_shot_repeated_s),
        "prepared_api_available": case.prepared_factory is not None,
        "prepared_api": case.prepared_api,
        "prepared_unavailable_reason": case.prepared_unavailable_reason,
        "prepare_s": None,
        "prepared_storage_bytes": None,
        "prepared_first_draw_s": None,
        "prepared_repeated_draw_s": None,
        "prepared_samples_per_s": None,
        "prepared_amortized_1_draw_s": None,
        "prepared_amortized_10_draw_s": None,
        "prepared_amortized_100_draw_s": None,
        "prepared_amortized_1000_draw_s": None,
    }
    row.update(_validation_fields(case, cold_lon, cold_lat))

    if case.prepared_factory is None:
        return row

    try:
        _sync(case.device)
        t0 = time.perf_counter()
        prepared = case.prepared_factory()
        _sync(case.device)
        prepare_s = time.perf_counter() - t0
        prepared_draw = _prepared_draw(prepared, case.n_samples, case.device)
        generator = _generator(case.device, seed)
        _sync(case.device)
        t0 = time.perf_counter()
        prepared_draw(generator)
        _sync(case.device)
        first_draw_s = time.perf_counter() - t0
        draw_samples: list[float] = []
        for index in range(run_count):
            generator = _generator(case.device, seed + index + 1)
            _sync(case.device)
            t0 = time.perf_counter()
            prepared_draw(generator)
            _sync(case.device)
            draw_samples.append(time.perf_counter() - t0)
        repeated_draw_s = float(np.median(draw_samples))
    except Exception as exc:
        raise RuntimeError(
            f"advertised prepared sampler {case.prepared_api} failed"
        ) from exc

    row.update(
        {
            "prepare_s": float(prepare_s),
            "prepared_storage_bytes": _tensor_storage_bytes(prepared),
            "prepared_first_draw_s": float(first_draw_s),
            "prepared_repeated_draw_s": repeated_draw_s,
            "prepared_samples_per_s": _rate(case.n_samples, repeated_draw_s),
        }
    )
    for count in (1, 10, 100, 1000):
        row[f"prepared_amortized_{count}_draw_s"] = float(
            (prepare_s + count * repeated_draw_s) / count
        )
    return row


def _validation_fields(
    case: RandomSamplingCase,
    lon: torch.Tensor,
    lat: torch.Tensor,
) -> dict[str, Any]:
    lon_flat = lon.detach().reshape(-1)
    lat_flat = lat.detach().reshape(-1)
    inside = case.contains(lon_flat, lat_flat).to(torch.bool).reshape(-1)
    categories = case.categories(lon_flat, lat_flat).to(torch.int64).reshape(-1)
    expected = case.expected_probabilities.detach().to(torch.float64).reshape(-1).cpu()
    observed = torch.bincount(
        categories.detach().cpu().clamp(min=0), minlength=int(expected.numel())
    )[: expected.numel()].to(torch.float64)
    observed /= max(1, int(categories.numel()))
    expected /= expected.sum()
    residual = observed - expected
    denom = torch.clamp(expected, min=torch.finfo(torch.float64).eps)
    chi_square = float(
        (int(categories.numel()) * torch.square(residual) / denom).sum().item()
    )
    checksum_weights = torch.arange(
        1,
        lon_flat.numel() + 1,
        dtype=torch.float64,
        device=lon_flat.device,
    )
    checksum = float(
        (
            torch.remainder(lon_flat.to(torch.float64), 360.0) * checksum_weights
            + (lat_flat.to(torch.float64) + 90.0) * (checksum_weights + 1.0)
        )
        .sum()
        .item()
    )
    sin_lat_mean = float(torch.sin(torch.deg2rad(lat_flat.to(torch.float64))).mean())
    lon_unit_mean = torch.exp(
        1j * torch.deg2rad(lon_flat.to(torch.float64).cpu())
    ).mean()
    return {
        "containment_all": bool(inside.all().item()),
        "containment_fraction": float(inside.to(torch.float64).mean().item()),
        "sample_checksum": checksum,
        "statistical_bins": int(expected.numel()),
        "statistical_total_variation": float(0.5 * residual.abs().sum().item()),
        "statistical_chi_square": chi_square,
        "statistical_chi_square_per_dof": chi_square / max(1, int(expected.numel()) - 1),
        "all_sky_uniformity_applicable": bool(case.all_sky),
        "longitude_resultant_length": float(abs(lon_unit_mean)),
        "sin_lat_mean": sin_lat_mean,
    }


def _prepared_draw(prepared: Any, n_samples: int, device: torch.device) -> Draw:
    for name in ("sample", "sample_randoms", "draw"):
        method = getattr(prepared, name, None)
        if callable(method):
            return lambda generator, method=method: method(
                n_samples,
                **_supported_kwargs(method, generator=generator, device=device),
            )
    if callable(prepared):
        return lambda generator: prepared(
            n_samples,
            **_supported_kwargs(prepared, generator=generator, device=device),
        )
    raise TypeError("prepared sampler exposes no sample, sample_randoms, draw, or callable API")


def _supported_kwargs(fn: Callable[..., Any], **candidates: Any) -> dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return candidates
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return {
        name: value
        for name, value in candidates.items()
        if value is not None and (accepts_kwargs or name in signature.parameters)
    }


def _tensor_storage_bytes(value: Any, seen: set[int] | None = None) -> int:
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_storage_bytes(item, visited) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return sum(_tensor_storage_bytes(item, visited) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tensor_storage_bytes(getattr(value, field.name), visited)
            for field in fields(value)
        )
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _tensor_storage_bytes(attributes, visited)
    return 0


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _rate(n_samples: int, seconds: float) -> float:
    return float(n_samples) / seconds if seconds > 0.0 else math.inf
