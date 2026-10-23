from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchsky_harness.benchmarks._random_sampling import (
    RandomSamplingCase,
    _tensor_storage_bytes,
    benchmark_random_sampling_case,
    discover_prepared_factory,
    sampling_sizes,
)


@dataclass(frozen=True, slots=True)
class _PreparedStorage:
    values: torch.Tensor


def _draw_all_sky(
    n_samples: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    lon = 360.0 * torch.rand(n_samples, generator=generator, dtype=torch.float64)
    z = 2.0 * torch.rand(n_samples, generator=generator, dtype=torch.float64) - 1.0
    return lon, torch.rad2deg(torch.asin(z))


def _case(
    *,
    prepared_factory=None,
    prepared_api: str | None = None,
    unavailable_reason: str | None = None,
) -> RandomSamplingCase:
    return RandomSamplingCase(
        subject="test",
        n_cells=12,
        n_samples=256,
        device=torch.device("cpu"),
        one_shot=lambda generator: _draw_all_sky(256, generator),
        contains=lambda lon, lat: torch.ones_like(lon, dtype=torch.bool),
        categories=lambda lon, lat: torch.floor(lon / 90.0).to(torch.int64),
        expected_probabilities=torch.full((4,), 0.25, dtype=torch.float64),
        ordering="NESTED",
        weighted=False,
        prepared_factory=prepared_factory,
        prepared_api=prepared_api,
        prepared_unavailable_reason=unavailable_reason,
        all_sky=True,
    )


def test_sampling_sizes_include_quick_and_full_ladders() -> None:
    assert sampling_sizes("off") == ((), ())
    assert sampling_sizes("quick") == ((1_000,), (1_000,))
    assert sampling_sizes("full") == (
        (1_000, 100_000, 1_000_000),
        (1_000, 100_000),
    )


def test_prepared_storage_counts_slotted_dataclass_tensors() -> None:
    assert _tensor_storage_bytes(
        _PreparedStorage(torch.ones(7, dtype=torch.float64))
    ) == 56


def test_unavailable_prepared_sampler_retains_one_shot_validation() -> None:
    row = benchmark_random_sampling_case(
        _case(unavailable_reason="not implemented"), runs=2, seed=7
    )

    assert row["one_shot_available"] is True
    assert row["prepared_api_available"] is False
    assert row["prepared_unavailable_reason"] == "not implemented"
    assert row["prepare_s"] is None
    assert row["prepared_amortized_1000_draw_s"] is None
    assert row["containment_all"] is True
    assert row["containment_fraction"] == 1.0
    assert math.isfinite(row["sample_checksum"])
    assert row["statistical_bins"] == 4
    assert row["all_sky_uniformity_applicable"] is True


def test_prepared_sampler_reports_preparation_draws_and_storage() -> None:
    class Prepared:
        def __init__(self) -> None:
            self.state = torch.ones(8, dtype=torch.float64)

        def sample(
            self, n_samples: int, *, generator: torch.Generator
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return _draw_all_sky(n_samples, generator)

    row = benchmark_random_sampling_case(
        _case(prepared_factory=Prepared, prepared_api="prepare_random_sampler"),
        runs=2,
        seed=11,
    )

    assert row["prepared_api_available"] is True
    assert row["prepared_storage_bytes"] == 64
    assert row["prepare_s"] is not None
    assert row["prepared_first_draw_s"] is not None
    assert row["prepared_repeated_draw_s"] is not None
    assert row["prepared_samples_per_s"] > 0.0
    assert row["prepared_amortized_1_draw_s"] >= row["prepared_repeated_draw_s"]
    assert row["prepared_amortized_1000_draw_s"] >= row["prepared_repeated_draw_s"]


def test_prepared_capability_discovery_is_explicit() -> None:
    class Subject:
        def prepare_random_sampler(self, *, device: torch.device):
            return device

    factory, api, reason = discover_prepared_factory(
        Subject(), device=torch.device("cpu")
    )
    assert factory is not None
    assert factory() == torch.device("cpu")
    assert api == "prepare_random_sampler"
    assert reason is None

    factory, api, reason = discover_prepared_factory(
        object(), device=torch.device("cpu")
    )
    assert factory is None
    assert api is None
    assert "no prepare_random_sampler" in str(reason)
