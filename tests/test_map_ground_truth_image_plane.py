from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchsky.cpp")

from torchsky.map import coadd_image_plane, evaluate_map_product
from torchsky_harness.benchmarks.bench_map_ground_truth import make_case


def test_coadd_image_plane_clamps_negative_resampled_weights_for_high_order_kernels(
    tmp_path: Path,
) -> None:
    observations, target_header, truth = make_case(
        tmp_path,
        case="tan_random10_large_union_rect_weighted_masked",
        size=1024,
        seed=1234,
        width=1024,
        height=768,
    )

    result = coadd_image_plane(
        [item.observation for item in observations],
        target_header=target_header,
        resampling_kernel="bicubic",
    )
    metrics = evaluate_map_product(
        result.data,
        truth,
        weight=result.weight,
        header=target_header,
        value_semantics="surface_brightness",
    )

    assert torch.isfinite(result.data).all()
    assert torch.isfinite(result.weight).all()
    assert bool((result.weight >= 0.0).all())
    assert math.isfinite(float(metrics["total_flux_relative_error"]))
