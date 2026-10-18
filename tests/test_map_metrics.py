from __future__ import annotations

import pytest
import torch
import torchsky
from torchsky.map import FieldObservation, MapBandpassSpec

from torchsky_harness.metrics import (
    check_map_benchmark_regressions,
    evaluate_map_frontier_regions,
    evaluate_map_product,
    evaluate_map_uncertainty,
    field_quality_acceptance_report,
)


def _simple_header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 6,
        "NAXIS2": 6,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 3.0,
        "CRPIX2": 3.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }

def test_check_map_benchmark_regressions_respects_skip_time_flag() -> None:
    baseline = [
        {
            "case": "case",
            "library": "torchsky",
            "method": "coadd",
            "variant": "bilinear",
            "reconstruction_rmse": 1.0,
            "signal_to_residual_db": 10.0,
            "total_flux_relative_error": 0.1,
            "time_ms": 1.0,
            "skip_time_regression": True,
        }
    ]
    current = [
        {
            "case": "case",
            "library": "torchsky",
            "method": "coadd",
            "variant": "bilinear",
            "reconstruction_rmse": 1.05,
            "signal_to_residual_db": 10.0,
            "total_flux_relative_error": 0.105,
            "time_ms": 100.0,
        }
    ]
    errors = check_map_benchmark_regressions(current, baseline)
    assert errors == []


def test_check_map_benchmark_regressions_requires_complete_coverage() -> None:
    baseline = [
        {
            "case": "case",
            "library": "torchsky",
            "method": "coadd",
            "variant": "bilinear",
            "reconstruction_rmse": 1.0,
            "signal_to_residual_db": 10.0,
            "total_flux_relative_error": 0.1,
            "time_ms": 1.0,
            "skip_time_regression": True,
        }
    ]
    assert check_map_benchmark_regressions([], baseline) == []
    errors = check_map_benchmark_regressions(
        [], baseline, require_complete=True
    )
    assert any("missing required benchmark row" in error for error in errors)

    incomplete = [
        {
            "case": "case",
            "library": "torchsky",
            "method": "coadd",
            "variant": "bilinear",
            "reconstruction_rmse": 1.0,
        }
    ]
    errors = check_map_benchmark_regressions(
        incomplete, baseline, require_complete=True
    )
    assert any(
        "missing required metric signal_to_residual_db" in error
        for error in errors
    )

def test_evaluate_map_frontier_regions_tracks_fringe_and_overlap() -> None:
    truth = torch.zeros((1, 7, 7), dtype=torch.float32)
    pred = truth.clone()
    support_weight = torch.zeros_like(pred)
    support_weight[:, 2:5, 2:5] = 1.0
    overlap = torch.zeros_like(pred, dtype=torch.int32)
    overlap[:, 2:5, 2:5] = 1
    overlap[:, 3, 3] = 2
    pred[:, 2, 2] = 1.0
    pred[:, 3, 3] = 2.0
    pred[:, 1, 3] = 3.0
    pred[:, 0, 0] = 4.0

    metrics = evaluate_map_frontier_regions(
        pred,
        truth,
        support_weight=support_weight,
        overlap_count=overlap,
        fringe_kernel_size=3,
    )

    assert metrics["support_single_overlap_reconstruction_rmse"] > 0.0
    assert metrics["support_multi_overlap_reconstruction_rmse"] > 0.0
    assert metrics["fringe_extrapolated_reconstruction_rmse"] > 0.0
    assert metrics["far_extrapolated_reconstruction_rmse"] > 0.0
    assert metrics["fringe_extrapolated_fraction"] > 0.0
    assert metrics["far_extrapolated_fraction"] > 0.0

def test_evaluate_map_product_uses_mask_when_weight_is_unavailable() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="single"),),
    )
    data = torch.ones((1, 6, 6), dtype=torch.float32)
    prediction = torchsky.map.MapProductResult(
        product=product,
        data=data,
        weight=None,
        header=_simple_header(),
        mask=torch.ones_like(data, dtype=torch.bool),
    )

    metrics = evaluate_map_product(prediction, data)

    assert metrics["coverage_fraction"] == pytest.approx(1.0)

def test_evaluate_map_product_uses_product_level_metrics() -> None:
    product = torchsky.map.MapProductSpec(
        target_header=_simple_header(),
        bands=(MapBandpassSpec(label="single"),),
    )
    truth = torch.ones((1, 6, 6), dtype=torch.float32)
    pred = truth.clone()
    pred[0, 0, 0] = 3.0
    metrics = evaluate_map_product(
        torchsky.map.MapProductResult(
            product=product,
            data=pred,
            weight=torch.ones_like(pred),
            header=_simple_header(),
        ),
        truth,
    )

    assert metrics["reconstruction_rmse"] is not None
    assert float(metrics["reconstruction_rmse"]) > 0.0
    assert float(metrics["coverage_fraction"]) == pytest.approx(1.0)
    assert float(metrics["total_flux_relative_error"]) > 0.0
    assert metrics["band_reconstruction_rmse"] == [
        pytest.approx(float(metrics["reconstruction_rmse"]))
    ]
    assert len(metrics["band_flux_relative_error"]) == 1
    assert float(metrics["band_flux_relative_error"][0]) > 0.0

def test_evaluate_map_uncertainty_tracks_residual_ordering() -> None:
    truth = torch.zeros((1, 4, 4), dtype=torch.float32)
    pred = truth.clone()
    pred[0, 0, 0] = 4.0
    pred[0, 3, 3] = 0.1
    ivar = torch.ones_like(pred)
    ivar[0, 0, 0] = 0.01
    ivar[0, 3, 3] = 100.0
    support_weight = torch.ones_like(pred)
    support_weight[0, 0, 0] = 0.0

    metrics = evaluate_map_uncertainty(
        ivar,
        pred,
        truth,
        support_weight=support_weight,
    )

    assert metrics["uncertainty_error_corr"] > 0.0
    assert (
        metrics["uncertainty_top_quartile_abs_err"]
        > metrics["uncertainty_bottom_quartile_abs_err"]
    )
    assert (
        metrics["uncertainty_extrapolated_mean_sigma"]
        > metrics["uncertainty_support_mean_sigma"]
    )


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }

def test_field_quality_acceptance_report_flags_explicit_threshold_violations() -> None:
    observations = (
        FieldObservation(
            image=torch.ones((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=150.0, crval2=2.0, scale_deg=0.01),
            wavelength=5000.0,
            band_name="g",
        ),
        FieldObservation(
            image=torch.ones((2, 2), dtype=torch.float32),
            header=_tan_header(2, crval1=150.0, crval2=2.0, scale_deg=0.01),
            wavelength=6500.0,
            band_name="r",
        ),
    )
    metrics = (
        {
            "weighted_rmse": 0.05,
            "residual_snr": 20.0,
            "support_fraction": 0.95,
            "effective_weight_sum": 10.0,
            "psf_residual_alignment": 0.1,
        },
        {
            "weighted_rmse": 0.2,
            "residual_snr": 3.0,
            "support_fraction": 0.4,
            "effective_weight_sum": 2.0,
        },
    )

    report = field_quality_acceptance_report(
        metrics,
        observations=observations,
        max_weighted_rmse=0.1,
        min_residual_snr=5.0,
        min_support_fraction=0.5,
        min_effective_weight_sum=3.0,
        max_abs_psf_residual_alignment=0.2,
    )

    assert report["passed"] is False
    assert report["scorecard"]["by_band"]["g"]["observation_count"] == 1
    assert {
        (violation["index"], violation["metric"], violation.get("reason"))
        for violation in report["violations"]
    } == {
        (1, "weighted_rmse", None),
        (1, "residual_snr", None),
        (1, "support_fraction", None),
        (1, "effective_weight_sum", None),
        (1, "psf_residual_alignment", "missing"),
    }
