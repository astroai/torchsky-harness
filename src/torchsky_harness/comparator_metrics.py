"""Bounded scientific metrics for drizzle, adaptive reprojection, and IFU cubes."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _common_finite(
    actual: Any, reference: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = _array(actual)
    right = _array(reference)
    if left.shape != right.shape:
        raise ValueError(f"array shapes differ: {left.shape} != {right.shape}")
    finite = np.isfinite(left) & np.isfinite(right)
    return left, right, finite


def floating_plane_metrics(
    actual: Any, reference: Any, *, prefix: str
) -> dict[str, float]:
    left, right, finite = _common_finite(actual, reference)
    if not bool(finite.any()):
        raise ValueError(f"{prefix} has no common finite samples")
    residual = left[finite].astype(np.float64) - right[finite].astype(np.float64)
    scale = max(
        float(np.sum(np.abs(right[finite]), dtype=np.float64)),
        np.finfo(np.float64).tiny,
    )
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        f"{prefix}_max_abs_error": float(np.max(np.abs(residual))),
        f"{prefix}_relative_l1_error": float(np.sum(np.abs(residual)) / scale),
        f"{prefix}_finite_fraction": float(finite.mean()),
    }


def exact_plane_metrics(
    actual: Any, reference: Any, *, prefix: str
) -> dict[str, float | int | bool]:
    left = _array(actual)
    right = _array(reference)
    if left.shape != right.shape:
        raise ValueError(f"array shapes differ: {left.shape} != {right.shape}")
    mismatch = left != right
    count = int(np.count_nonzero(mismatch))
    return {
        f"{prefix}_exact": count == 0,
        f"{prefix}_mismatch_count": count,
        f"{prefix}_mismatch_fraction": float(count / max(left.size, 1)),
    }


def support_metrics(
    actual: Any, reference: Any, *, prefix: str = "support"
) -> dict[str, float]:
    left = _array(actual).astype(bool, copy=False)
    right = _array(reference).astype(bool, copy=False)
    if left.shape != right.shape:
        raise ValueError(f"array shapes differ: {left.shape} != {right.shape}")
    union = np.count_nonzero(left | right)
    intersection = np.count_nonzero(left & right)
    return {
        f"{prefix}_jaccard": float(intersection / union) if union else 1.0,
        f"{prefix}_symmetric_difference_fraction": float(
            np.count_nonzero(left ^ right) / max(left.size, 1)
        ),
        f"{prefix}_fraction": float(left.mean()),
    }


def flux_centroid_metrics(
    actual: Any,
    reference: Any,
    *,
    support: Any | None = None,
) -> dict[str, float]:
    left, right, finite = _common_finite(actual, reference)
    if left.ndim != 2:
        raise ValueError("flux and centroid metrics require 2D arrays")
    if support is not None:
        finite &= _array(support).astype(bool, copy=False)
    y, x = np.indices(left.shape, dtype=np.float64)

    def moments(values: np.ndarray) -> tuple[float, float, float]:
        selected = np.where(finite, values, 0.0).astype(np.float64)
        flux = float(selected.sum(dtype=np.float64))
        denominator = max(abs(flux), np.finfo(np.float64).tiny)
        return (
            flux,
            float((selected * x).sum() / denominator),
            float((selected * y).sum() / denominator),
        )

    actual_flux, actual_x, actual_y = moments(left)
    reference_flux, reference_x, reference_y = moments(right)
    return {
        "flux_relative_error": abs(actual_flux - reference_flux)
        / max(abs(reference_flux), np.finfo(np.float64).tiny),
        "centroid_error_px": math.hypot(actual_x - reference_x, actual_y - reference_y),
        "actual_flux": actual_flux,
        "reference_flux": reference_flux,
    }


def product_metrics(actual: Any, reference: Any) -> dict[str, float | int | bool]:
    metrics: dict[str, float | int | bool] = {}
    metrics.update(
        floating_plane_metrics(actual.science, reference.science, prefix="science")
    )
    if actual.weight is not None and reference.weight is not None:
        metrics.update(
            floating_plane_metrics(actual.weight, reference.weight, prefix="weight")
        )
        metrics.update(
            support_metrics(_array(actual.weight) > 0, _array(reference.weight) > 0)
        )
    actual_variance = dict(actual.variance_components)
    reference_variance = dict(reference.variance_components)
    for name in sorted(set(actual_variance) & set(reference_variance)):
        metrics.update(
            floating_plane_metrics(
                actual_variance[name],
                reference_variance[name],
                prefix=f"variance_{name.lower()}",
            )
        )
    for name in ("dq", "context", "nexp", "rejection"):
        left = getattr(actual, name)
        right = getattr(reference, name)
        if left is not None and right is not None:
            metrics.update(exact_plane_metrics(left, right, prefix=name))
    if _array(actual.science).ndim == 2:
        metrics.update(flux_centroid_metrics(actual.science, reference.science))
    return metrics


def adaptive_metrics(
    actual: Any,
    reference: Any,
    *,
    truth: Any | None = None,
    adjoint_relative_error: float,
) -> dict[str, float]:
    metrics = floating_plane_metrics(actual, reference, prefix="adaptive_peer")
    if truth is not None:
        metrics.update(floating_plane_metrics(actual, truth, prefix="adaptive_truth"))
    metrics["adaptive_adjoint_relative_error"] = float(adjoint_relative_error)
    return metrics


def ifu_cube_metrics(
    cube: Any,
    reference: Any,
    *,
    spectral_axis: Any,
    continuum: float,
    spatial_position: tuple[int, int],
    dq: Any | None = None,
    reference_dq: Any | None = None,
    covariance_variance: float | None = None,
    reference_covariance_variance: float | None = None,
) -> dict[str, float | int | bool]:
    actual = _array(cube).astype(np.float64)
    expected = _array(reference).astype(np.float64)
    if actual.ndim != 3 or actual.shape != expected.shape:
        raise ValueError("IFU cube and reference must be matching 3D arrays")
    wavelength = _array(spectral_axis).astype(np.float64)
    if wavelength.shape != (actual.shape[0],):
        raise ValueError("spectral_axis must match cube depth")
    row, column = spatial_position
    spectrum = actual[:, row, column]
    reference_spectrum = expected[:, row, column]
    line = np.clip(spectrum - float(continuum), 0.0, None)
    reference_line = np.clip(reference_spectrum - float(continuum), 0.0, None)

    def line_moments(values: np.ndarray) -> tuple[float, float, float]:
        widths = np.gradient(wavelength)
        flux = float(np.sum(values * widths))
        total = max(float(np.sum(values)), np.finfo(np.float64).tiny)
        centroid = float(np.sum(wavelength * values) / total)
        variance = float(np.sum(np.square(wavelength - centroid) * values) / total)
        return flux, centroid, 2.354820045 * math.sqrt(max(variance, 0.0))

    flux, centroid, fwhm = line_moments(line)
    reference_flux, reference_centroid, reference_fwhm = line_moments(reference_line)
    edge_count = max(1, min(3, spectrum.size // 4))
    continuum_samples = np.concatenate((spectrum[:edge_count], spectrum[-edge_count:]))
    metrics: dict[str, float | int | bool] = {}
    metrics.update(floating_plane_metrics(actual, expected, prefix="ifu_science"))
    metrics.update(
        support_metrics(
            np.isfinite(actual) & (actual != 0),
            np.isfinite(expected) & (expected != 0),
            prefix="ifu_voxel_support",
        )
    )
    metrics.update(
        {
            "ifu_line_flux_relative_error": abs(flux - reference_flux)
            / max(abs(reference_flux), np.finfo(np.float64).tiny),
            "ifu_line_centroid_error": abs(centroid - reference_centroid),
            "ifu_line_fwhm_error": abs(fwhm - reference_fwhm),
            "ifu_continuum_absolute_error": abs(
                float(np.mean(continuum_samples)) - float(continuum)
            ),
        }
    )
    if dq is not None and reference_dq is not None:
        metrics.update(exact_plane_metrics(dq, reference_dq, prefix="ifu_dq"))
    if covariance_variance is not None and reference_covariance_variance is not None:
        metrics["ifu_covariance_relative_error"] = abs(
            float(covariance_variance) - float(reference_covariance_variance)
        ) / max(abs(float(reference_covariance_variance)), np.finfo(np.float64).tiny)
    return metrics


def required_metric_failures(
    metrics: Mapping[str, Any], required: list[str] | tuple[str, ...]
) -> list[str]:
    failures = []
    for name in required:
        value = metrics.get(name)
        if value is None:
            failures.append(f"required metric unavailable: {name}")
        elif isinstance(value, float) and not math.isfinite(value):
            failures.append(f"required metric non-finite: {name}")
    return failures


def _residual_rmse(value: Any) -> float:
    array = _array(value).astype(np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("residuals must be finite with shape (N, 2)")
    return float(np.sqrt(np.mean(np.sum(np.square(array), axis=1))))


def _binned_bias_norm(
    residual: np.ndarray, feature: np.ndarray, *, bins: int
) -> float:
    if feature.shape != (residual.shape[0],):
        raise ValueError("residual feature length differs")
    edges = np.linspace(float(feature.min()), float(feature.max()), bins + 1)
    if edges[0] == edges[-1]:
        return float(np.linalg.norm(residual.mean(axis=0)))
    index = np.clip(np.digitize(feature, edges[1:-1]), 0, bins - 1)
    norms = [
        float(np.linalg.norm(residual[index == value].mean(axis=0)))
        for value in range(bins)
        if np.any(index == value)
    ]
    return max(norms, default=0.0)


def astrometry_certification_metrics(
    residual_arcsec: Any,
    *,
    exposure_holdout: Any,
    source_holdout: Any,
    epoch_holdout: Any,
    focal_xy: Any,
    color: Any,
    epoch: Any,
    covariance_arcsec2: Any,
    gauge_residual_norm: float,
    gauge_count: int,
    runtime_seconds: float,
    peak_bytes: int,
) -> dict[str, float | int]:
    residual = _array(residual_arcsec).astype(np.float64)
    if residual.ndim != 2 or residual.shape[1] != 2:
        raise ValueError("residual_arcsec must have shape (N, 2)")
    masks = {
        "exposure": _array(exposure_holdout).astype(bool),
        "source": _array(source_holdout).astype(bool),
        "epoch": _array(epoch_holdout).astype(bool),
    }
    if any(mask.shape != (residual.shape[0],) for mask in masks.values()):
        raise ValueError("holdout masks must match residual rows")
    if any(not bool(mask.any()) for mask in masks.values()):
        raise ValueError("every holdout split must contain rows")
    focal = _array(focal_xy).astype(np.float64)
    if focal.shape != residual.shape:
        raise ValueError("focal_xy must match residual shape")
    covariance = _array(covariance_arcsec2).astype(np.float64)
    if covariance.shape != (residual.shape[0], 2, 2):
        raise ValueError("covariance_arcsec2 must have shape (N, 2, 2)")
    solved = np.linalg.solve(covariance, residual[..., None]).squeeze(-1)
    mahalanobis2 = np.einsum("ni,ni->n", residual, solved)
    spatial = np.hypot(focal[:, 0], focal[:, 1])
    metrics: dict[str, float | int] = {
        "astrometry_residual_rmse_arcsec": _residual_rmse(residual),
        "astrometry_heldout_exposure_rmse_arcsec": _residual_rmse(
            residual[masks["exposure"]]
        ),
        "astrometry_heldout_source_rmse_arcsec": _residual_rmse(
            residual[masks["source"]]
        ),
        "astrometry_heldout_epoch_rmse_arcsec": _residual_rmse(
            residual[masks["epoch"]]
        ),
        "astrometry_covariance_95_coverage": float(np.mean(mahalanobis2 <= 5.991464547)),
        "astrometry_spatial_residual_map_max_bias_arcsec": _binned_bias_norm(
            residual, spatial, bins=4
        ),
        "astrometry_color_residual_map_max_bias_arcsec": _binned_bias_norm(
            residual, _array(color).astype(np.float64), bins=4
        ),
        "astrometry_time_residual_map_max_bias_arcsec": _binned_bias_norm(
            residual, _array(epoch).astype(np.float64), bins=4
        ),
        "astrometry_gauge_residual_norm": float(gauge_residual_norm),
        "astrometry_gauge_count": int(gauge_count),
        "astrometry_runtime_seconds": float(runtime_seconds),
        "astrometry_peak_bytes": int(peak_bytes),
        "astrometry_sources_per_second": float(
            residual.shape[0] / max(float(runtime_seconds), np.finfo(np.float64).tiny)
        ),
        "astrometry_bytes_per_row": float(peak_bytes / max(residual.shape[0], 1)),
    }
    return metrics


def harmonic_certification_metrics(
    actual: Any,
    reference: Any,
    *,
    runtime_seconds: float,
    peak_bytes: int,
    prefix: str,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = dict(
        floating_plane_metrics(actual, reference, prefix=prefix)
    )
    metrics[f"{prefix}_runtime_seconds"] = float(runtime_seconds)
    metrics[f"{prefix}_peak_bytes"] = int(peak_bytes)
    return metrics


def pseudocl_certification_metrics(
    estimate: Any,
    truth: Any,
    *,
    standard_b_power: float,
    pure_b_power: float,
    monte_carlo_coverage: float,
) -> dict[str, float]:
    metrics = floating_plane_metrics(estimate, truth, prefix="pseudocl")
    metrics.update(
        {
            "pseudocl_pure_b_leakage_ratio": float(
                pure_b_power
                / max(abs(float(standard_b_power)), np.finfo(np.float64).tiny)
            ),
            "pseudocl_monte_carlo_coverage": float(monte_carlo_coverage),
        }
    )
    return metrics


def deblend_certification_metrics(
    flux: Any,
    truth_flux: Any,
    *,
    position_error_arcsec: Any,
    covariance: Any,
    reference_covariance: Any,
    catastrophic_swaps: Any,
    heldout_log_likelihood: float,
    source_detection: bool,
) -> dict[str, float | int | bool]:
    fitted = _array(flux).astype(np.float64)
    truth = _array(truth_flux).astype(np.float64)
    if fitted.shape != truth.shape or fitted.ndim != 2:
        raise ValueError("flux arrays must be matching (source, band) matrices")
    positive = truth > 0
    relative = np.abs(fitted[positive] - truth[positive]) / truth[positive]
    if fitted.shape[1] < 2 or np.any(fitted[:, :2] <= 0) or np.any(truth[:, :2] <= 0):
        color_error = float("inf")
    else:
        fitted_color = -2.5 * np.log10(fitted[:, 0] / fitted[:, 1])
        truth_color = -2.5 * np.log10(truth[:, 0] / truth[:, 1])
        color_error = float(np.max(np.abs(fitted_color - truth_color)))
    covariance_metrics = floating_plane_metrics(
        covariance, reference_covariance, prefix="deblend_covariance"
    )
    swaps = _array(catastrophic_swaps).astype(bool)
    metrics: dict[str, float | int | bool] = {
        "deblend_flux_median_relative_error": float(np.median(relative)),
        "deblend_flux_max_relative_error": float(np.max(relative)),
        "deblend_color_max_abs_error_mag": color_error,
        "deblend_astrometry_max_error_arcsec": float(
            np.max(_array(position_error_arcsec).astype(np.float64))
        ),
        "deblend_catastrophic_swap_count": int(np.count_nonzero(swaps)),
        "deblend_heldout_log_likelihood": float(heldout_log_likelihood),
        "deblend_source_detection": bool(source_detection),
    }
    metrics.update(covariance_metrics)
    return metrics


__all__ = [
    "adaptive_metrics",
    "astrometry_certification_metrics",
    "deblend_certification_metrics",
    "exact_plane_metrics",
    "floating_plane_metrics",
    "flux_centroid_metrics",
    "ifu_cube_metrics",
    "harmonic_certification_metrics",
    "pseudocl_certification_metrics",
    "product_metrics",
    "required_metric_failures",
    "support_metrics",
]
