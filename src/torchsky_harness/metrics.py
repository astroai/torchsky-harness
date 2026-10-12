"""Scientific map metrics and benchmark reports for torchsky harness runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torchsky.map import MapProductResult, MapValueSemantics
from torchsky.map.field import FieldObservation, field_quality_scorecard
from torchsky.map.operators import pixel_area_sr_map
from torchsky.wcs import WCS


def _as_cube(tensor: Tensor) -> Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError("Map product evaluation expects 2D or 3D tensors")

def evaluate_map_product(
    prediction: MapProductResult | Tensor,
    truth: MapProductResult | Tensor,
    *,
    weight: Tensor | None = None,
    header: dict[str, Any] | None = None,
    value_semantics: MapValueSemantics | None = None,
) -> dict[str, float | list[float] | None]:
    if isinstance(prediction, MapProductResult):
        pred_data = prediction.data
        if weight is not None:
            pred_weight = _as_cube(weight)
        elif prediction.weight is not None:
            pred_weight = prediction.weight
        elif prediction.mask is not None:
            pred_weight = prediction.mask.to(dtype=torch.float32)
        elif prediction.uncertainty is not None:
            pred_weight = prediction.uncertainty.inverse_variance
        else:
            pred_weight = torch.isfinite(pred_data).to(dtype=torch.float32)
        header = dict(prediction.header)
        value_semantics = prediction.product.value_semantics
    else:
        pred_data = _as_cube(prediction)
        pred_weight = (
            _as_cube(weight) if weight is not None else torch.ones_like(pred_data)
        )
    if isinstance(truth, MapProductResult):
        truth_data = truth.data
        if header is None:
            header = dict(truth.header)
        if value_semantics is None:
            value_semantics = truth.product.value_semantics
    else:
        truth_data = _as_cube(truth)
    if header is None:
        raise ValueError("evaluate_map_product requires a header")
    if value_semantics is None:
        raise ValueError("evaluate_map_product requires value_semantics")
    pred_data = _as_cube(pred_data).to(dtype=torch.float32)
    truth_data = _as_cube(truth_data).to(dtype=torch.float32, device=pred_data.device)
    pred_weight = _as_cube(pred_weight).to(dtype=torch.float32, device=pred_data.device)
    if pred_data.shape != truth_data.shape or pred_data.shape != pred_weight.shape:
        raise ValueError(
            "Prediction, truth, and weight tensors must share the same shape"
        )
    valid = pred_weight > 0.0
    if not bool(valid.any()):
        zeros: dict[str, float | list[float] | None] = {
            "coverage_fraction": 0.0,
            "zero_support_fraction": 1.0,
            "mean_weight": float(pred_weight.mean().item()),
            "reconstruction_rmse": 0.0,
            "reconstruction_mae": 0.0,
            "reconstruction_p95_abs_err": 0.0,
            "reconstruction_max_abs_err": 0.0,
            "signal_to_residual_ratio": 0.0,
            "signal_to_residual_db": float("-inf"),
            "band_reconstruction_rmse": [0.0 for _ in range(int(pred_data.shape[0]))],
            "band_flux_relative_error_mean": 0.0,
            "band_flux_relative_error_max": 0.0,
            "band_flux_relative_error": [0.0 for _ in range(int(pred_data.shape[0]))],
            "total_flux_truth": 0.0,
            "total_flux_output": 0.0,
            "total_flux_absolute_error": 0.0,
            "total_flux_relative_error": 0.0,
            "blue_red_flux_ratio_truth": None,
            "blue_red_flux_ratio_output": None,
            "blue_red_flux_ratio_absolute_error": None,
            "mid_blue_flux_ratio_truth": None,
            "mid_blue_flux_ratio_output": None,
            "mid_blue_flux_ratio_absolute_error": None,
        }
        return zeros
    residual = pred_data - truth_data
    abs_err = torch.abs(residual)
    band_rmse: list[Tensor] = []
    for band_idx in range(int(pred_data.shape[0])):
        band_valid = valid[band_idx]
        if bool(band_valid.any()):
            band_residual = residual[band_idx][band_valid]
            band_rmse.append(torch.sqrt(torch.mean(torch.square(band_residual))))
        else:
            band_rmse.append(
                torch.zeros((), dtype=torch.float32, device=pred_data.device)
            )
    band_rmse_tensor = torch.stack(band_rmse)
    valid_flat = valid.reshape(-1)
    signal_flat = truth_data.reshape(-1)[valid_flat]
    residual_flat = residual.reshape(-1)[valid_flat]
    abs_flat = abs_err.reshape(-1)[valid_flat]
    signal_rms = float(torch.sqrt(torch.mean(torch.square(signal_flat))).item())
    residual_rms = float(torch.sqrt(torch.mean(torch.square(residual_flat))).item())
    snr_linear = signal_rms / max(residual_rms, 1.0e-12)
    if value_semantics == "surface_brightness":
        area_sr = pixel_area_sr_map(
            WCS(header),
            (int(pred_data.shape[-2]), int(pred_data.shape[-1])),
            device=pred_data.device,
        ).unsqueeze(0)
        truth_flux = truth_data * area_sr
        pred_flux = pred_data * area_sr
    else:
        truth_flux = truth_data
        pred_flux = pred_data
    valid_flux = valid.to(dtype=torch.float32)
    band_flux_truth = torch.sum(truth_flux * valid_flux, dim=(1, 2))
    band_flux_output = torch.sum(pred_flux * valid_flux, dim=(1, 2))
    band_flux_rel_err = torch.abs(band_flux_output - band_flux_truth) / torch.clamp(
        torch.abs(band_flux_truth),
        min=1.0e-8,
    )
    metrics: dict[str, float | list[float] | None] = {
        "coverage_fraction": float(valid.to(dtype=torch.float32).mean().item()),
        "zero_support_fraction": float((~valid).to(dtype=torch.float32).mean().item()),
        "mean_weight": float(pred_weight.mean().item()),
        "reconstruction_rmse": float(
            torch.sqrt(torch.mean(torch.square(residual_flat))).item()
        ),
        "reconstruction_mae": float(torch.mean(abs_flat).item()),
        "reconstruction_p95_abs_err": float(torch.quantile(abs_flat, 0.95).item()),
        "reconstruction_max_abs_err": float(torch.max(abs_flat).item()),
        "signal_to_residual_ratio": snr_linear,
        "signal_to_residual_db": 20.0 * math.log10(max(snr_linear, 1.0e-12)),
        "band_reconstruction_rmse": [
            float(value) for value in band_rmse_tensor.detach().cpu().tolist()
        ],
        "band_flux_relative_error_mean": float(torch.mean(band_flux_rel_err).item()),
        "band_flux_relative_error_max": float(torch.max(band_flux_rel_err).item()),
        "band_flux_relative_error": [
            float(value) for value in band_flux_rel_err.detach().cpu().tolist()
        ],
        "band_reconstruction_rmse_mean": float(torch.mean(band_rmse_tensor).item()),
        "band_reconstruction_rmse_max": float(torch.max(band_rmse_tensor).item()),
        "total_flux_truth": float(band_flux_truth.sum().item()),
        "total_flux_output": float(band_flux_output.sum().item()),
        "total_flux_absolute_error": float(
            torch.abs(band_flux_output.sum() - band_flux_truth.sum()).item()
        ),
        "total_flux_relative_error": float(
            (
                torch.abs(band_flux_output.sum() - band_flux_truth.sum())
                / torch.clamp(torch.abs(band_flux_truth.sum()), min=1.0e-8)
            ).item()
        ),
        "blue_red_flux_ratio_truth": None,
        "blue_red_flux_ratio_output": None,
        "blue_red_flux_ratio_absolute_error": None,
        "mid_blue_flux_ratio_truth": None,
        "mid_blue_flux_ratio_output": None,
        "mid_blue_flux_ratio_absolute_error": None,
    }
    if int(pred_data.shape[0]) >= 2:
        blue_red_truth = float(
            (band_flux_truth[0] / torch.clamp(band_flux_truth[-1], min=1.0e-8)).item()
        )
        blue_red_output = float(
            (band_flux_output[0] / torch.clamp(band_flux_output[-1], min=1.0e-8)).item()
        )
        metrics["blue_red_flux_ratio_truth"] = blue_red_truth
        metrics["blue_red_flux_ratio_output"] = blue_red_output
        metrics["blue_red_flux_ratio_absolute_error"] = abs(
            blue_red_output - blue_red_truth
        )
    if int(pred_data.shape[0]) >= 3:
        mid = int(pred_data.shape[0] // 2)
        mid_blue_truth = float(
            (band_flux_truth[mid] / torch.clamp(band_flux_truth[0], min=1.0e-8)).item()
        )
        mid_blue_output = float(
            (
                band_flux_output[mid] / torch.clamp(band_flux_output[0], min=1.0e-8)
            ).item()
        )
        metrics["mid_blue_flux_ratio_truth"] = mid_blue_truth
        metrics["mid_blue_flux_ratio_output"] = mid_blue_output
        metrics["mid_blue_flux_ratio_absolute_error"] = abs(
            mid_blue_output - mid_blue_truth
        )
    return metrics

def evaluate_map_frontier_regions(
    prediction: MapProductResult | Tensor,
    truth: MapProductResult | Tensor,
    *,
    support_weight: Tensor,
    overlap_count: Tensor | None = None,
    fringe_kernel_size: int = 9,
) -> dict[str, float]:
    if fringe_kernel_size <= 0 or fringe_kernel_size % 2 == 0:
        raise ValueError("fringe_kernel_size must be a positive odd integer")
    pred_data = (
        prediction.data if isinstance(prediction, MapProductResult) else prediction
    )
    truth_data = truth.data if isinstance(truth, MapProductResult) else truth
    pred_cube = _as_cube(pred_data).to(dtype=torch.float32)
    truth_cube = _as_cube(truth_data).to(dtype=torch.float32, device=pred_cube.device)
    support_cube = _as_cube(support_weight).to(
        dtype=torch.float32, device=pred_cube.device
    )
    if pred_cube.shape != truth_cube.shape or pred_cube.shape != support_cube.shape:
        raise ValueError(
            "prediction, truth, and support_weight must share the same shape"
        )
    residual = pred_cube - truth_cube
    sq_residual = torch.square(residual)
    support_mask = support_cube > 0.0
    local_support = _mean_pool_cube(
        support_mask.to(dtype=torch.float32),
        kernel_size=fringe_kernel_size,
    )
    fringe_mask = (~support_mask) & (local_support > 0.0)
    far_mask = (~support_mask) & (local_support <= 0.0)
    metrics: dict[str, float] = {
        "support_single_overlap_reconstruction_rmse": 0.0,
        "support_multi_overlap_reconstruction_rmse": 0.0,
        "fringe_extrapolated_reconstruction_rmse": 0.0,
        "far_extrapolated_reconstruction_rmse": 0.0,
        "support_single_overlap_fraction": 0.0,
        "support_multi_overlap_fraction": 0.0,
        "fringe_extrapolated_fraction": float(
            fringe_mask.to(dtype=torch.float32).mean().item()
        ),
        "far_extrapolated_fraction": float(
            far_mask.to(dtype=torch.float32).mean().item()
        ),
    }
    if overlap_count is not None:
        overlap_cube = _as_cube(overlap_count).to(
            dtype=torch.float32, device=pred_cube.device
        )
        if overlap_cube.shape != pred_cube.shape:
            raise ValueError("overlap_count must match prediction shape")
        single_mask = support_mask & (overlap_cube <= 1.0)
        multi_mask = support_mask & (overlap_cube >= 2.0)
        metrics["support_single_overlap_fraction"] = float(
            single_mask.to(dtype=torch.float32).mean().item()
        )
        metrics["support_multi_overlap_fraction"] = float(
            multi_mask.to(dtype=torch.float32).mean().item()
        )
        if bool(single_mask.any()):
            metrics["support_single_overlap_reconstruction_rmse"] = float(
                torch.sqrt(torch.mean(sq_residual[single_mask])).item()
            )
        if bool(multi_mask.any()):
            metrics["support_multi_overlap_reconstruction_rmse"] = float(
                torch.sqrt(torch.mean(sq_residual[multi_mask])).item()
            )
    if bool(fringe_mask.any()):
        metrics["fringe_extrapolated_reconstruction_rmse"] = float(
            torch.sqrt(torch.mean(sq_residual[fringe_mask])).item()
        )
    if bool(far_mask.any()):
        metrics["far_extrapolated_reconstruction_rmse"] = float(
            torch.sqrt(torch.mean(sq_residual[far_mask])).item()
        )
    return metrics

def _mean_pool_cube(cube: Tensor, *, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return cube
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pooled = F.avg_pool2d(
        cube.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return pooled.squeeze(0)

def evaluate_map_uncertainty(
    inverse_variance: Tensor,
    prediction: MapProductResult | Tensor,
    truth: MapProductResult | Tensor,
    *,
    support_weight: Tensor | None = None,
) -> dict[str, float]:
    pred_data = (
        prediction.data if isinstance(prediction, MapProductResult) else prediction
    )
    truth_data = truth.data if isinstance(truth, MapProductResult) else truth
    pred_cube = _as_cube(pred_data).to(dtype=torch.float32)
    truth_cube = _as_cube(truth_data).to(dtype=torch.float32, device=pred_cube.device)
    ivar_cube = _as_cube(inverse_variance).to(
        dtype=torch.float32, device=pred_cube.device
    )
    if pred_cube.shape != truth_cube.shape or pred_cube.shape != ivar_cube.shape:
        raise ValueError(
            "inverse_variance, prediction, and truth must share the same shape"
        )
    sigma_cube = torch.rsqrt(torch.clamp(ivar_cube, min=1.0e-12))
    abs_err_cube = torch.abs(pred_cube - truth_cube)
    flat_sigma = sigma_cube.reshape(-1)
    flat_abs_err = abs_err_cube.reshape(-1)
    valid = torch.isfinite(flat_sigma) & torch.isfinite(flat_abs_err)
    if not bool(valid.any()):
        return {
            "uncertainty_mean_sigma": 0.0,
            "uncertainty_mean_inverse_variance": 0.0,
            "uncertainty_error_corr": 0.0,
            "uncertainty_top_quartile_abs_err": 0.0,
            "uncertainty_bottom_quartile_abs_err": 0.0,
            "uncertainty_top_bottom_abs_err_ratio": 0.0,
            "uncertainty_support_mean_sigma": 0.0,
            "uncertainty_extrapolated_mean_sigma": 0.0,
            "uncertainty_support_mean_abs_err": 0.0,
            "uncertainty_extrapolated_mean_abs_err": 0.0,
        }
    sigma = flat_sigma[valid]
    abs_err = flat_abs_err[valid]
    sigma_centered = sigma - torch.mean(sigma)
    abs_centered = abs_err - torch.mean(abs_err)
    denom = torch.sqrt(
        torch.sum(torch.square(sigma_centered)) * torch.sum(torch.square(abs_centered))
    )
    corr = (
        float(
            torch.sum(sigma_centered * abs_centered).item()
            / max(float(denom.item()), 1.0e-12)
        )
        if sigma.numel() > 1
        else 0.0
    )
    q25 = torch.quantile(sigma, 0.25)
    q75 = torch.quantile(sigma, 0.75)
    top_mask = sigma >= q75
    bottom_mask = sigma <= q25
    top_abs_err = (
        float(torch.mean(abs_err[top_mask]).item()) if bool(top_mask.any()) else 0.0
    )
    bottom_abs_err = (
        float(torch.mean(abs_err[bottom_mask]).item())
        if bool(bottom_mask.any())
        else 0.0
    )
    metrics: dict[str, float] = {
        "uncertainty_mean_sigma": float(torch.mean(sigma).item()),
        "uncertainty_mean_inverse_variance": float(
            torch.mean(ivar_cube.reshape(-1)[valid]).item()
        ),
        "uncertainty_error_corr": corr,
        "uncertainty_top_quartile_abs_err": top_abs_err,
        "uncertainty_bottom_quartile_abs_err": bottom_abs_err,
        "uncertainty_top_bottom_abs_err_ratio": top_abs_err
        / max(bottom_abs_err, 1.0e-12),
        "uncertainty_support_mean_sigma": 0.0,
        "uncertainty_extrapolated_mean_sigma": 0.0,
        "uncertainty_support_mean_abs_err": 0.0,
        "uncertainty_extrapolated_mean_abs_err": 0.0,
    }
    if support_weight is not None:
        support_cube = _as_cube(support_weight).to(
            dtype=torch.float32, device=pred_cube.device
        )
        if support_cube.shape != pred_cube.shape:
            raise ValueError("support_weight must match prediction shape")
        support_mask = support_cube > 0.0
        extrap_mask = ~support_mask
        if bool(support_mask.any()):
            metrics["uncertainty_support_mean_sigma"] = float(
                torch.mean(sigma_cube[support_mask]).item()
            )
            metrics["uncertainty_support_mean_abs_err"] = float(
                torch.mean(abs_err_cube[support_mask]).item()
            )
        if bool(extrap_mask.any()):
            metrics["uncertainty_extrapolated_mean_sigma"] = float(
                torch.mean(sigma_cube[extrap_mask]).item()
            )
            metrics["uncertainty_extrapolated_mean_abs_err"] = float(
                torch.mean(abs_err_cube[extrap_mask]).item()
            )
    return metrics

def _row_duration_ms(row: Mapping[str, Any]) -> float | None:
    value = row.get("fit_time_ms")
    if value is None:
        value = row.get("time_ms")
    if value is None:
        return None
    return float(value)

def _row_label(row: Mapping[str, Any]) -> str:
    if row.get("type") == "field_recommendation":
        return str(row.get("recommended_model"))
    if row.get("library") == "torchsky_recommendation":
        return str(row.get("recommended_variant"))
    model = row.get("model_kind")
    if model is not None:
        return str(model)
    method = row.get("method", row.get("library", "unknown"))
    variant = row.get("variant")
    if variant is None or variant == "" or variant == "default":
        return str(method)
    return f"{method}:{variant}"

def build_map_benchmark_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    cases = sorted({str(row["case"]) for row in rows if "case" in row})
    lines = [
        "| case | best accuracy | best support accuracy | best SNR | best flux conservation | fastest | recommended stack | recommended field |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        case_rows = [
            row
            for row in rows
            if str(row.get("case")) == case
            and row.get("status", "OK") in {"OK", "INFO"}
        ]
        metric_rows = [
            row
            for row in case_rows
            if row.get("reconstruction_rmse") is not None
            and row.get("type") != "field_recommendation"
        ]
        best_accuracy = (
            min(
                metric_rows,
                key=lambda row: float(row["reconstruction_rmse"]),
            )
            if metric_rows
            else None
        )
        support_metric_rows = [
            row
            for row in metric_rows
            if row.get("support_reconstruction_rmse") is not None
        ]
        best_support_accuracy = (
            min(
                support_metric_rows,
                key=lambda row: float(row["support_reconstruction_rmse"]),
            )
            if support_metric_rows
            else None
        )
        best_snr = (
            max(
                metric_rows,
                key=lambda row: float(row["signal_to_residual_db"]),
            )
            if metric_rows
            else None
        )
        best_flux = (
            min(
                metric_rows,
                key=lambda row: float(row["total_flux_relative_error"]),
            )
            if metric_rows
            else None
        )
        timed_rows = [row for row in metric_rows if _row_duration_ms(row) is not None]
        fastest = (
            min(timed_rows, key=lambda row: float(_row_duration_ms(row)))
            if timed_rows
            else None
        )
        stack_rec = next(
            (
                row
                for row in case_rows
                if row.get("library") == "torchsky_recommendation"
                and row.get("profile") == "default_general"
            ),
            None,
        )
        field_rec = next(
            (
                row
                for row in case_rows
                if row.get("type") == "field_recommendation"
                and row.get("profile") == "default_general"
            ),
            None,
        )

        def fmt_metric(row: Mapping[str, Any] | None, key: str) -> str:
            if row is None:
                return "n/a"
            return f"{_row_label(row)} ({float(row[key]):.4g})"

        def fmt_time(row: Mapping[str, Any] | None) -> str:
            if row is None:
                return "n/a"
            duration = _row_duration_ms(row)
            if duration is None:
                return "n/a"
            return f"{_row_label(row)} ({duration:.3f} ms)"

        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    fmt_metric(best_accuracy, "reconstruction_rmse"),
                    fmt_metric(best_support_accuracy, "support_reconstruction_rmse"),
                    fmt_metric(best_snr, "signal_to_residual_db"),
                    fmt_metric(best_flux, "total_flux_relative_error"),
                    fmt_time(fastest),
                    _row_label(stack_rec) if stack_rec is not None else "n/a",
                    _row_label(field_rec) if field_rec is not None else "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"

def map_benchmark_row_key(row: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("case", "")),
        str(row.get("library", "")),
        str(row.get("type", "")),
        str(row.get("method", "")),
        str(row.get("variant", "")),
        str(row.get("model_kind", "")),
        str(row.get("profile", "")),
    ]
    return "|".join(parts)

def check_map_benchmark_regressions(
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    reconstruction_rmse_tolerance: float = 0.10,
    signal_to_residual_db_tolerance: float = 0.10,
    total_flux_relative_error_tolerance: float = 0.10,
    time_tolerance: float = 0.15,
) -> list[str]:
    baseline_by_key = {map_benchmark_row_key(row): row for row in baseline_rows}
    errors: list[str] = []
    for row in rows:
        key = map_benchmark_row_key(row)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        current_status = str(row.get("status", "OK"))
        baseline_status = str(baseline.get("status", "OK"))
        if baseline_status == "OK" and current_status != "OK":
            errors.append(f"{key}: status regressed from OK to {current_status}")
            continue
        if baseline_status != "OK":
            continue

        def compare_upper(metric: str, tolerance: float) -> None:
            base_value = baseline.get(metric)  # noqa: B023
            current_value = row.get(metric)  # noqa: B023
            if base_value is None or current_value is None:
                return
            allowed = float(base_value) * (1.0 + tolerance)
            if float(current_value) > allowed:
                errors.append(
                    f"{key}: {metric} regressed {float(current_value):.6g} > {allowed:.6g}"  # noqa: B023
                )

        def compare_lower(metric: str, tolerance: float) -> None:
            base_value = baseline.get(metric)  # noqa: B023
            current_value = row.get(metric)  # noqa: B023
            if base_value is None or current_value is None:
                return
            floor = float(base_value) - abs(float(base_value)) * tolerance
            if float(current_value) < floor:
                errors.append(
                    f"{key}: {metric} regressed {float(current_value):.6g} < {floor:.6g}"  # noqa: B023
                )

        compare_upper("reconstruction_rmse", reconstruction_rmse_tolerance)
        compare_lower("signal_to_residual_db", signal_to_residual_db_tolerance)
        compare_upper(
            "total_flux_relative_error",
            total_flux_relative_error_tolerance,
        )
        if not bool(baseline.get("skip_time_regression", False)):
            if (
                baseline.get("fit_time_ms") is not None
                or row.get("fit_time_ms") is not None
            ):
                compare_upper("fit_time_ms", time_tolerance)
            else:
                compare_upper("time_ms", time_tolerance)
    return errors

def write_map_benchmark_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> str:
    markdown = build_map_benchmark_markdown(rows)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    return markdown



def _observation_quality_group_key(
    observation: FieldObservation | None,
    index: int,
) -> str:
    if observation is None:
        return "all"
    if observation.band_name is not None:
        return str(observation.band_name)
    if observation.wavelength is not None:
        return f"lambda={float(observation.wavelength):.8g}"
    if observation.spectral_min is not None or observation.spectral_max is not None:
        return (
            f"range={observation.spectral_min!s}:"
            f"{observation.spectral_max!s}"
        )
    return f"observation_{int(index)}"

def field_quality_acceptance_report(
    per_observation: Sequence[dict[str, float]],
    *,
    observations: Sequence[FieldObservation] | None = None,
    max_weighted_rmse: float | None = None,
    min_residual_snr: float | None = None,
    min_support_fraction: float | None = None,
    min_effective_weight_sum: float | None = None,
    max_abs_psf_residual_alignment: float | None = None,
) -> dict[str, Any]:
    """Evaluate explicit per-observation quality gates with no default thresholds."""
    if observations is not None and len(observations) != len(per_observation):
        raise ValueError("observations must have the same length as per_observation")
    criteria: dict[str, tuple[str, float]] = {}
    if max_weighted_rmse is not None:
        criteria["weighted_rmse"] = ("max", float(max_weighted_rmse))
    if min_residual_snr is not None:
        criteria["residual_snr"] = ("min", float(min_residual_snr))
    if min_support_fraction is not None:
        criteria["support_fraction"] = ("min", float(min_support_fraction))
    if min_effective_weight_sum is not None:
        criteria["effective_weight_sum"] = ("min", float(min_effective_weight_sum))
    if max_abs_psf_residual_alignment is not None:
        criteria["psf_residual_alignment"] = (
            "max_abs",
            float(max_abs_psf_residual_alignment),
        )
    violations: list[dict[str, Any]] = []
    for index, metrics in enumerate(per_observation):
        band = (
            _observation_quality_group_key(observations[index], index)
            if observations is not None
            else None
        )
        for metric, (direction, threshold) in criteria.items():
            if metric not in metrics:
                violations.append(
                    {
                        "index": int(index),
                        "band": band,
                        "metric": metric,
                        "reason": "missing",
                        "threshold": threshold,
                    }
                )
                continue
            value = float(metrics[metric])
            compare_value = abs(value) if direction == "max_abs" else value
            failed = (
                compare_value > threshold
                if direction in {"max", "max_abs"}
                else compare_value < threshold
            )
            if failed:
                violations.append(
                    {
                        "index": int(index),
                        "band": band,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "direction": direction,
                    }
                )
    return {
        "passed": not violations,
        "criteria": {
            metric: {"direction": direction, "threshold": threshold}
            for metric, (direction, threshold) in criteria.items()
        },
        "violations": tuple(violations),
        "scorecard": field_quality_scorecard(
            per_observation,
            observations=observations,
        ),
    }
