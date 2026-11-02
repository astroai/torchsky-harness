"""Execute bounded Phase-0 comparator matrices with fail-closed evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from . import phase0
from .comparator_adapters import (
    ComparatorProducts,
    UnavailableEvidence,
    adapt_stcal_resample_output,
    load_external_association_manifest,
    run_astrodrizzle_adapter,
    run_jwst_cube_build_adapter,
    run_reproject_adaptive_adapter,
    run_stcal_outlier_adapter,
)
from .comparator_metrics import (
    adaptive_metrics,
    exact_plane_metrics,
    flux_centroid_metrics,
    ifu_cube_metrics,
    product_metrics,
    required_metric_failures,
)
from .certification_campaigns import RUNNERS as CERTIFICATION_RUNNERS


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path: Path, *, campaign_id: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise phase0.Phase0ConfigError(f"{path}: matrix root must be an object")
    phase0.validate_campaign_matrix(payload, campaign_id=campaign_id)
    return payload


def _header(
    shape: tuple[int, int], *, scale_deg: float = 1.0 / 3600.0
) -> dict[str, Any]:
    height, width = shape
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (width + 1) / 2.0,
        "CRPIX2": (height + 1) / 2.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -scale_deg,
        "CDELT2": scale_deg,
    }


def _torchsky_product_matrix(
    parameters: Mapping[str, Any],
) -> tuple[ComparatorProducts, ComparatorProducts]:
    import torch
    from torchsky.domain import DomainSpec
    from torchsky.map import DQPolicy
    from torchsky.map.operators import IndexedResponsePlan, apply_response_products

    count = int(parameters["contributors"])
    coefficients = torch.linspace(0.25, 1.0, count, dtype=torch.float64)
    coefficients = coefficients / coefficients.sum()
    source_indices = torch.arange(count, dtype=torch.int64)
    target_indices = torch.zeros(count, dtype=torch.int64)
    plan = IndexedResponsePlan(
        DomainSpec((count,), torch.float64),
        DomainSpec((1,), torch.float64),
        source_indices,
        target_indices,
        coefficients,
        contributor_ids=source_indices,
        default_chunk_size=int(parameters["chunk_size"]),
    )
    science = torch.linspace(-1.0, 2.0, count, dtype=torch.float64)
    variance = {
        "var_poisson": torch.linspace(0.5, 1.5, count, dtype=torch.float64),
        "var_rnoise": torch.linspace(0.2, 0.4, count, dtype=torch.float64),
        "var_flat": torch.linspace(0.01, 0.03, count, dtype=torch.float64),
    }
    dq = torch.tensor([1 << (index % 6) for index in range(count)], dtype=torch.int64)
    products = apply_response_products(
        plan,
        science,
        variance_components=variance,
        dq=dq,
        dq_policy=DQPolicy(reject_bits=0, unknown_bit_policy="preserve"),
        chunk_size=5,
    )
    actual = ComparatorProducts(
        science=products.science.detach().cpu().numpy(),
        weight=np.ones((1,), dtype=np.float64),
        variance_components={
            name: value.detach().cpu().numpy()
            for name, value in products.variance_components.items()
        },
        dq=products.dq.detach().cpu().numpy(),
        context=products.context.detach().cpu().numpy(),
        nexp=products.nexp.detach().cpu().numpy(),
    )
    expected_variance = {
        name: np.asarray([(coefficients.square() @ value).item()])
        for name, value in variance.items()
    }
    reference = adapt_stcal_resample_output(
        {
            "data": np.asarray([(coefficients @ science).item()]),
            "wht": np.ones((1,), dtype=np.float64),
            **expected_variance,
            "dq": np.asarray([63], dtype=np.int64),
            "con": np.asarray([[2**32 - 1], [1]], dtype=np.int64),
            "nexp": np.asarray([count], dtype=np.int32),
        }
    )
    return actual, reference


def _run_drizzle_product(case: Mapping[str, Any]) -> dict[str, Any]:
    actual, reference = _torchsky_product_matrix(case["parameters"])
    return {
        "metrics": product_metrics(actual, reference),
        "metadata": reference.metadata,
    }


def _run_drizzle_outlier(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.map import DQPolicy, Observation, OutlierConfig, detect_outliers
    from torchsky.quality import QualityBit

    parameters = case["parameters"]
    height, width = map(int, parameters["shape"])
    count = int(parameters["exposures"])
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, dtype=torch.float64),
        torch.linspace(-1.0, 1.0, width, dtype=torch.float64),
        indexing="ij",
    )
    truth = 1.0 + 3.0 * torch.exp(-7.0 * (x.square() + y.square()))
    generator = torch.Generator().manual_seed(int(parameters["seed"]))
    images = [
        truth + 0.02 * torch.randn(truth.shape, generator=generator, dtype=truth.dtype)
        for _ in range(count)
    ]
    positions = ((2, 3), (5, 11), (9, 7), (13, 12))
    injected = torch.zeros((count, height, width), dtype=torch.bool)
    for exposure, position in enumerate(positions):
        images[exposure][position] += 50.0
        injected[exposure][position] = True
    variance = torch.full_like(truth, 0.25)
    policy = DQPolicy(
        reject_bits=int(QualityBit.COSMIC_RAY),
        cosmic_ray_bit=int(QualityBit.COSMIC_RAY),
        unknown_bit_policy="preserve",
    )
    observations = [
        Observation(
            data=image,
            wcs=_header((height, width)),
            dq=torch.zeros_like(truth, dtype=torch.int64),
            dq_policy=policy,
            var_rnoise=variance,
        )
        for image in images
    ]
    config = OutlierConfig(
        parity_mode=True,
        snr=tuple(float(value) for value in parameters["snr"]),
        scale=tuple(float(value) for value in parameters["scale"]),
        grow=0,
        dq_policy=policy,
    )
    native_masks, _, logs = detect_outliers(
        observations, config=config, return_logs=True, device="cpu"
    )
    native = torch.stack(native_masks).detach().cpu().numpy()
    peer = np.stack(
        [
            run_stcal_outlier_adapter(
                image.detach().cpu().numpy(),
                np.sqrt(variance.detach().cpu().numpy()),
                truth.detach().cpu().numpy(),
                snr=tuple(float(value) for value in parameters["snr"]),
                scale=tuple(float(value) for value in parameters["scale"]),
            ).rejection
            for image in images
        ]
    )
    metrics: dict[str, Any] = exact_plane_metrics(native, peer, prefix="rejection")
    true = injected.detach().cpu().numpy()
    true_count = max(int(np.count_nonzero(true)), 1)
    clean = ~true
    metrics["cosmic_ray_recall"] = float(np.count_nonzero(native & true) / true_count)
    metrics["clean_false_positive_rate"] = float(
        np.count_nonzero(native & clean) / max(int(np.count_nonzero(clean)), 1)
    )
    clean_stack = torch.stack(
        [
            torch.where(mask, torch.nan, image)
            for mask, image in zip(native_masks, images, strict=True)
        ]
    )
    coadd = torch.nanmean(clean_stack, dim=0)
    metrics.update(flux_centroid_metrics(coadd, truth))
    return {
        "metrics": metrics,
        "metadata": {
            "native_rejection_log": list(logs),
            "stcal_adapter": "stcal.outlier_detection.flag_resampled_crs",
        },
    }


def _run_astrodrizzle_real(case: Mapping[str, Any]) -> dict[str, Any]:
    parameters = case["parameters"]
    fixture = ROOT / str(parameters["fixture"])
    pixfrac = float(parameters["pixfrac"])
    astrodrizzle_options = {
        "context": False,
        "static": False,
        "skysub": False,
        "median": False,
        "blot": False,
        "driz_cr": False,
        "final_bits": 0,
        "final_kernel": "square",
        "final_pixfrac": pixfrac,
        "final_units": "cps",
        "final_wht_type": "EXP",
    }
    with tempfile.TemporaryDirectory(prefix="torchsky-hst-drizzle-") as directory:
        reference = run_astrodrizzle_adapter(
            [fixture],
            Path(directory),
            output_name=str(parameters["output_name"]),
            options=astrodrizzle_options,
        )
    actual = _run_torchsky_hst_product(
        fixture,
        reference,
        pixfrac=pixfrac,
    )
    metrics = _real_hst_drizzle_metrics(actual, reference)
    return {
        "metrics": metrics,
        "metadata": {
            **dict(reference.metadata),
            "scientific_claim": (
                "diagnostic single-exposure HST/WFC3-IR square-drizzle science "
                "comparison on the exact AstroDrizzle output grid, restricted to "
                "shared finite support"
            ),
            "science_units": str(reference.metadata["output_header"]["BUNIT"]),
            "matched_controls": {
                "kernel": "square",
                "pixfrac": pixfrac,
                "output_units": "cps",
                "weight_input": "exposure-time support",
                "dq_mask": "DQ == 0",
                "background_subtraction": False,
                "static_masking": False,
                "cosmic_ray_rejection": False,
                "context": False,
            },
            "unavailable_metrics": {
                "weight_plane_parity": (
                    "AstroDrizzle EXP weight includes drizzle/geometric scaling not "
                    "encoded as an equivalent Torchsky product-plane contract"
                ),
                "dq_rejection_context_parity": (
                    "disabled for this one-exposure bounded science/support comparison"
                ),
            },
            "limitations": [
                "one compact 10x10 WFC3/IR exposure; not a mosaic or survey-scale claim",
                "no sky matching, static-mask, cosmic-ray, blot, rejection, DQ-product, or context parity claim",
                "support compares positive accumulated weights only; raw weight values are not compared",
                "metrics are diagnostic and have no certifying numerical tolerance until independently repeated",
            ],
        },
    }


def _numeric_header(header: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(header)
    for key, value in tuple(normalized.items()):
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        try:
            normalized[key] = int(stripped)
            continue
        except ValueError:
            pass
        try:
            normalized[key] = float(stripped)
        except ValueError:
            pass
    return normalized


def _run_torchsky_hst_product(
    fixture: Path,
    reference: ComparatorProducts,
    *,
    pixfrac: float,
) -> ComparatorProducts:
    import torch
    import torchfits

    try:
        from torchsky.map import coadd
        from torchsky.scene import ImageObservation
    except Exception as exc:
        raise UnavailableEvidence(f"installed Torchsky subject unavailable: {exc}") from exc

    target_header = reference.metadata.get("target_wcs")
    if not isinstance(target_header, Mapping):
        raise UnavailableEvidence("AstroDrizzle target WCS metadata is unavailable")
    source, source_header = torchfits.read(
        str(fixture), hdu="SCI", mmap=False, return_header=True
    )
    dq = torchfits.read(str(fixture), hdu="DQ", mmap=False)
    primary_header = torchfits.read_header(str(fixture), 0)
    source_header = _numeric_header(dict(source_header))
    target_header = _numeric_header(target_header)
    source_unit = str(source_header.get("BUNIT", "")).strip().upper()
    target_unit = str(target_header.get("BUNIT", "")).strip().upper()
    if source_unit != "ELECTRONS/S" or target_unit != source_unit:
        raise UnavailableEvidence(
            f"science units are not identical: source={source_unit!r}, target={target_unit!r}"
        )
    exposure_time = float(primary_header["EXPTIME"])
    source = source.to(dtype=torch.float64)
    valid = torch.isfinite(source) & (dq == 0)
    observation = ImageObservation(
        data=source,
        header=source_header,
        wcs=source_header,
        weight_map=valid.to(dtype=source.dtype) * exposure_time,
        mask=valid,
        weight_is_inverse_variance=False,
    )
    result = coadd(
        [observation],
        target=target_header,
        kernel="drizzle_square",
        semantics="surface_brightness",
        method="weighted_mean",
        pixfrac=pixfrac,
        surface_brightness_area_scaling=False,
        device="cpu",
    )
    return ComparatorProducts(
        science=result.data.detach().cpu().numpy(),
        weight=result.weight.detach().cpu().numpy(),
        metadata={
            "adapter": "torchsky.map.coadd",
            "source_unit": source_unit,
            "exposure_time_seconds": exposure_time,
        },
    )


def _real_hst_drizzle_metrics(
    actual: ComparatorProducts, reference: ComparatorProducts
) -> dict[str, Any]:
    if actual.weight is None or reference.weight is None:
        raise UnavailableEvidence("science support requires both accumulated weight planes")
    left = np.asarray(actual.science, dtype=np.float64)
    right = np.asarray(reference.science, dtype=np.float64)
    if left.shape != right.shape:
        raise UnavailableEvidence(
            f"Torchsky/AstroDrizzle output shapes differ: {left.shape} != {right.shape}"
        )
    actual_support = np.isfinite(left) & (np.asarray(actual.weight) > 0)
    reference_support = np.isfinite(right) & (np.asarray(reference.weight) > 0)
    shared = actual_support & reference_support
    if not bool(shared.any()):
        raise UnavailableEvidence("Torchsky and AstroDrizzle have no shared finite support")
    residual = left[shared] - right[shared]
    reference_rms = max(
        float(np.sqrt(np.mean(np.square(right[shared])))),
        np.finfo(np.float64).tiny,
    )
    union = int(np.count_nonzero(actual_support | reference_support))
    intersection = int(np.count_nonzero(shared))
    metrics: dict[str, Any] = {
        "real_hst_shared_support_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "real_hst_shared_support_relative_error": float(
            np.sqrt(np.mean(np.square(residual))) / reference_rms
        ),
        "real_hst_support_iou": float(intersection / union) if union else 1.0,
        "real_hst_torchsky_finite_fraction": float(np.isfinite(left).mean()),
        "real_hst_astrodrizzle_finite_fraction": float(np.isfinite(right).mean()),
        "real_hst_shared_support_fraction": float(shared.mean()),
    }
    metrics.update(flux_centroid_metrics(left, right, support=shared))
    return metrics


def _synthetic_adaptive_input(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    return (
        0.1
        + 2.0
        * np.exp(
            -((x - 0.48 * shape[1]) ** 2 + (y - 0.52 * shape[0]) ** 2) / (2.0 * 4.5**2)
        )
        + 0.003 * x
        - 0.002 * y
    )


def _crop_cfht(parameters: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    payload = np.load(ROOT / str(parameters["fixture"]), allow_pickle=False)
    image = np.asarray(payload["image"], dtype=np.float64)
    header = json.loads(str(payload["header_json"].item()))
    height, width = map(int, parameters["bounded_shape"])
    y0 = (image.shape[0] - height) // 2
    x0 = (image.shape[1] - width) // 2
    bounded = image[y0 : y0 + height, x0 : x0 + width]
    header["NAXIS1"] = width
    header["NAXIS2"] = height
    header["CRPIX1"] = float(header["CRPIX1"]) - x0
    header["CRPIX2"] = float(header["CRPIX2"]) - y0
    return bounded, header


def _run_adaptive(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.map import AdaptiveResampleConfig, adaptive_reprojection_response
    from torchsky.wcs import WCS

    parameters = case["parameters"]
    if case["kind"] == "adaptive_cfht":
        image, source_header = _crop_cfht(parameters)
        source_shape = image.shape
    else:
        source_shape = tuple(int(value) for value in parameters["shape"])
        image = _synthetic_adaptive_input(source_shape)
        source_header = _header(source_shape)
    target_shape = tuple(
        int(value) for value in parameters.get("target_shape", source_shape)
    )
    target_header = dict(source_header)
    target_header["NAXIS1"] = target_shape[1]
    target_header["NAXIS2"] = target_shape[0]
    shift_x, shift_y = map(float, parameters["target_shift_px"])
    target_header["CRPIX1"] = float(target_header["CRPIX1"]) + shift_x
    target_header["CRPIX2"] = float(target_header["CRPIX2"]) + shift_y
    ratio = float(parameters.get("target_scale_ratio", 1.0))
    for key in ("CDELT1", "CDELT2"):
        if key in target_header:
            target_header[key] = float(target_header[key]) * ratio
    semantics = str(parameters["value_semantics"])
    config = AdaptiveResampleConfig(
        value_semantics=semantics,
        max_edge_error_px=0.01,
        max_jacobian_variation=0.01,
        max_source_step_px=0.75,
        max_depth=8,
        planning_chunk_size=512,
        max_plan_entries=5_000_000,
    )
    source_wcs = WCS(source_header)
    target_wcs = WCS(target_header)
    plan = adaptive_reprojection_response(
        source_wcs,
        target_wcs,
        source_shape,
        target_shape,
        config=config,
        dtype=torch.float64,
        device="cpu",
    )
    source = torch.from_numpy(np.ascontiguousarray(image)).to(torch.float64)
    native = plan.apply(source)
    peer = run_reproject_adaptive_adapter(
        image,
        source_header,
        target_header,
        target_shape,
        conserve_flux=semantics == "integrated_flux",
    )
    trial = torch.linspace(-0.75, 1.25, native.numel(), dtype=torch.float64).reshape(
        target_shape
    )
    left = torch.sum(native * trial)
    right = torch.sum(source * plan.adjoint(trial))
    adjoint_error = float(
        torch.abs(left - right)
        / torch.clamp(torch.maximum(torch.abs(left), torch.abs(right)), min=1.0e-15)
    )
    metrics: dict[str, Any] = adaptive_metrics(
        native,
        peer.science,
        adjoint_relative_error=adjoint_error,
    )
    support = np.asarray(peer.weight) > 0
    metrics.update(flux_centroid_metrics(native, peer.science, support=support))
    return {
        "metrics": metrics,
        "metadata": {
            **dict(peer.metadata),
            "response_entries": int(plan.coefficients.numel()),
            "response_fingerprint": plan.fingerprint.digest,
        },
    }


def _run_ifu_synthetic(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.map import CubeTarget, DQPolicy, IFUObservation, coadd_cube
    from torchsky.wcs import WCS
    from torchsky.wcs import IFUSliceWCS

    parameters = case["parameters"]
    count = int(parameters["spectral_bins"])
    start = float(parameters["wavelength_start"])
    step = float(parameters["wavelength_step"])
    wavelength = torch.arange(count, dtype=torch.float64) * step + start
    line = float(parameters["line_amplitude"]) * torch.exp(
        -0.5
        * (
            (wavelength - float(parameters["line_center"]))
            / float(parameters["line_sigma"])
        ).square()
    )
    spectrum = float(parameters["continuum"]) + line
    shape = (count, 3)
    science = spectrum[:, None].expand(shape).clone()
    spatial_wcs = WCS(_header((3, 3)))
    x = torch.arange(3, dtype=torch.float64)
    ra, dec = spatial_wcs.pixel_to_world(x, torch.ones_like(x))
    trace_offset = (torch.arange(count, dtype=torch.float64) - (count - 1) / 2) * (
        0.01 / 3600.0
    )
    transform = IFUSliceWCS(
        ra[None].expand(shape).clone(),
        dec[None].expand(shape).clone() + trace_offset[:, None],
        wavelength[:, None].expand(shape).clone(),
        torch.zeros(shape, dtype=torch.int64),
    )
    variance = torch.full(shape, 0.25, dtype=torch.float64)
    dq = torch.zeros(shape, dtype=torch.int64)
    dq[count // 2, 1] = 4
    observation = IFUObservation(
        science,
        transform,
        dq=dq,
        dq_policy=DQPolicy(reject_bits=0, unknown_bit_policy="preserve"),
        variance_components={"poisson": variance, "read": 0.2 * variance},
    )
    edges = torch.arange(
        start - 1.0, start - 1.0 + (count + 1) * step, step, dtype=torch.float64
    )
    target = CubeTarget(spatial_wcs, edges, shape=(3, 3))
    result = coadd_cube([observation], target, spectral_kernel="drizzle")
    reference = torch.zeros_like(result.data)
    reference[:, 1, :] = science
    reference_dq = torch.zeros_like(result.dq)
    plan = result.response_plan
    dq_source_index = (count // 2) * shape[1] + 1
    dq_targets = plan.target_indices[plan.source_indices == dq_source_index]
    reference_dq.reshape(-1)[dq_targets] = 4
    query = torch.zeros_like(result.data)
    query[:, 1, 1] = 1.0
    covariance_variance = float(result.covariance.query_variance(query))
    dense = torch.zeros((result.data.numel(), science.numel()), dtype=torch.float64)
    dense.index_put_(
        (plan.target_indices, plan.source_indices), plan.coefficients, accumulate=True
    )
    source_variance = (1.2 * variance).reshape(-1)
    dense_covariance = dense @ torch.diag(source_variance) @ dense.mT
    flat_query = query.reshape(-1)
    reference_covariance_variance = float(flat_query @ dense_covariance @ flat_query)
    metrics = ifu_cube_metrics(
        result.data,
        reference,
        spectral_axis=result.spectral_axis.values,
        continuum=float(parameters["continuum"]),
        spatial_position=(1, 1),
        dq=result.dq,
        reference_dq=reference_dq,
        covariance_variance=covariance_variance,
        reference_covariance_variance=reference_covariance_variance,
    )
    return {
        "metrics": metrics,
        "metadata": {
            "response_entries": int(plan.coefficients.numel()),
            "response_fingerprint": plan.fingerprint.digest,
            "kernel": result.provenance["kernel"],
        },
    }


def _run_jwst_external(case: Mapping[str, Any]) -> dict[str, Any]:
    parameters = case["parameters"]
    manifest_path = ROOT / str(parameters["association_manifest"])
    manifest = load_external_association_manifest(manifest_path)
    product = run_jwst_cube_build_adapter(
        manifest, weighting=str(parameters["weighting"])
    )
    finite = np.isfinite(product.science)
    support = finite & (product.science != 0)
    return {
        "metrics": {
            "jwst_cube_science_finite_fraction": float(finite.mean()),
            "jwst_cube_voxel_support_fraction": float(support.mean()),
            "jwst_cube_has_dq": product.dq is not None,
            "jwst_cube_has_weight": product.weight is not None,
        },
        "metadata": dict(product.metadata),
    }


_RUNNERS = {
    "synthetic_product_matrix": _run_drizzle_product,
    "stcal_outlier": _run_drizzle_outlier,
    "astrodrizzle_real": _run_astrodrizzle_real,
    "adaptive_synthetic": _run_adaptive,
    "adaptive_cfht": _run_adaptive,
    "ifu_synthetic": _run_ifu_synthetic,
    "jwst_cube_build_external": _run_jwst_external,
    **CERTIFICATION_RUNNERS,
}


def _gate_failures(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, gate in gates.items():
        if name not in metrics:
            failures.append(f"gate metric unavailable: {name}")
            continue
        value = metrics[name]
        if "equals" in gate and value != gate["equals"]:
            failures.append(f"{name}={value!r} != {gate['equals']!r}")
        if "max" in gate and float(value) > float(gate["max"]):
            failures.append(f"{name}={value!r} > {gate['max']!r}")
        if "min" in gate and float(value) < float(gate["min"]):
            failures.append(f"{name}={value!r} < {gate['min']!r}")
    return failures


def _peer_status(manifest: Mapping[str, Any], peer_id: str) -> str:
    for record in manifest.get("peers", []):
        if record.get("id") == peer_id:
            return str(record.get("status", "missing"))
    return "missing"


def execute_case(
    case: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    unavailable_peers = [
        peer_id
        for peer_id in case["required_peers"]
        if _peer_status(manifest, peer_id) != "available"
    ]
    if unavailable_peers:
        return {
            "id": case["id"],
            "kind": case["kind"],
            "status": "unavailable",
            "metrics": {},
            "failures": [
                f"required peer unavailable: {peer_id}" for peer_id in unavailable_peers
            ],
        }
    runner = _RUNNERS.get(str(case["kind"]))
    if runner is None:
        return {
            "id": case["id"],
            "kind": case["kind"],
            "status": "failed",
            "metrics": {},
            "failures": [f"no runner for case kind {case['kind']}"],
        }
    try:
        output = runner(case)
    except UnavailableEvidence as exc:
        return {
            "id": case["id"],
            "kind": case["kind"],
            "status": "unavailable",
            "metrics": {},
            "failures": [str(exc)],
        }
    except Exception as exc:
        return {
            "id": case["id"],
            "kind": case["kind"],
            "status": "failed",
            "metrics": {},
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    metrics = dict(output["metrics"])
    failures = required_metric_failures(metrics, tuple(case["required_metrics"]))
    failures.extend(_gate_failures(metrics, case.get("gates", {})))
    return {
        "id": case["id"],
        "kind": case["kind"],
        "status": "passed" if not failures else "failed",
        "metrics": metrics,
        "metadata": dict(output.get("metadata", {})),
        "failures": failures,
    }


def run_campaign(config_path: Path) -> dict[str, Any]:
    config_dir = config_path.parent
    peers = phase0.load_peer_catalog(config_dir / "peers.toml")
    campaign = phase0.load_campaign(config_path, peers)
    if campaign["status"] != "bounded":
        raise phase0.Phase0ConfigError(f"{config_path}: campaign is not bounded")
    matrix_path = (config_dir / campaign["matrix"]).resolve()
    matrix = load_matrix(matrix_path, campaign_id=str(campaign["id"]))
    data_manifest_paths = [
        (config_dir / value).resolve() for value in campaign["data_manifests"]
    ]
    provenance = phase0.build_manifest(
        campaign, peers, data_manifest_paths=data_manifest_paths
    )
    cases = [execute_case(case, provenance) for case in matrix["cases"]]
    failures = list(provenance["coverage"]["failures"])
    for case in cases:
        if case["status"] != "passed":
            failures.extend(f"{case['id']}: {failure}" for failure in case["failures"])
    observed_metrics = {name for case in cases for name in case.get("metrics", {})}
    for name in campaign["required_metrics"]:
        if name not in observed_metrics:
            failures.append(f"campaign required metric unavailable: {name}")
    return {
        "schema_version": 1,
        "campaign_id": campaign["id"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matrix": {
            "id": matrix["id"],
            "path": str(matrix_path),
            "sha256": _sha256(matrix_path),
            "case_count": len(matrix["cases"]),
        },
        "provenance": provenance,
        "cases": cases,
        "certification": {
            "status": "fail" if failures else "pass",
            "failures": failures,
        },
    }


def validate_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise phase0.Phase0ConfigError("bounded result: unsupported schema_version")
    if not isinstance(payload.get("matrix"), dict) or not payload["matrix"].get(
        "sha256"
    ):
        raise phase0.Phase0ConfigError("bounded result: matrix identity missing")
    phase0.validate_manifest(payload.get("provenance", {}))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise phase0.Phase0ConfigError("bounded result: cases must be non-empty")
    for case in cases:
        if not isinstance(case, dict) or case.get("status") not in {
            "passed",
            "failed",
            "unavailable",
        }:
            raise phase0.Phase0ConfigError("bounded result: invalid case status")
        if not isinstance(case.get("metrics"), dict) or not isinstance(
            case.get("failures"), list
        ):
            raise phase0.Phase0ConfigError("bounded result: invalid case evidence")
    certification = payload.get("certification")
    if not isinstance(certification, dict) or certification.get("status") not in {
        "pass",
        "fail",
    }:
        raise phase0.Phase0ConfigError("bounded result: invalid certification")
    if not isinstance(certification.get("failures"), list):
        raise phase0.Phase0ConfigError("bounded result: certification failures missing")


def _config_path(directory: Path, value: str) -> Path:
    path = Path(value)
    return path if path.suffix == ".toml" else directory / f"{value}.toml"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--config-dir", type=Path, default=Path("campaigns/phase0"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--subject-wheel", type=Path)
    args = parser.parse_args(argv)
    if args.subject_wheel is not None:
        wheel = args.subject_wheel.resolve(strict=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ],
            check=True,
        )
    result = run_campaign(_config_path(args.config_dir, args.campaign))
    validate_result(result)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.certify and result["certification"]["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
