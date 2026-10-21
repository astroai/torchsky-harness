"""Bounded astrometry, sphere, and supplied-source certification cases."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import time
import tracemalloc
from typing import Any, Mapping

import numpy as np

from .comparator_adapters import (
    load_external_asset_manifest,
    resolve_external_asset,
    run_car_sht_product_adapter,
    run_supplied_centroid_adapter,
    run_supplied_source_adapter,
)
from .comparator_metrics import (
    astrometry_certification_metrics,
    deblend_certification_metrics,
    harmonic_certification_metrics,
    pseudocl_certification_metrics,
)


ROOT = Path(__file__).resolve().parents[2]


def _timed(callable_: Any) -> tuple[Any, float, int]:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        value = callable_()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, elapsed, peak


def _tangent_world(frame: Any, plane_arcsec: Any) -> Any:
    import torch

    ra, dec = frame.plane_to_world(
        plane_arcsec[:, 0] / 3600.0, plane_arcsec[:, 1] / 3600.0
    )
    return torch.stack((ra, dec), dim=-1)


def _astrometry_observation(
    exposure_id: str,
    frame: Any,
    observed_plane: Any,
    source_ids: Any,
    *,
    sigma: float = 0.02,
    epoch: float | None = None,
    color: Any | None = None,
    dcr_basis: Any | None = None,
    parallax_factor: Any | None = None,
    detector_xy: Any | None = None,
) -> Any:
    import torch
    from torchsky.wcs import AstrometricObservation

    if detector_xy is None:
        detector_xy = observed_plane
    covariance = torch.eye(2, dtype=observed_plane.dtype).expand(
        source_ids.numel(), 2, 2
    )
    return AstrometricObservation(
        exposure_id,
        detector_xy,
        source_ids,
        covariance * sigma**2,
        sky_position=_tangent_world(frame, observed_plane),
        covariance_space="tangent",
        epoch=epoch,
        color=color,
        dcr_basis=dcr_basis,
        parallax_factor=parallax_factor,
    )


def _run_astrometry_synthetic(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.wcs import (
        AstrometricMatchGraph,
        AstrometryConfig,
        ResidualSupport,
        TangentPlaneFrame,
        fit_sparse_gp_residual,
    )

    parameters = case["parameters"]
    fixture = json.loads((ROOT / str(parameters["fixture"])).read_text(encoding="utf-8"))
    dtype = torch.float64
    generator = torch.Generator().manual_seed(int(fixture["seed"]))
    count = int(fixture["source_count"])
    frame = TangentPlaneFrame(*map(float, fixture["tangent_center_deg"]))
    sources = float(fixture["field_radius_arcsec"]) * torch.randn(
        (count, 2), generator=generator, dtype=dtype
    )
    source_ids = torch.arange(count)

    def evaluate() -> dict[str, Any]:
        shifts = torch.tensor(fixture["exposure_shifts_arcsec"], dtype=dtype)
        observations = [
            _astrometry_observation(
                f"global-{index}", frame, sources - shift, source_ids
            )
            for index, shift in enumerate(shifts)
        ]
        graph = AstrometricMatchGraph(
            observations,
            reference_source_ids=source_ids,
            reference_positions=_tangent_world(frame, sources),
            reference_covariance=torch.eye(2, dtype=dtype) * 1.0e-6,
            tangent_frame=frame,
        )
        global_solution = graph.solve(
            AstrometryConfig(
                pose_model="shift",
                huber_delta=10.0,
                convergence_tolerance=1.0e-10,
            )
        )

        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 6, dtype=dtype),
            torch.linspace(-1.0, 1.0, 6, dtype=dtype),
            indexing="ij",
        )
        detector = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        nonlinear = 0.2 * torch.stack(
            (x.square() - y.square(), x * y), dim=-1
        ).reshape(-1, 2)
        local_ids = torch.arange(detector.shape[0])
        local_graph = AstrometricMatchGraph(
            [
                _astrometry_observation(
                    "local",
                    frame,
                    detector,
                    local_ids,
                    detector_xy=detector,
                )
            ],
            reference_source_ids=local_ids,
            reference_positions=_tangent_world(frame, detector + nonlinear),
            reference_covariance=torch.eye(2, dtype=dtype) * 1.0e-6,
            tangent_frame=frame,
        )
        local_solution = local_graph.solve(
            AstrometryConfig(
                pose_model="affine",
                fit_detector_grid=True,
                detector_grid_shape=(4, 4),
                detector_grid_regularization=1.0e-12,
                convergence_tolerance=1.0e-10,
            )
        )

        color = torch.linspace(-1.0, 1.5, count, dtype=dtype)
        dcr_basis = torch.tensor([0.6, 0.8], dtype=dtype).expand(count, 2)
        dcr_truth = float(fixture["dcr_coefficient_arcsec_per_color"])
        dcr_graph = AstrometricMatchGraph(
            [
                _astrometry_observation(
                    "dcr",
                    frame,
                    sources - dcr_truth * color[:, None] * dcr_basis,
                    source_ids,
                    color=color,
                    dcr_basis=dcr_basis,
                )
            ],
            reference_source_ids=source_ids,
            reference_positions=_tangent_world(frame, sources),
            reference_covariance=torch.eye(2, dtype=dtype) * 1.0e-6,
            tangent_frame=frame,
        )
        dcr_solution = dcr_graph.solve(
            AstrometryConfig(
                pose_model="shift",
                fit_dcr=True,
                convergence_tolerance=1.0e-10,
            )
        )

        moving_count = min(count, 12)
        moving_ids = source_ids[:moving_count]
        moving_sources = sources[:moving_count]
        proper_motion = torch.stack(
            (
                torch.linspace(-0.008, 0.012, moving_count, dtype=dtype),
                torch.linspace(0.009, -0.006, moving_count, dtype=dtype),
            ),
            dim=-1,
        )
        parallax = torch.linspace(0.001, 0.006, moving_count, dtype=dtype)
        moving_observations = []
        epoch_values = [2019.0, 2020.0, 2021.0, 2022.0]
        factors = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
        for index, (epoch_value, factor) in enumerate(zip(epoch_values, factors, strict=True)):
            factor_tensor = torch.tensor(factor, dtype=dtype).expand(moving_count, 2)
            observed = (
                moving_sources
                + (epoch_value - 2020.0) * proper_motion
                + parallax[:, None] * factor_tensor
            )
            moving_observations.append(
                _astrometry_observation(
                    f"epoch-{index}",
                    frame,
                    observed,
                    moving_ids,
                    epoch=epoch_value,
                    parallax_factor=factor_tensor,
                )
            )
        moving_graph = AstrometricMatchGraph(
            moving_observations,
            reference_source_ids=moving_ids,
            reference_positions=_tangent_world(frame, moving_sources),
            reference_covariance=torch.eye(2, dtype=dtype) * 1.0e-6,
            reference_proper_motion_mas_per_year=proper_motion * 1000.0,
            reference_proper_motion_covariance=torch.eye(2, dtype=dtype) * 1.0e-4,
            reference_parallax_mas=parallax * 1000.0,
            reference_parallax_variance=torch.full(
                (moving_count,), 1.0e-4, dtype=dtype
            ),
            tangent_frame=frame,
        )
        moving_solution = moving_graph.solve(
            AstrometryConfig(
                pose_model="shift",
                fit_proper_motion=True,
                fit_parallax=True,
                reference_epoch=2020.0,
                huber_delta=100.0,
                convergence_tolerance=1.0e-10,
            )
        )

        moving_training = AstrometricMatchGraph(
            moving_observations[:-1],
            reference_source_ids=moving_ids,
            reference_positions=_tangent_world(frame, moving_sources),
            reference_covariance=torch.eye(2, dtype=dtype) * 1.0e-6,
            reference_proper_motion_mas_per_year=proper_motion * 1000.0,
            reference_proper_motion_covariance=torch.eye(2, dtype=dtype) * 1.0e-4,
            reference_parallax_mas=parallax * 1000.0,
            reference_parallax_variance=torch.full(
                (moving_count,), 1.0e-4, dtype=dtype
            ),
            tangent_frame=frame,
        ).solve(
            AstrometryConfig(
                pose_model="shift",
                fit_proper_motion=True,
                fit_parallax=True,
                reference_epoch=2020.0,
                huber_delta=100.0,
                convergence_tolerance=1.0e-10,
            )
        )

        theta = 4.0e-3
        log_scale = -2.0e-3
        shift = torch.tensor([0.7, -1.1], dtype=dtype)
        rotation = torch.tensor(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=dtype,
        )
        robust_observed = ((sources - shift) @ rotation) / math.exp(log_scale)
        robust_observed[0] += torch.tensor([20.0, -15.0], dtype=dtype)
        robust_graph = AstrometricMatchGraph(
            [
                _astrometry_observation(
                    "robust", frame, robust_observed, source_ids, sigma=0.05
                )
            ],
            reference_source_ids=source_ids,
            reference_positions=_tangent_world(frame, sources),
            reference_covariance=torch.eye(2, dtype=dtype) * 0.01**2,
            tangent_frame=frame,
        )
        robust = robust_graph.solve(
            AstrometryConfig(
                pose_model="rscale",
                huber_delta=2.5,
                convergence_tolerance=1.0e-10,
                max_iterations=20,
            )
        )
        nonrobust = robust_graph.solve(
            AstrometryConfig(
                pose_model="rscale",
                huber_delta=1.0e9,
                convergence_tolerance=1.0e-10,
                max_iterations=20,
            )
        )
        robust_truth = torch.tensor([shift[0], shift[1], theta, log_scale], dtype=dtype)
        robust_error = torch.linalg.vector_norm(robust.pose_parameters[0] - robust_truth)
        nonrobust_error = torch.linalg.vector_norm(
            nonrobust.pose_parameters[0] - robust_truth
        )

        gp_train = torch.linspace(-1.0, 1.0, 18, dtype=dtype)[:, None]
        gp_query = torch.linspace(-0.9, 0.9, 9, dtype=dtype)[:, None]
        gp_residual = 0.04 * torch.stack(
            (torch.sin(2.0 * gp_train[:, 0]), torch.cos(1.5 * gp_train[:, 0])),
            dim=-1,
        )
        gp_truth = 0.04 * torch.stack(
            (torch.sin(2.0 * gp_query[:, 0]), torch.cos(1.5 * gp_query[:, 0])),
            dim=-1,
        )
        gp = fit_sparse_gp_residual(
            gp_train,
            gp_residual,
            torch.eye(2, dtype=dtype).expand(18, 2, 2) * 1.0e-5,
            gp_train[::2],
            length_scale=torch.tensor([0.35], dtype=dtype),
            support=ResidualSupport(
                torch.tensor([-1.1], dtype=dtype),
                torch.tensor([1.1], dtype=dtype),
                2.0,
            ),
        )
        gp_prediction = gp.predict(gp_query)
        learned_error = torch.sqrt(torch.mean((gp_prediction.mean - gp_truth).square()))
        baseline_error = torch.sqrt(torch.mean(gp_truth.square()))

        disconnected = AstrometricMatchGraph(
            [
                _astrometry_observation(
                    "a0", frame, sources[:4], torch.arange(4)
                ),
                _astrometry_observation(
                    "a1", frame, sources[:4] - 1.0, torch.arange(4)
                ),
                _astrometry_observation(
                    "b0", frame, sources[4:8], torch.arange(10, 14)
                ),
                _astrometry_observation(
                    "b1", frame, sources[4:8] + 2.0, torch.arange(10, 14)
                ),
            ],
            tangent_frame=frame,
        ).solve(AstrometryConfig(pose_model="shift"))

        heldout_source_mask = source_ids % 5 == 0
        training_source_mask = ~heldout_source_mask
        heldout_observed = sources - shifts[-1]
        heldout_exposure_solution = AstrometricMatchGraph(
            [
                _astrometry_observation(
                    "heldout-exposure",
                    frame,
                    heldout_observed[training_source_mask],
                    source_ids[training_source_mask],
                )
            ],
            reference_source_ids=source_ids[training_source_mask],
            reference_positions=_tangent_world(
                frame, sources[training_source_mask]
            ),
            reference_covariance=torch.eye(2, dtype=dtype) * 1.0e-6,
            tangent_frame=frame,
        ).solve(AstrometryConfig(pose_model="shift"))
        exposure_residual = (
            heldout_observed[heldout_source_mask]
            + heldout_exposure_solution.pose_parameters[0]
            - sources[heldout_source_mask]
        )
        source_residual = torch.cat(
            tuple(
                (sources - shift)[heldout_source_mask]
                + global_solution.pose_parameters[index]
                - sources[heldout_source_mask]
                for index, shift in enumerate(shifts)
            )
        )
        heldout_epoch_observed = (
            moving_sources
            + (epoch_values[-1] - 2020.0) * proper_motion
            + parallax[:, None]
            * torch.tensor(factors[-1], dtype=dtype).expand(moving_count, 2)
        )
        if moving_training.proper_motion_mas_per_year is None:
            raise RuntimeError("held-out proper-motion solution unavailable")
        if moving_training.parallax_mas is None:
            raise RuntimeError("held-out parallax solution unavailable")
        epoch_prediction = (
            moving_training.source_tangent_positions
            + (epoch_values[-1] - 2020.0)
            * moving_training.proper_motion_mas_per_year
            / 1000.0
            + moving_training.parallax_mas[:, None]
            * torch.tensor(factors[-1], dtype=dtype).expand(moving_count, 2)
            / 1000.0
        )
        epoch_residual = heldout_epoch_observed - epoch_prediction
        residual = torch.cat((exposure_residual, source_residual, epoch_residual))
        exposure_count = exposure_residual.shape[0]
        source_count = source_residual.shape[0]
        epoch_count = epoch_residual.shape[0]
        rows = residual.shape[0]
        focal = torch.cat(
            (
                heldout_observed[heldout_source_mask],
                torch.cat(
                    tuple(
                        (sources - shift)[heldout_source_mask] for shift in shifts
                    )
                ),
                heldout_epoch_observed,
            )
        )
        repeated_color = torch.cat(
            (
                color[heldout_source_mask],
                color[heldout_source_mask].repeat(len(observations)),
                color[:moving_count],
            )
        )
        repeated_epoch = torch.cat(
            (
                torch.full((exposure_count,), 2022.0, dtype=dtype),
                torch.arange(len(observations), dtype=dtype).repeat_interleave(
                    int(heldout_source_mask.sum())
                )
                + 2019.0,
                torch.full((epoch_count,), epoch_values[-1], dtype=dtype),
            )
        )
        holdout_exposure = torch.zeros(rows, dtype=torch.bool)
        holdout_exposure[:exposure_count] = True
        holdout_source = torch.zeros(rows, dtype=torch.bool)
        holdout_source[exposure_count : exposure_count + source_count] = True
        holdout_epoch = torch.zeros(rows, dtype=torch.bool)
        holdout_epoch[-epoch_count:] = True
        covariance = torch.eye(2, dtype=dtype).expand(rows, 2, 2) * 0.02**2
        local_control = local_solution.detector_residual_control
        if local_control is None:
            raise RuntimeError("local astrometry fit returned no detector grid")
        if moving_solution.proper_motion_mas_per_year is None:
            raise RuntimeError("proper-motion solution unavailable")
        if moving_solution.parallax_mas is None:
            raise RuntimeError("parallax solution unavailable")
        if dcr_solution.dcr_coefficients is None:
            raise RuntimeError("DCR solution unavailable")
        return {
            "residual": residual,
            "focal": focal,
            "color": repeated_color,
            "epoch": repeated_epoch,
            "holdout_exposure": holdout_exposure,
            "holdout_source": holdout_source,
            "holdout_epoch": holdout_epoch,
            "covariance": covariance,
            "pose_error": torch.max(torch.abs(global_solution.pose_parameters - shifts)),
            "local_norm": torch.linalg.vector_norm(local_control),
            "dcr_error": torch.max(
                torch.abs(dcr_solution.dcr_coefficients - dcr_truth)
            ),
            "pm_error": torch.max(
                torch.abs(
                    moving_solution.proper_motion_mas_per_year
                    - proper_motion * 1000.0
                )
            ),
            "parallax_error": torch.max(
                torch.abs(moving_solution.parallax_mas - parallax * 1000.0)
            ),
            "robust_improvement": nonrobust_error
            / torch.clamp(robust_error, min=torch.finfo(dtype).tiny),
            "learned_improvement": baseline_error
            / torch.clamp(learned_error, min=torch.finfo(dtype).tiny),
            "gauge_count": len(disconnected.diagnostics.gauge_exposures),
            "gauge_norm": torch.linalg.vector_norm(
                disconnected.pose_for("a0")
            )
            + torch.linalg.vector_norm(disconnected.pose_for("b0")),
        }

    result, elapsed, peak = _timed(evaluate)
    metrics = astrometry_certification_metrics(
        result["residual"],
        exposure_holdout=result["holdout_exposure"],
        source_holdout=result["holdout_source"],
        epoch_holdout=result["holdout_epoch"],
        focal_xy=result["focal"],
        color=result["color"],
        epoch=result["epoch"],
        covariance_arcsec2=result["covariance"],
        gauge_residual_norm=float(result["gauge_norm"]),
        gauge_count=int(result["gauge_count"]),
        runtime_seconds=elapsed,
        peak_bytes=peak,
    )
    metrics.update(
        {
            "astrometry_pose_max_abs_error_arcsec": float(result["pose_error"]),
            "astrometry_local_grid_control_norm": float(result["local_norm"]),
            "astrometry_dcr_abs_error": float(result["dcr_error"]),
            "astrometry_pm_max_abs_error_mas_per_year": float(result["pm_error"]),
            "astrometry_parallax_max_abs_error_mas": float(result["parallax_error"]),
            "astrometry_robust_improvement_ratio": float(
                result["robust_improvement"]
            ),
            "astrometry_learned_improvement_ratio": float(
                result["learned_improvement"]
            ),
        }
    )
    return {
        "metrics": metrics,
        "metadata": {
            "fixture_id": fixture["id"],
            "ablations": [
                "global",
                "local",
                "color",
                "dcr",
                "proper_motion",
                "parallax",
                "robust",
                "learned",
            ],
        },
    }


def _run_astrometry_peer(case: Mapping[str, Any]) -> dict[str, Any]:
    parameters = case["parameters"]
    peer = str(parameters["peer"])
    manifest = load_external_asset_manifest(
        ROOT / str(parameters["asset_manifest"]), expected_family="astrometry-v1"
    )
    product = run_supplied_centroid_adapter(peer, manifest)
    observed = np.asarray(product["observed_tangent_arcsec"], dtype=np.float64)
    predicted = np.asarray(product["predicted_tangent_arcsec"], dtype=np.float64)
    residual = observed - predicted
    return {
        "metrics": {
            "astrometry_peer_residual_rmse_arcsec": float(
                np.sqrt(np.mean(np.sum(np.square(residual), axis=1)))
            ),
            "astrometry_peer_row_count": int(residual.shape[0]),
            "astrometry_peer_detection": False,
        },
        "metadata": {"peer": peer, "manifest_id": manifest["id"]},
    }


def _run_external_asset_identity(case: Mapping[str, Any]) -> dict[str, Any]:
    parameters = case["parameters"]
    family = str(parameters["family"])
    manifest = load_external_asset_manifest(
        ROOT / str(parameters["asset_manifest"]), expected_family=family
    )
    asset = resolve_external_asset(manifest)
    return {
        "metrics": {
            "external_asset_identity_complete": True,
            "external_asset_size_bytes": asset.stat().st_size,
        },
        "metadata": {"manifest_id": manifest["id"], "asset": str(asset)},
    }


def _car_geometry(lmax: int) -> Any:
    from torchsky.sphere.sht import CARGeometry

    nlat = 2 * lmax + 1
    nlon = 4 * lmax + 1
    return CARGeometry.from_header(
        {
            "NAXIS1": nlon,
            "NAXIS2": nlat,
            "CTYPE1": "RA---CAR",
            "CTYPE2": "DEC--CAR",
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRPIX1": 1.0,
            "CRPIX2": (nlat + 1.0) / 2.0,
            "CDELT1": 360.0 / nlon,
            "CDELT2": -180.0 / (nlat - 1),
        },
        spin_convention="iau",
    )


def _random_alm(lmax: int, *, leading: tuple[int, ...] = (), seed: int = 13) -> Any:
    import torch
    from torchsky.sphere import alm_index, alm_size

    generator = torch.Generator().manual_seed(seed)
    real = torch.randn((*leading, alm_size(lmax)), generator=generator, dtype=torch.float64)
    imag = torch.randn((*leading, alm_size(lmax)), generator=generator, dtype=torch.float64)
    for ell in range(lmax + 1):
        imag[..., alm_index(ell, 0, lmax)] = 0.0
    return torch.complex(real, imag)


def _run_sphere_internal(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.sphere.sht import analyze, synthesize

    lmax = int(case["parameters"]["lmax"])
    geometry = _car_geometry(lmax)

    def evaluate() -> tuple[Any, Any, Any, Any]:
        scalar_alm = _random_alm(lmax, seed=17)
        scalar_map = synthesize(scalar_alm, geometry, lmax=lmax).values
        scalar_recovered = analyze(scalar_map, geometry, lmax=lmax).values
        spin_alm = _random_alm(lmax, leading=(2,), seed=23)
        for component in range(2):
            for ell in range(2):
                for m in range(ell + 1):
                    from torchsky.sphere import alm_index

                    spin_alm[component, alm_index(ell, m, lmax)] = 0.0
        spin_map = synthesize(spin_alm, geometry, lmax=lmax, spin=2).values
        spin_recovered = analyze(spin_map, geometry, lmax=lmax, spin=2).values
        return scalar_recovered, scalar_alm, spin_recovered, spin_alm

    result, elapsed, peak = _timed(evaluate)
    scalar_recovered, scalar_alm, spin_recovered, spin_alm = result
    metrics = harmonic_certification_metrics(
        torch.view_as_real(scalar_recovered),
        torch.view_as_real(scalar_alm),
        runtime_seconds=elapsed,
        peak_bytes=peak,
        prefix="car_scalar",
    )
    metrics.update(
        harmonic_certification_metrics(
            torch.view_as_real(spin_recovered),
            torch.view_as_real(spin_alm),
            runtime_seconds=elapsed,
            peak_bytes=peak,
            prefix="car_spin",
        )
    )
    return {
        "metrics": metrics,
        "metadata": {"geometry": "CAR", "lmax": lmax, "spin_convention": "iau"},
    }


def _run_sphere_index(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.catalogs import HealpixIndex

    parameters = case["parameters"]
    count = int(parameters["source_count"])
    generator = torch.Generator().manual_seed(int(parameters["seed"]))
    lon = 360.0 * torch.rand(count, generator=generator, dtype=torch.float64)
    lat = torch.rad2deg(
        torch.asin(2.0 * torch.rand(count, generator=generator, dtype=torch.float64) - 1.0)
    )
    index = HealpixIndex.build(lon, lat, nside=int(parameters["nside"]))
    center_lon = float(lon[count // 3])
    center_lat = float(lat[count // 3])
    radius = float(parameters["radius_deg"])
    indexed = torch.sort(index.query_disc(center_lon, center_lat, radius)).values
    lon1 = torch.deg2rad(torch.tensor(center_lon, dtype=torch.float64))
    lat1 = torch.deg2rad(torch.tensor(center_lat, dtype=torch.float64))
    lon2 = torch.deg2rad(lon)
    lat2 = torch.deg2rad(lat)
    cosine = torch.sin(lat1) * torch.sin(lat2) + torch.cos(lat1) * torch.cos(
        lat2
    ) * torch.cos(lon2 - lon1)
    separation = torch.rad2deg(torch.acos(torch.clamp(cosine, -1.0, 1.0)))
    brute = torch.nonzero(separation <= radius, as_tuple=False).flatten()
    mismatch = torch.sum(indexed != brute) if indexed.numel() == brute.numel() else 1
    return {
        "metrics": {
            "sphere_existing_index_mismatch_count": int(mismatch),
            "sphere_existing_index_candidate_fraction": float(
                indexed.numel() / max(count, 1)
            ),
        },
        "metadata": {"nside": index.nside, "source_count": count},
    }


def _run_mhealpy_parity(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.sphere import healpix
    import mhealpy.pixelfunc.single as mhealpy_single

    parameters = case["parameters"]
    count = int(parameters["sample_count"])
    nside = int(parameters["nside"])
    lon = torch.linspace(0.01, 359.99, count, dtype=torch.float64)
    lat = torch.linspace(-89.0, 89.0, count, dtype=torch.float64)
    actual = healpix.ang2pix(nside, lon, lat, nest=True, lonlat=True).cpu().numpy()
    peer = mhealpy_single.ang2pix(
        nside, lon.numpy(), lat.numpy(), nest=True, lonlat=True
    )
    return {
        "metrics": {
            "sphere_mhealpy_index_mismatch_count": int(
                np.count_nonzero(actual != np.asarray(peer))
            )
        },
        "metadata": {"nside": nside, "sample_count": count},
    }


def _run_healsparse_parity(case: Mapping[str, Any]) -> dict[str, Any]:
    import healsparse
    import torch
    from torchsky.sphere.sparse import SparseHealpixMap

    parameters = case["parameters"]
    nside = int(parameters["nside"])
    nside_coverage = int(parameters["nside_coverage"])
    pixels = torch.arange(0, min(96, 12 * nside * nside), 3, dtype=torch.int64)
    values = torch.linspace(-1.0, 1.0, pixels.numel(), dtype=torch.float64)
    native = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=nside,
        nest=True,
        nside_coverage=nside_coverage,
    )
    peer = healsparse.HealSparseMap.make_empty(
        nside_coverage, nside, np.float64
    )
    peer.update_values_pix(pixels.numpy(), values.numpy(), nest=True)
    query = torch.arange(96, dtype=torch.int64)
    native_values = native.get_values(query).cpu().numpy()
    peer_values = peer.get_values_pix(query.numpy(), nest=True)
    return {
        "metrics": {
            "sphere_healsparse_max_abs_error": float(
                np.max(np.abs(native_values - peer_values))
            ),
            "sphere_healsparse_support_mismatch_count": int(
                np.count_nonzero(np.isfinite(native_values) != np.isfinite(peer_values))
            ),
        },
        "metadata": {"nside": nside, "nside_coverage": nside_coverage},
    }


def _run_sphere_peer_product(case: Mapping[str, Any]) -> dict[str, Any]:
    parameters = case["parameters"]
    peer = str(parameters["peer"])
    manifest = load_external_asset_manifest(
        ROOT / str(parameters["asset_manifest"]), expected_family="sphere-v1"
    )
    product = run_car_sht_product_adapter(peer, manifest)
    metrics = harmonic_certification_metrics(
        product["torchsky_values"],
        product["peer_values"],
        runtime_seconds=float(np.asarray(product["runtime_seconds"]).reshape(-1)[0]),
        peak_bytes=int(np.asarray(product["peak_bytes"]).reshape(-1)[0]),
        prefix="sphere_peer",
    )
    return {"metrics": metrics, "metadata": {"peer": peer, "manifest_id": manifest["id"]}}


def _run_pseudocl(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.sphere import alm_index
    from torchsky.sphere.pseudocl import (
        ApodizedMask,
        BandpowerBinning,
        ModeCouplingWorkspace,
        PseudoClField,
        coupled_spectra,
        pure_alms,
    )
    from torchsky.sphere.sht import synthesize

    parameters = case["parameters"]
    lmax = int(parameters["lmax"])
    samples = int(parameters["monte_carlo_samples"])
    geometry = _car_geometry(lmax)
    latitudes = geometry.latitudes(dtype=torch.float64, device="cpu")
    mask = (0.72 + 0.18 * torch.sin(latitudes)).unsqueeze(1).expand(geometry.shape)
    truth = torch.linspace(0.8, 0.3, lmax + 1, dtype=torch.float64)
    alms = _random_alm(lmax, leading=(samples,), seed=31)
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            index = alm_index(ell, m, lmax)
            scale = math.sqrt(float(truth[ell])) if m == 0 else math.sqrt(float(truth[ell]) / 2.0)
            alms[:, index] = alms[:, index] * scale
    maps = synthesize(alms, geometry, lmax=lmax).values
    field = PseudoClField(maps, mask, geometry, lmax)
    binning = BandpowerBinning.from_edges(list(range(lmax + 2)), lmax)
    workspace = ModeCouplingWorkspace.from_fields(field, None, binning)
    samples_estimate = workspace.decouple(coupled_spectra(field))
    estimate = samples_estimate.mean(0)
    standard_error = samples_estimate.std(0, unbiased=True) / math.sqrt(samples)
    coverage = float(torch.mean((torch.abs(estimate - truth) <= 3.0 * standard_error).to(torch.float64)))

    pure_lmax = max(lmax, 5)
    pure_geometry = _car_geometry(pure_lmax)
    pure_latitudes = pure_geometry.latitudes(dtype=torch.float64, device="cpu")
    pure_mask = torch.cos(pure_latitudes).square().unsqueeze(1).expand(pure_geometry.shape).clone()
    apodized = ApodizedMask.from_mask(pure_mask, pure_geometry, pure_lmax)
    e_alm = _random_alm(pure_lmax, seed=41)
    b_alm = torch.zeros_like(e_alm)
    for ell in range(2):
        for m in range(ell + 1):
            e_alm[alm_index(ell, m, pure_lmax)] = 0.0
    values = synthesize(
        torch.stack((e_alm, b_alm)), pure_geometry, lmax=pure_lmax, spin=2
    ).values
    standard = PseudoClField(values, pure_mask, pure_geometry, pure_lmax, spin=2)
    purified = PseudoClField(
        values,
        pure_mask,
        pure_geometry,
        pure_lmax,
        spin=2,
        apodized_mask=apodized,
        purify_e=True,
        purify_b=True,
    )
    standard_b = float(coupled_spectra(standard)[3, 2:].sum())
    pure_alms(purified)
    pure_b = float(coupled_spectra(purified)[3, 2:].sum())
    metrics = pseudocl_certification_metrics(
        estimate,
        truth,
        standard_b_power=standard_b,
        pure_b_power=pure_b,
        monte_carlo_coverage=coverage,
    )
    return {
        "metrics": metrics,
        "metadata": {"lmax": lmax, "monte_carlo_samples": samples, "pure_lmax": pure_lmax},
    }


def _deblend_header(size: int) -> dict[str, Any]:
    center = 0.5 * (size + 1)
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": center,
        "CRPIX2": center,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
    }


def _run_deblend_synthetic(case: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torchsky.catalogs import fit_forced_photometry
    from torchsky.map.field import FieldObservation, PSFSpec
    from torchsky.scene import SourceCatalogScene
    from torchsky.wcs import WCS

    parameters = case["parameters"]
    fixture = json.loads((ROOT / str(parameters["fixture"])).read_text(encoding="utf-8"))
    size = int(fixture["image_size"])
    header = _deblend_header(size)
    pixels = torch.tensor(fixture["source_pixels"], dtype=torch.float64)
    lon, lat = WCS(header).pixel_to_world(pixels[:, 0], pixels[:, 1])
    templates = torch.tensor(fixture["templates"], dtype=torch.float64)
    truth_flux = torch.tensor(fixture["flux"], dtype=torch.float64)
    truth = SourceCatalogScene(
        lon,
        lat,
        component_ids=torch.tensor(fixture["component_ids"]),
        morphology="template",
        fixed_templates=templates,
        template_pixel_scale_arcsec=float(fixture["template_pixel_scale_arcsec"]),
        coefficients=truth_flux,
        band_labels=("g", "r"),
    )
    observations = []
    heldout_masks = []
    for index, band in enumerate(("g", "r")):
        image = torch.zeros((size, size), dtype=torch.float64)
        blank = FieldObservation(
            image=image,
            weight=torch.full_like(image, float(fixture["weight"])),
            header=header,
            band_name=band,
            psf_spec=PSFSpec.gaussian(float(fixture["psf_sigma"][index])),
        )
        rendered = truth.render_observation(blank)
        generator = torch.Generator().manual_seed(int(fixture["seed"]) + index)
        noise = torch.randn(rendered.shape, generator=generator, dtype=rendered.dtype) / math.sqrt(float(fixture["weight"]))
        observations.append(replace(blank, image=rendered + noise))
        holdout = torch.zeros_like(image, dtype=torch.bool)
        holdout[index::4, (index + 1) :: 4] = True
        heldout_masks.append(holdout)
    initial = SourceCatalogScene(
        lon,
        lat,
        component_ids=truth.component_ids,
        morphology="template",
        fixed_templates=templates,
        template_pixel_scale_arcsec=float(fixture["template_pixel_scale_arcsec"]),
        coefficients=torch.ones_like(truth_flux),
        band_labels=("g", "r"),
    )
    training_observations = [
        replace(
            observation,
            weight=torch.where(
                holdout,
                torch.zeros_like(observation.weight),
                observation.weight,
            ),
        )
        for observation, holdout in zip(observations, heldout_masks, strict=True)
    ]
    result = fit_forced_photometry(
        training_observations,
        initial,
        background="none",
    )
    covariance = result.covariance.to_dense()
    columns = []
    for parameter in range(initial.parameter_count):
        state = torch.zeros(initial.parameter_count, dtype=torch.float64)
        state[parameter] = 1.0
        rendered = [
            initial.render_observation(observation, parameter_values=state)
            for observation in observations
        ]
        columns.append(
            torch.cat(
                [
                    value[~holdout].reshape(-1)
                    for value, holdout in zip(
                        rendered, heldout_masks, strict=True
                    )
                ]
            )
        )
    design = torch.stack(columns, dim=1)
    training_weight = torch.cat(
        [
            observation.weight[~holdout].reshape(-1)
            for observation, holdout in zip(
                observations, heldout_masks, strict=True
            )
        ]
    )
    reference_covariance = torch.linalg.inv(
        design.mT @ (design * training_weight[:, None])
    )
    position_error = 3600.0 * torch.sqrt(
        ((result.scene.lon - lon) * torch.cos(torch.deg2rad(lat))).square()
        + (result.scene.lat - lat).square()
    )
    swaps = torch.zeros(result.scene.component_count, dtype=torch.bool)
    heldout = sum(
        -0.5
        * torch.sum(
            observation.weight[mask]
            * torch.square(observation.image[mask] - prediction[mask])
        )
        for observation, prediction, mask in zip(
            observations, result.predictions, heldout_masks, strict=True
        )
    )
    metrics = deblend_certification_metrics(
        result.catalog.columns["flux"],
        truth_flux,
        position_error_arcsec=position_error,
        covariance=covariance,
        reference_covariance=reference_covariance,
        catastrophic_swaps=swaps,
        heldout_log_likelihood=float(heldout),
        source_detection=False,
    )
    metrics["deblend_component_count"] = result.scene.component_count
    metrics["deblend_source_birth_death"] = False
    return {
        "metrics": metrics,
        "metadata": {
            "fixture_id": fixture["id"],
            "selection": "caller-supplied-known-sources",
            "detection": False,
        },
    }


def _run_deblend_peer(case: Mapping[str, Any]) -> dict[str, Any]:
    parameters = case["parameters"]
    peer = str(parameters["peer"])
    manifest = load_external_asset_manifest(
        ROOT / str(parameters["asset_manifest"]), expected_family="deblend-v1"
    )
    product = run_supplied_source_adapter(peer, manifest)
    required = {
        "truth_flux",
        "truth_lon_deg",
        "truth_lat_deg",
        "reference_covariance",
        "heldout_log_likelihood",
        "catastrophic_swaps",
    }
    missing = sorted(required - set(product))
    if missing:
        from .comparator_adapters import UnavailableEvidence

        raise UnavailableEvidence(
            "supplied-source product missing truth arrays: " + ", ".join(missing)
        )
    lon = np.asarray(product["lon_deg"], dtype=np.float64)
    lat = np.asarray(product["lat_deg"], dtype=np.float64)
    truth_lon = np.asarray(product["truth_lon_deg"], dtype=np.float64)
    truth_lat = np.asarray(product["truth_lat_deg"], dtype=np.float64)
    position_error = 3600.0 * np.hypot(
        (lon - truth_lon) * np.cos(np.deg2rad(truth_lat)), lat - truth_lat
    )
    metrics = deblend_certification_metrics(
        product["flux"],
        product["truth_flux"],
        position_error_arcsec=position_error,
        covariance=product["covariance"],
        reference_covariance=product["reference_covariance"],
        catastrophic_swaps=product["catastrophic_swaps"],
        heldout_log_likelihood=float(
            np.asarray(product["heldout_log_likelihood"]).reshape(-1)[0]
        ),
        source_detection=False,
    )
    return {"metrics": metrics, "metadata": {"peer": peer, "manifest_id": manifest["id"]}}


RUNNERS = {
    "astrometry_synthetic_v1": _run_astrometry_synthetic,
    "astrometry_supplied_centroid_peer": _run_astrometry_peer,
    "external_asset_identity": _run_external_asset_identity,
    "sphere_car_internal": _run_sphere_internal,
    "sphere_existing_index": _run_sphere_index,
    "sphere_mhealpy_parity": _run_mhealpy_parity,
    "sphere_healsparse_parity": _run_healsparse_parity,
    "sphere_car_peer_product": _run_sphere_peer_product,
    "sphere_pseudocl_internal": _run_pseudocl,
    "deblend_synthetic_v1": _run_deblend_synthetic,
    "deblend_supplied_source_peer": _run_deblend_peer,
}


__all__ = ["RUNNERS"]
