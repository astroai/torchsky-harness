#!/usr/bin/env python3
"""Benchmark coupled-calibration graph construction, components, and solve phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import torch

from torchsky_harness.subject import identify_subject
from torchsky.wcs import (
    AstrometricMatchGraph,
    AstrometricObservation,
    AstrometryConfig,
    CoupledCalibrationConfig,
    CoupledCalibrationGraph,
    PhotometricMeasurementBlock,
    PhotometricReference,
    TangentPlaneFrame,
)
from torchsky.wcs.calibration import _PhotometrySolver


T = TypeVar("T")
Anchoring = Literal["all", "none", "alternating"]
RowOrder = Literal["grouped", "shuffled"]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    components: int
    exposures_per_component: int
    sources_per_component: int
    bands: int
    anchoring: Anchoring
    row_order: RowOrder
    seed: int = 20261023

    @property
    def exposure_count(self) -> int:
        return self.components * self.exposures_per_component

    @property
    def source_count(self) -> int:
        return self.components * self.sources_per_component

    @property
    def astrometric_row_count(self) -> int:
        return self.exposure_count * self.sources_per_component

    @property
    def photometric_row_count(self) -> int:
        return self.astrometric_row_count * self.bands


QUICK_CASES = (
    CaseSpec("connected_anchored_grouped", 1, 4, 24, 2, "all", "grouped"),
    CaseSpec("connected_unanchored_shuffled", 1, 4, 24, 2, "none", "shuffled"),
    CaseSpec("many_components_mixed_grouped", 8, 2, 6, 2, "alternating", "grouped"),
    CaseSpec("many_components_mixed_shuffled", 8, 2, 6, 2, "alternating", "shuffled"),
    CaseSpec("many_components_unanchored", 12, 2, 4, 1, "none", "grouped"),
)

FULL_CASES = (
    CaseSpec("connected_anchored_grouped", 1, 16, 128, 3, "all", "grouped"),
    CaseSpec("connected_unanchored_shuffled", 1, 16, 128, 3, "none", "shuffled"),
    CaseSpec("many_components_mixed_grouped", 64, 2, 16, 3, "alternating", "grouped"),
    CaseSpec("many_components_mixed_shuffled", 64, 2, 16, 3, "alternating", "shuffled"),
    CaseSpec("many_components_unanchored", 128, 2, 8, 2, "none", "grouped"),
)

PROFILES = {"quick": QUICK_CASES, "full": FULL_CASES}


@dataclass(frozen=True)
class Fixture:
    observations: tuple[AstrometricObservation, ...]
    photometric_blocks: tuple[PhotometricMeasurementBlock, ...]
    astrometric_reference: dict[str, torch.Tensor]
    photometric_reference: PhotometricReference | None
    tangent_frame: TangentPlaneFrame
    row_exposure: torch.Tensor
    row_source: torch.Tensor
    reference_mask: torch.Tensor
    true_shifts: torch.Tensor


def _world(frame: TangentPlaneFrame, plane_arcsec: torch.Tensor) -> torch.Tensor:
    lon, lat = frame.plane_to_world(
        plane_arcsec[:, 0] / 3600.0, plane_arcsec[:, 1] / 3600.0
    )
    return torch.stack((lon, lat), dim=-1)


def _anchored_components(spec: CaseSpec) -> tuple[int, ...]:
    if spec.anchoring == "all":
        return tuple(range(spec.components))
    if spec.anchoring == "alternating":
        return tuple(range(0, spec.components, 2))
    return ()


def build_fixture(spec: CaseSpec) -> Fixture:
    dtype = torch.float64
    frame = TangentPlaneFrame(15.0, -30.0)
    source_ids = torch.arange(spec.source_count, dtype=torch.int64)
    angles = torch.arange(spec.source_count, dtype=dtype) * 0.7548776662466927
    radii = 4.0 + torch.remainder(torch.arange(spec.source_count, dtype=dtype), 17.0)
    source_plane = torch.stack(
        (radii * torch.cos(angles), radii * torch.sin(angles)), dim=-1
    )
    covariance = torch.eye(2, dtype=dtype) * 0.01**2
    observations: list[AstrometricObservation] = []
    blocks: list[PhotometricMeasurementBlock] = []
    row_exposure: list[torch.Tensor] = []
    row_source: list[torch.Tensor] = []
    shifts: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(spec.seed)
    bands = tuple(f"band-{index}" for index in range(spec.bands))

    exposure_index = 0
    for component in range(spec.components):
        start = component * spec.sources_per_component
        stop = start + spec.sources_per_component
        component_ids = source_ids[start:stop]
        component_plane = source_plane[start:stop]
        for local_exposure in range(spec.exposures_per_component):
            phase = exposure_index + 1
            shift = torch.tensor(
                [0.03 * phase, -0.02 * phase + 0.005 * component], dtype=dtype
            )
            order = torch.arange(spec.sources_per_component)
            if spec.row_order == "shuffled":
                order = torch.randperm(spec.sources_per_component, generator=generator)
            ordered_ids = component_ids[order]
            observed = component_plane[order] - shift
            exposure_id = f"exposure-{exposure_index:05d}"
            observations.append(
                AstrometricObservation(
                    exposure_id,
                    observed,
                    ordered_ids,
                    covariance.expand(spec.sources_per_component, 2, 2),
                    sky_position=_world(frame, observed),
                    covariance_space="tangent",
                )
            )
            for band_index, band in enumerate(bands):
                source_log_flux = 0.01 * ordered_ids.to(dtype) + 0.05 * band_index
                blocks.append(
                    PhotometricMeasurementBlock(
                        exposure_id=exposure_id,
                        source_ids=ordered_ids,
                        band=band,
                        flux=torch.exp(source_log_flux),
                        flux_covariance=torch.full(
                            (spec.sources_per_component,), 1.0e-4, dtype=dtype
                        ),
                    )
                )
            row_exposure.append(
                torch.full(
                    (spec.sources_per_component,), exposure_index, dtype=torch.int64
                )
            )
            row_source.append(ordered_ids)
            shifts.append(shift)
            exposure_index += 1

    anchored_components = _anchored_components(spec)
    anchored_ids = torch.tensor(
        [component * spec.sources_per_component for component in anchored_components],
        dtype=torch.int64,
    )
    reference_mask = torch.zeros(spec.source_count, dtype=torch.bool)
    reference_mask[anchored_ids] = True
    astrometric_reference: dict[str, torch.Tensor] = {}
    photometric_reference = None
    if anchored_ids.numel():
        astrometric_reference = {
            "reference_source_ids": anchored_ids,
            "reference_positions": _world(frame, source_plane[anchored_ids]),
            "reference_covariance": covariance,
        }
        reference_log_flux = (
            0.01 * anchored_ids.to(dtype)[:, None]
            + 0.05 * torch.arange(spec.bands, dtype=dtype)[None, :]
        )
        reference_flux = torch.exp(reference_log_flux)
        photometric_reference = PhotometricReference(
            source_ids=anchored_ids,
            bands=bands,
            flux=reference_flux,
            flux_variance=torch.full_like(reference_flux, 1.0e-4),
        )
    return Fixture(
        observations=tuple(observations),
        photometric_blocks=tuple(blocks),
        astrometric_reference=astrometric_reference,
        photometric_reference=photometric_reference,
        tangent_frame=frame,
        row_exposure=torch.cat(row_exposure),
        row_source=torch.cat(row_source),
        reference_mask=reference_mask,
        true_shifts=torch.stack(shifts),
    )


def _timed(fn: Callable[[], T], runs: int) -> tuple[float, T]:
    samples = []
    result: T | None = None
    for _ in range(runs):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    if result is None:
        raise ValueError("runs must be positive")
    return float(statistics.median(samples)), result


def _component_signature(
    components: tuple[tuple[int, ...], ...],
    exposures: tuple[tuple[int, ...], ...],
    anchored: tuple[bool, ...],
    gauges: tuple[int, ...],
) -> str:
    payload = {
        "components": [list(values) for values in components],
        "exposures": [list(values) for values in exposures],
        "anchored": list(anchored),
        "gauges": list(gauges),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _config() -> CoupledCalibrationConfig:
    return CoupledCalibrationConfig(
        astrometry=AstrometryConfig(
            pose_model="shift",
            max_iterations=3,
            convergence_tolerance=1.0e-10,
            huber_delta=1.0e6,
            covariance_max_parameters=1,
        ),
        max_iterations=1,
        photometry_max_iterations=3,
        convergence_tolerance=1.0e-10,
        photometry_convergence_tolerance=1.0e-10,
        huber_delta=1.0e6,
        fit_extinction=False,
        fit_cloud=False,
        fit_star_flat=False,
        fit_color_terms=False,
        missing_color="zero",
        covariance_max_parameters=1,
    )


def _pose_error_modulo_gauge(
    recovered: torch.Tensor,
    truth: torch.Tensor,
    component_exposures: tuple[tuple[int, ...], ...],
    anchored: tuple[bool, ...],
) -> float:
    differences = []
    for exposures, is_anchored in zip(component_exposures, anchored, strict=True):
        index = torch.tensor(exposures, dtype=torch.int64)
        estimate = recovered[index, :2]
        target = truth[index]
        if not is_anchored:
            estimate = estimate - estimate[:1]
            target = target - target[:1]
        differences.append((estimate - target).reshape(-1))
    return float(torch.sqrt(torch.mean(torch.cat(differences).square())).item())


def run_case(spec: CaseSpec, *, runs: int = 3) -> dict[str, Any]:
    fixture = build_fixture(spec)
    config = _config()

    astrometric_ms, astrometric_graph = _timed(
        lambda: AstrometricMatchGraph(
            fixture.observations,
            tangent_frame=fixture.tangent_frame,
            **fixture.astrometric_reference,
        ),
        runs,
    )
    coupled_ms, graph = _timed(
        lambda: CoupledCalibrationGraph(
            astrometric_graph,
            fixture.photometric_blocks,
            reference=fixture.photometric_reference,
        ),
        runs,
    )
    astrometric_component_ms, astrometric_components = _timed(
        lambda: AstrometricMatchGraph._components(
            spec.exposure_count,
            spec.source_count,
            fixture.row_exposure,
            fixture.row_source,
            fixture.reference_mask,
        ),
        runs,
    )
    photometry_solver = _PhotometrySolver(graph, config)
    component_ms, component_result = _timed(photometry_solver._components, runs)
    solve_ms, solution = _timed(lambda: graph.solve(config), runs)

    components, component_exposures, anchored, gauges = component_result
    ast_components, ast_exposures, ast_anchored = astrometric_components
    if component_exposures != ast_exposures or anchored != ast_anchored:
        raise RuntimeError("astrometric and photometric component topology diverged")
    sizes = [len(values) for values in component_exposures]
    active_residual = solution.photometry.residual_log_flux[
        solution.photometry.fit_mask
    ]
    residual_rms = float(
        torch.sqrt(torch.mean(active_residual.square())).detach().cpu().item()
    )
    signature = _component_signature(components, component_exposures, anchored, gauges)
    astrometric_signature = _component_signature(
        ast_components, ast_exposures, ast_anchored, gauges
    )
    return {
        "case": spec.name,
        "spec": asdict(spec),
        "exposure_count": spec.exposure_count,
        "source_count": spec.source_count,
        "band_count": spec.bands,
        "astrometric_row_count": spec.astrometric_row_count,
        "photometric_row_count": spec.photometric_row_count,
        "component_count": len(components),
        "component_exposure_sizes": sizes,
        "component_exposure_size_min": min(sizes),
        "component_exposure_size_max": max(sizes),
        "component_reference_anchored": list(anchored),
        "gauge_exposure_indices": list(gauges),
        "component_signature": signature,
        "astrometric_component_signature": astrometric_signature,
        "row_order_stable": None,
        "median_ms": {
            "astrometric_graph_construction": astrometric_ms,
            "coupled_association_construction": coupled_ms,
            "graph_construction_total": astrometric_ms + coupled_ms,
            "astrometric_component_discovery": astrometric_component_ms,
            "photometric_component_discovery": component_ms,
            "coupled_solve": solve_ms,
        },
        "throughput_rows_per_second": {
            "graph_construction": 1000.0
            * (spec.astrometric_row_count + spec.photometric_row_count)
            / (astrometric_ms + coupled_ms),
            "component_discovery": 1000.0 * spec.photometric_row_count / component_ms,
            "coupled_solve": 1000.0 * spec.photometric_row_count / solve_ms,
        },
        "deterministic_metrics": {
            "pose_rmse_arcsec_modulo_gauge": _pose_error_modulo_gauge(
                solution.astrometry.pose_parameters,
                fixture.true_shifts,
                component_exposures,
                anchored,
            ),
            "photometric_log_residual_rms": residual_rms,
            "astrometry_gauge_exposures": list(
                solution.astrometry.diagnostics.gauge_exposures
            ),
            "photometry_gauge_exposures": list(
                solution.photometry.diagnostics.gauge_exposures
            ),
            "fit_row_count": int(solution.photometry.fit_mask.sum().item()),
        },
        "peak_rss_bytes": None,
        "peak_rss_unavailable_reason": "case-isolated subprocess measurement required",
    }


def _mark_row_order_stability(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], str] = {}
    for row in rows:
        spec = row["spec"]
        key = (
            spec["components"],
            spec["exposures_per_component"],
            spec["sources_per_component"],
            spec["bands"],
            spec["anchoring"],
            spec["seed"],
        )
        if spec["row_order"] == "grouped":
            grouped[key] = row["component_signature"]
    for row in rows:
        spec = row["spec"]
        key = (
            spec["components"],
            spec["exposures_per_component"],
            spec["sources_per_component"],
            spec["bands"],
            spec["anchoring"],
            spec["seed"],
        )
        baseline = grouped.get(key)
        if baseline is not None:
            row["row_order_stable"] = row["component_signature"] == baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--cases", default="", help="Comma-separated case names")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    selected = {value.strip() for value in args.cases.split(",") if value.strip()}
    available = PROFILES[args.profile]
    unknown = selected - {case.name for case in available}
    if unknown:
        parser.error(f"unknown cases for {args.profile}: {', '.join(sorted(unknown))}")
    cases = tuple(case for case in available if not selected or case.name in selected)
    rows = [run_case(case, runs=args.runs) for case in cases]
    _mark_row_order_stability(rows)
    artifact = {
        "schema_version": 1,
        "benchmark": "wcs_coupled_calibration",
        "profile": args.profile,
        "runs": args.runs,
        "subject": identify_subject(),
        "phase_contract": {
            "fixture_generation_timed": False,
            "graph_construction": "AstrometricMatchGraph plus CoupledCalibrationGraph association validation",
            "component_discovery": "installed _PhotometrySolver._components",
            "solve": "CoupledCalibrationGraph.solve end-to-end",
        },
        "rows": rows,
    }
    print(json.dumps(artifact, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
