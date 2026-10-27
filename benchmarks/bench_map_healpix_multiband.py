#!/usr/bin/env python3
"""Multi-band HEALPix gate for direct-to-HEALPix coaddition.

Three synthetic bands share one sky but use different native pixel scales, which
is the multi-band imaging case: each band is coadded straight onto a common
HEALPix grid, and the bands must agree photometrically and astrometrically. The
gate checks

* exact constant-field recovery per band on a sparse HEALPix grid,
* covered solid angle against the analytic pixel-centre footprint area,
* per-pixel agreement with independent bilinear sampling of a resolved source,
* absolute and cross-band source-position recovery (sub-pixel astrometry),
* band-ratio (SED) and integrated band-flux consistency on shared pixels,
* a band-stacked round trip through ``resample_healpix``.

It does not claim flux-conserving reprojection: ``coadd_healpix`` samples the
image at HEALPix pixel centres, so point-source photometry depends on the
interpolation kernel and on the ratio of HEALPix to image pixel scale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from torchsky.map import Observation, Target, coadd_healpix, resample_healpix
from torchsky.sphere import nside2pixarea, pix2ang
from torchsky.wcs import WCS
from torchsky_harness.metrics import (
    check_map_benchmark_regressions,
    write_map_benchmark_artifacts,
)

ARCSEC_PER_RAD = 206264.80624709636


@dataclass(frozen=True)
class Band:
    name: str
    size: int
    cdelt_deg: float
    constant: float
    amplitude: float


BANDS = (
    Band(name="r", size=40, cdelt_deg=0.04, constant=10.0, amplitude=1.0),
    Band(name="g", size=20, cdelt_deg=0.08, constant=4.0, amplitude=0.5),
    Band(name="i", size=10, cdelt_deg=0.16, constant=1.0, amplitude=0.25),
)
CRVAL = (150.0, 2.0)
SOURCE_CENTRE = (149.75, 1.85)
SOURCE_SIGMA_DEG = 0.5
PLANE_GRADIENT = (0.3, 0.2)
NSIDE = 2048
SPARSE_COVERAGE_NSIDE = 8
PEAK_FIT_THRESHOLD = 0.6


def _header(band: Band) -> dict:
    size = band.size
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (size + 1) / 2.0,
        "CRPIX2": (size + 1) / 2.0,
        "CRVAL1": CRVAL[0],
        "CRVAL2": CRVAL[1],
        "CDELT1": -band.cdelt_deg,
        "CDELT2": band.cdelt_deg,
    }


def _angular_separation_deg(
    ra_deg: torch.Tensor, dec_deg: torch.Tensor
) -> torch.Tensor:
    ra = torch.deg2rad(ra_deg)
    dec = torch.deg2rad(dec_deg)
    centre_ra = math.radians(SOURCE_CENTRE[0])
    centre_dec = math.radians(SOURCE_CENTRE[1])
    cosine = math.sin(centre_dec) * torch.sin(dec) + math.cos(
        centre_dec
    ) * torch.cos(dec) * torch.cos(ra - centre_ra)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def _source_image(band: Band) -> torch.Tensor:
    """Gaussian source of the given band amplitude, sampled at pixel centres."""
    coords = torch.arange(band.size, dtype=torch.float64)
    x, y = torch.meshgrid(coords, coords, indexing="xy")
    ra, dec = WCS(_header(band)).pixel_to_world(x, y)
    separation = _angular_separation_deg(
        ra.to(torch.float64), dec.to(torch.float64)
    )
    profile = torch.exp(-separation.square() / (2.0 * SOURCE_SIGMA_DEG**2))
    return (band.amplitude * profile).to(torch.float32)


def _plane_profile(ra: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
    """Sky plane in degrees about ``CRVAL``, shared by every band."""
    dra = (ra - CRVAL[0]) * math.cos(math.radians(CRVAL[1]))
    ddec = dec - CRVAL[1]
    return 1.0 + PLANE_GRADIENT[0] * dra + PLANE_GRADIENT[1] * ddec


def _plane_image(band: Band) -> torch.Tensor:
    """Band-scaled linear sky slope sampled at pixel centres.

    Bilinear interpolation reproduces a linear sky slope up to the small TAN
    curvature, so recovered deviations come from the WCS-to-HEALPix mapping
    rather than from input discretisation.
    """
    coords = torch.arange(band.size, dtype=torch.float64)
    x, y = torch.meshgrid(coords, coords, indexing="xy")
    ra, dec = WCS(_header(band)).pixel_to_world(x, y)
    profile = _plane_profile(ra.to(torch.float64), dec.to(torch.float64))
    return (band.constant * profile).to(torch.float32)


def _bilinear(image: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Independently sample ``image`` at 0-based pixel coordinates ``x``, ``y``."""
    height, width = int(image.shape[0]), int(image.shape[1])
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    fx = x - x0
    fy = y - y0
    x0i = x0.long().clamp(0, width - 1)
    y0i = y0.long().clamp(0, height - 1)
    x1i = (x0.long() + 1).clamp(0, width - 1)
    y1i = (y0.long() + 1).clamp(0, height - 1)
    values = image.to(torch.float64)
    return (
        values[y0i, x0i] * (1.0 - fx) * (1.0 - fy)
        + values[y0i, x1i] * fx * (1.0 - fy)
        + values[y1i, x0i] * (1.0 - fx) * fy
        + values[y1i, x1i] * fx * fy
    )


def _pixel_centre_footprint_sr(band: Band) -> float:
    """Analytic solid angle of the TAN frame bounded by the outermost pixel centres."""
    extent_rad = math.radians((band.size - 1) * band.cdelt_deg)
    return extent_rad * extent_rad * math.cos(math.radians(CRVAL[1]))


def _coadd(band: Band, image: torch.Tensor, nside: int):
    observation = Observation(
        data=image,
        wcs=WCS(_header(band)),
        weight=torch.ones_like(image),
        band_name=band.name,
    )
    return coadd_healpix(
        [observation],
        nside=nside,
        nest=True,
        sparse=True,
        nside_coverage=SPARSE_COVERAGE_NSIDE,
    )


def _sparse_pixels_and_values(result) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = result.data.valid_pixels.to(torch.int64)
    return pixels, result.data.get_values(pixels).to(torch.float64)


def _lookup(pixels: torch.Tensor, values: torch.Tensor, wanted: torch.Tensor):
    order = torch.searchsorted(pixels, wanted)
    return values[order]


def _common_pixels(pixel_sets: list[torch.Tensor]) -> torch.Tensor:
    common = pixel_sets[0]
    for pixels in pixel_sets[1:]:
        common = common[torch.isin(common, pixels)]
    return common


def _relative_error(value: float, expected: float) -> float:
    if expected == 0.0:
        return abs(value)
    return abs(value - expected) / abs(expected)


def _signal_to_residual_db(residual_scale: float, signal_scale: float) -> float:
    if residual_scale == 0.0:
        return 300.0
    return -20.0 * math.log10(residual_scale / signal_scale)


def _quadratic_peak(
    pixels: torch.Tensor, values: torch.Tensor, nside: int
) -> tuple[float, float]:
    """Sub-pixel peak of a resolved source from a log-quadratic tangent-plane fit."""
    peak = int(torch.argmax(values))
    ra_all, dec_all = pix2ang(nside, pixels, nest=True, lonlat=True)
    peak_ra = float(ra_all[peak])
    peak_dec = float(dec_all[peak])
    selected = values >= PEAK_FIT_THRESHOLD * float(values[peak])
    if int(selected.sum()) < 6:
        return peak_ra, peak_dec
    ra = torch.deg2rad(ra_all[selected].to(torch.float64))
    dec = torch.deg2rad(dec_all[selected].to(torch.float64))
    cos_dec = math.cos(math.radians(peak_dec))
    xi = (ra - math.radians(peak_ra)) * cos_dec * ARCSEC_PER_RAD
    eta = (dec - math.radians(peak_dec)) * ARCSEC_PER_RAD
    design = torch.stack(
        [torch.ones_like(xi), xi, eta, xi * xi, xi * eta, eta * eta], dim=-1
    )
    log_values = torch.log(values[selected])
    solution = torch.linalg.lstsq(design, log_values.unsqueeze(-1)).solution
    _, linear_ra, linear_dec, quad_ra, cross, quad_dec = (
        float(value) for value in solution.squeeze(-1)
    )
    hessian = torch.tensor(
        [[2.0 * quad_ra, cross], [cross, 2.0 * quad_dec]], dtype=torch.float64
    )
    offset = torch.linalg.solve(
        hessian, torch.tensor([-linear_ra, -linear_dec], dtype=torch.float64)
    )
    ra_offset_deg = math.degrees(float(offset[0]) / ARCSEC_PER_RAD / cos_dec)
    dec_offset_deg = math.degrees(float(offset[1]) / ARCSEC_PER_RAD)
    return peak_ra + ra_offset_deg, peak_dec + dec_offset_deg


def _separation_arcsec(ra_a: float, dec_a: float, ra_b: float, dec_b: float) -> float:
    cosine = math.sin(math.radians(dec_a)) * math.sin(math.radians(dec_b)) + math.cos(
        math.radians(dec_a)
    ) * math.cos(math.radians(dec_b)) * math.cos(math.radians(ra_a - ra_b))
    return ARCSEC_PER_RAD * math.acos(min(1.0, max(-1.0, cosine)))


def _row(*, case: str, variant: str, status: str, **metrics: object) -> dict:
    return {
        "case": case,
        "library": "torchsky",
        "method": "coadd_healpix",
        "variant": variant,
        "status": status,
        "skip_time_regression": True,
        **metrics,
    }


def run_flat_multiscale_case(*, nside: int = NSIDE) -> dict:
    """Constant skies at three native pixel scales must coadd exactly."""
    pixel_area = nside2pixarea(nside)
    maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    max_constant_error = 0.0
    max_area_error = 0.0
    max_residual = 0.0
    for band in BANDS:
        image = torch.full(
            (band.size, band.size), float(band.constant), dtype=torch.float32
        )
        pixels, values = _sparse_pixels_and_values(_coadd(band, image, nside))
        if pixels.numel() == 0:
            return _row(
                case="healpix_multiband_flat_multiscale",
                variant=f"nside={nside}",
                status="FAILED",
                failure_reason=f"band {band.name} covers no HEALPix pixel",
            )
        maps[band.name] = (pixels, values)
        residual = float((values - band.constant).abs().max())
        max_residual = max(max_residual, residual)
        max_constant_error = max(max_constant_error, residual / abs(band.constant))
        covered_area = float(pixels.numel()) * pixel_area
        max_area_error = max(
            max_area_error,
            _relative_error(covered_area, _pixel_centre_footprint_sr(band)),
        )

    pixel_sets = [maps[band.name][0] for band in BANDS]
    common = _common_pixels(pixel_sets)
    reference = BANDS[0]
    ref_values = _lookup(*maps[reference.name], common)
    ratio_error = 0.0
    for band in BANDS[1:]:
        expected = band.constant / reference.constant
        recovered = _lookup(*maps[band.name], common) / ref_values
        ratio_error = max(ratio_error, float((recovered - expected).abs().max()))
    common_fraction = float(common.numel()) / float(pixel_sets[0].numel())

    signal = max(band.constant for band in BANDS)
    status = (
        "OK"
        if max_constant_error <= 1.0e-5
        and max_area_error <= 0.05
        and ratio_error <= 1.0e-5
        and common_fraction >= 0.8
        else "FAILED"
    )
    return _row(
        case="healpix_multiband_flat_multiscale",
        variant=f"nside={nside}",
        status=status,
        nside=nside,
        bands=len(BANDS),
        band_pixel_counts={
            band.name: int(maps[band.name][0].numel()) for band in BANDS
        },
        constant_recovery_max_rel_error=max_constant_error,
        coverage_area_max_rel_error=max_area_error,
        common_pixel_fraction=common_fraction,
        band_ratio_max_rel_error=ratio_error,
        reconstruction_rmse=max_residual,
        signal_to_residual_db=_signal_to_residual_db(max_residual, signal),
        total_flux_relative_error=max_constant_error,
    )


def run_sed_registration_case(*, nside: int = NSIDE) -> dict:
    """One resolved source must land at the same sky position in every band."""
    maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    positions: list[tuple[float, float]] = []
    max_rel_residual = 0.0
    max_rms_rel_residual = 0.0
    max_position_offset = 0.0
    for band in BANDS:
        image = _source_image(band)
        pixels, values = _sparse_pixels_and_values(_coadd(band, image, nside))
        if pixels.numel() == 0:
            return _row(
                case="healpix_multiband_sed_registration",
                variant=f"nside={nside}",
                status="FAILED",
                failure_reason=f"band {band.name} covers no HEALPix pixel",
            )
        maps[band.name] = (pixels, values)
        ra, dec = pix2ang(nside, pixels, nest=True, lonlat=True)
        x, y = WCS(_header(band)).world_to_pixel(ra, dec)
        truth = _bilinear(image, x.to(torch.float64), y.to(torch.float64))
        residual = (values - truth).abs()
        max_rel_residual = max(max_rel_residual, float(residual.max()) / band.amplitude)
        max_rms_rel_residual = max(
            max_rms_rel_residual,
            float(residual.square().mean().sqrt()) / band.amplitude,
        )
        band_ra, band_dec = _quadratic_peak(pixels, values, nside)
        positions.append((band_ra, band_dec))
        max_position_offset = max(
            max_position_offset,
            _separation_arcsec(band_ra, band_dec, *SOURCE_CENTRE),
        )

    max_separation = 0.0
    for index, position in enumerate(positions):
        for other in positions[index + 1 :]:
            max_separation = max(
                max_separation, _separation_arcsec(*position, *other)
            )

    pixel_sets = [maps[band.name][0] for band in BANDS]
    common = _common_pixels(pixel_sets)
    reference = BANDS[0]
    ref_values = _lookup(*maps[reference.name], common)
    # Diagnostic only: the coarsest band samples a 0.5 deg source every 0.16 deg,
    # so its integrated flux differs from the fine band by input discretisation.
    flux_ratio_error = 0.0
    for band in BANDS[1:]:
        band_values = _lookup(*maps[band.name], common)
        flux_ratio_error = max(
            flux_ratio_error,
            _relative_error(
                float(band_values.sum() / ref_values.sum()),
                band.amplitude / reference.amplitude,
            ),
        )

    healpix_pixel_deg = math.degrees(math.sqrt(nside2pixarea(nside)))
    signal = max(band.amplitude for band in BANDS)
    status = (
        "OK"
        if max_rel_residual <= 1.0e-5
        and max_rms_rel_residual <= 1.0e-6
        and max_position_offset <= 1.0
        and max_separation <= 0.5
        else "FAILED"
    )
    return _row(
        case="healpix_multiband_sed_registration",
        variant=f"nside={nside}",
        status=status,
        nside=nside,
        bands=len(BANDS),
        source_sigma_deg=SOURCE_SIGMA_DEG,
        healpix_pixel_scale_deg=healpix_pixel_deg,
        max_residual_vs_bilinear_max_rel_error=max_rel_residual,
        max_residual_vs_bilinear_rms_rel_error=max_rms_rel_residual,
        source_position_max_offset_arcsec=max_position_offset,
        band_position_max_separation_arcsec=max_separation,
        reconstruction_rmse=max_rel_residual * signal,
        signal_to_residual_db=_signal_to_residual_db(
            max_rel_residual * signal, signal
        ),
        band_flux_ratio_max_rel_error=flux_ratio_error,
        total_flux_relative_error=flux_ratio_error,
    )


def run_plane_photometry_case(*, nside: int = NSIDE) -> dict:
    """A linear sky slope must give exact per-pixel photometry and band ratios."""
    maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    max_profile_residual = 0.0
    max_residual = 0.0
    for band in BANDS:
        image = _plane_image(band)
        pixels, values = _sparse_pixels_and_values(_coadd(band, image, nside))
        if pixels.numel() == 0:
            return _row(
                case="healpix_multiband_plane_photometry",
                variant=f"nside={nside}",
                status="FAILED",
                failure_reason=f"band {band.name} covers no HEALPix pixel",
            )
        maps[band.name] = (pixels, values)
        ra, dec = pix2ang(nside, pixels, nest=True, lonlat=True)
        truth = band.constant * _plane_profile(
            ra.to(torch.float64), dec.to(torch.float64)
        )
        residual = (values - truth).abs()
        max_residual = max(max_residual, float(residual.max()))
        max_profile_residual = max(
            max_profile_residual, float(residual.max()) / band.constant
        )

    pixel_sets = [maps[band.name][0] for band in BANDS]
    common = _common_pixels(pixel_sets)
    reference = BANDS[0]
    ref_values = _lookup(*maps[reference.name], common)
    ratio_error = 0.0
    flux_ratio_error = 0.0
    for band in BANDS[1:]:
        band_values = _lookup(*maps[band.name], common)
        expected_ratio = band.constant / reference.constant
        ratio_error = max(
            ratio_error,
            float((band_values / ref_values - expected_ratio).abs().max()),
        )
        flux_ratio_error = max(
            flux_ratio_error,
            _relative_error(
                float(band_values.sum() / ref_values.sum()), expected_ratio
            ),
        )

    signal = max(band.constant for band in BANDS)
    status = (
        "OK"
        if max_profile_residual <= 1.0e-5 and ratio_error <= 1.0e-5
        else "FAILED"
    )
    return _row(
        case="healpix_multiband_plane_photometry",
        variant=f"nside={nside}",
        status=status,
        nside=nside,
        bands=len(BANDS),
        plane_gradient_per_deg=list(PLANE_GRADIENT),
        plane_max_rel_residual=max_profile_residual,
        common_pixel_fraction=float(common.numel()) / float(pixel_sets[0].numel()),
        band_ratio_max_rel_error=ratio_error,
        band_flux_ratio_max_rel_error=flux_ratio_error,
        reconstruction_rmse=max_residual,
        signal_to_residual_db=_signal_to_residual_db(max_residual, signal),
        total_flux_relative_error=flux_ratio_error,
    )


def run_roundtrip_case(*, nside: int = NSIDE, target_size: int = 48) -> dict:
    """Band-stack HEALPix coadds back onto one TAN target via ``resample_healpix``."""
    target_band = Band(
        name="target", size=target_size, cdelt_deg=0.03, constant=1.0, amplitude=1.0
    )
    target = Target(wcs=WCS(_header(target_band)), shape=(target_size, target_size))
    cubes: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for band in BANDS:
        image = torch.full(
            (band.size, band.size), float(band.constant), dtype=torch.float32
        )
        cutout = resample_healpix(_coadd(band, image, nside), target)
        cubes.append(cutout.image.to(torch.float64))
        weights.append((cutout.weight > 0.0).to(torch.float64))
    cube = torch.stack(cubes)
    common = torch.stack(weights).prod(dim=0) > 0.0
    if not bool(common.any()):
        return _row(
            case="healpix_multiband_roundtrip_cube",
            variant=f"nside={nside}/{target_size}x{target_size}",
            status="FAILED",
            failure_reason="bands share no resampled target pixel",
        )
    constants = torch.tensor([band.constant for band in BANDS], dtype=torch.float64)
    expected = constants.view(-1, 1, 1).expand_as(cube)
    residual = (cube - expected)[:, common].abs()
    stacks = cube[:, common]
    ratio_error = 0.0
    for index in range(1, len(BANDS)):
        recovered = stacks[index] / stacks[0]
        expected_ratio = BANDS[index].constant / BANDS[0].constant
        ratio_error = max(ratio_error, float((recovered - expected_ratio).abs().max()))
    max_rel_error = float(residual.max()) / max(band.constant for band in BANDS)
    status = "OK" if max_rel_error <= 1.0e-5 and ratio_error <= 1.0e-5 else "FAILED"
    return _row(
        case="healpix_multiband_roundtrip_cube",
        variant=f"nside={nside}/{target_size}x{target_size}",
        status=status,
        nside=nside,
        bands=len(BANDS),
        stacked_cube_shape=list(cube.shape),
        common_coverage_fraction=float(common.numel()) / float(target_size**2),
        roundtrip_max_rel_error=max_rel_error,
        band_ratio_max_rel_error=ratio_error,
        reconstruction_rmse=float(residual.max()),
        signal_to_residual_db=_signal_to_residual_db(
            float(residual.max()), max(band.constant for band in BANDS)
        ),
        total_flux_relative_error=max_rel_error,
    )


def run_case(*, nside: int = NSIDE, target_size: int = 48) -> list[dict]:
    return [
        run_flat_multiscale_case(nside=nside),
        run_sed_registration_case(nside=nside),
        run_plane_photometry_case(nside=nside),
        run_roundtrip_case(nside=nside, target_size=target_size),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=NSIDE)
    parser.add_argument("--target-size", type=int, default=48)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument(
        "--require-complete-baseline",
        action="store_true",
        help="Fail when a required baseline row or metric is missing",
    )
    args = parser.parse_args()
    rows = run_case(nside=args.nside, target_size=args.target_size)
    markdown = write_map_benchmark_artifacts(
        rows,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
    )
    print(markdown)
    if args.json_out is not None:
        print(f"JSON: {args.json_out}")
    if args.baseline_json is not None:
        baseline_rows = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        errors = check_map_benchmark_regressions(
            rows,
            baseline_rows,
            require_complete=args.require_complete_baseline,
        )
        if errors:
            print("\n".join(errors))
            return 1
    return 0 if all(row["status"] == "OK" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
