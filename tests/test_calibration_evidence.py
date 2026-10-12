from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import torchsky
from torchfits import HDUList, Header, TensorHDU, read

from torchsky_harness.calibration_command import (
    _write_json_atomic,
    build_calibration_manifest,
)
from torchsky_harness.calibration_evidence import (
    discover_detector_calibration_positions,
    extract_calibration_evidence,
    extract_reviewed_calibration_evidence,
    read_reviewed_sky_catalog,
)


def _write_jwst_exposure(
    path: Path,
    *,
    band: str = "F277W",
    detector: str = "NRCALONG",
    patt_num: int = 3,
) -> None:
    primary_header = {
        "TELESCOP": "JWST",
        "INSTRUME": "NIRCAM",
        "FILTER": band,
        "DETECTOR": detector,
        "PATT_NUM": patt_num,
        "SUBSTRT1": 101,
        "SUBSTRT2": 201,
        "SUBSIZE1": 10,
        "SUBSIZE2": 8,
        "EXPSTART": 60000.0,
        "EXPMID": 60000.0005,
        "EXPEND": 60000.001,
    }
    science_header = {
        "EXTNAME": "SCI",
        "MJD-BEG": 60000.0,
        "MJD-MID": 60000.0005,
        "MJD-END": 60000.001,
    }
    HDUList(
        [
            TensorHDU(torch.zeros((1, 1), dtype=torch.uint8), Header(primary_header)),
            TensorHDU(torch.zeros((8, 10), dtype=torch.float32), Header(science_header)),
        ]
    ).write(str(path))


def test_calibration_manifest_reads_torchfits_headers_and_evidence(tmp_path: Path) -> None:
    exposure = tmp_path / "jw_test_nrcalong_cal.fits"
    _write_jwst_exposure(exposure)
    evidence = tmp_path / "evidence.csv"
    with evidence.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "exposure",
                "source_id",
                "detector_x",
                "detector_y",
                "role",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "exposure": exposure.name,
                "source_id": "star-1",
                "detector_x": 104.0,
                "detector_y": 203.0,
                "role": "psf",
            }
        )

    manifest = build_calibration_manifest(
        [str(exposure)], evidence_path=evidence, coverage_rtol=0.05
    )

    row = manifest["exposures"][0]
    assert row["band"] == "F277W"
    assert row["split"] == "validation"
    assert row["detector_bounds_xy"] == [100, 109, 200, 207]
    assert row["timing"]["midpoint_source"] == "SCI:MJD-MID"
    assert row["instrument"]["empirical_psf_fwhm_arcsec"] == 0.092
    assert manifest["evidence"][0]["split"] == "validation"
    assert manifest["coverage"][0]["role"] == "psf"
    assert len(manifest["manifest_sha256"]) == 64


def test_calibration_manifest_rejects_out_of_bounds_evidence(tmp_path: Path) -> None:
    exposure = tmp_path / "jw_test_nrcalong_cal.fits"
    _write_jwst_exposure(exposure)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            [
                {
                    "exposure": exposure.name,
                    "detector_x": 120,
                    "detector_y": 203,
                    "role": "background",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside"):
        build_calibration_manifest([str(exposure)], evidence_path=evidence)


def test_calibration_manifest_atomic_write_requires_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    _write_json_atomic(output, {"ok": True}, overwrite=False)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_atomic(output, {"ok": False}, overwrite=False)
    _write_json_atomic(output, {"ok": False}, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": False}


def _write_calibration_image(path: Path) -> tuple[dict[str, object], torch.Tensor]:
    size = 25
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    science_data = 10.0 + 100.0 * torch.exp(
        -0.5 * ((xx - 12.0) ** 2 + (yy - 12.0) ** 2)
    )
    science_data = science_data.to(torch.float32)
    primary_header = {
        "TELESCOP": "JWST",
        "INSTRUME": "NIRCAM",
        "FILTER": "F277W",
        "DETECTOR": "NRCALONG",
        "PATT_NUM": 1,
        "SUBSTRT1": 1,
        "SUBSTRT2": 1,
        "SUBSIZE1": size,
        "SUBSIZE2": size,
        "EXPSTART": 60000.0,
        "EXPMID": 60000.0005,
        "EXPEND": 60000.001,
    }
    science_header: dict[str, object] = {
        "EXTNAME": "SCI",
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 13.0,
        "CRPIX2": 13.0,
        "CRVAL1": 110.82,
        "CRVAL2": -73.45,
        "CD1_1": -0.00001,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.00001,
        "MJD-MID": 60000.0005,
        "BUNIT": "MJy/sr",
    }
    error_header = {"EXTNAME": "ERR"}
    dq_header = {"EXTNAME": "DQ"}
    HDUList(
        [
            TensorHDU(torch.zeros((1, 1), dtype=torch.uint8), Header(primary_header)),
            TensorHDU(science_data, Header(science_header)),
            TensorHDU(torch.ones((size, size), dtype=torch.float32), Header(error_header)),
            TensorHDU(torch.zeros((size, size), dtype=torch.int16), Header(dq_header)),
        ]
    ).write(str(path))
    return science_header, science_data


def test_extract_reviewed_calibration_evidence_persists_split_targets(
    tmp_path: Path,
) -> None:
    exposure = tmp_path / "jw_cal.fits"
    header, _ = _write_calibration_image(exposure)
    wcs = torchsky.WCS(header)
    world = wcs.pixel_to_world(
        torch.tensor([[12.0, 12.0], [5.0, 5.0]], dtype=torch.float64), origin=0
    )
    catalog = tmp_path / "reviewed.csv"
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_id", "role", "ra_deg", "dec_deg", "reviewed"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "star-1",
                "role": "psf",
                "ra_deg": float(world[0, 0]),
                "dec_deg": float(world[0, 1]),
                "reviewed": "true",
            }
        )
        writer.writerow(
            {
                "source_id": "blank-1",
                "role": "background",
                "ra_deg": float(world[1, 0]),
                "dec_deg": float(world[1, 1]),
                "reviewed": "true",
            }
        )
    base = build_calibration_manifest([str(exposure)])

    result = extract_reviewed_calibration_evidence(
        base,
        catalog_path=catalog,
        output_dir=tmp_path / "evidence",
        psf_radius_px=3,
        background_radius_px=3,
        minimum_psf_snr=2.0,
    )

    assert {row["role"] for row in result["evidence"]} == {"psf", "background"}
    assert result["rejected"] == []
    psf_row = next(row for row in result["evidence"] if row["role"] == "psf")
    assert psf_row["pixel_response_included"] is True
    assert psf_row["pixel_response_provenance"] == "registered detector SCI stamp"
    kernels = read(result["products"]["psf_kernels"]["path"])
    torch.testing.assert_close(kernels.sum(dim=(-2, -1)), torch.ones(1))
    assert result["products"]["detector_evidence"]["size_bytes"] > 0


def test_psf_selection_checks_neighbors_across_extracted_kernel() -> None:
    yy, xx = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    star = 100 * torch.exp(-((xx - 24).square() + (yy - 24).square()) / 2)
    neighbor = 100 * torch.exp(-((xx - 30).square() + (yy - 30).square()) / 2)
    error = torch.ones_like(star)
    dq = torch.zeros_like(star, dtype=torch.int16)
    kwargs = {
        "psf_fwhm_px": 2.354820045,
        "psf_radius_px": 9,
        "max_background_candidates": 0,
    }
    isolated = discover_detector_calibration_positions(star, error, dq, **kwargs)
    assert len(isolated) == 1
    blended = discover_detector_calibration_positions(star + neighbor, error, dq, **kwargs)
    assert blended == []


def test_automatic_calibration_evidence_finds_star_and_blank_sky(
    tmp_path: Path,
) -> None:
    exposure = tmp_path / "jw_crf_rshift_tweakreg.fits"
    _, science = _write_calibration_image(exposure)
    candidates = discover_detector_calibration_positions(
        science,
        torch.ones_like(science),
        torch.zeros_like(science, dtype=torch.int16),
        psf_fwhm_px=2.354820045,
        psf_radius_px=3,
        background_radius_px=3,
        detection_sigma=5.0,
        max_psf_candidates=4,
        max_background_candidates=4,
    )

    psf = [row for row in candidates if row["role"] == "psf"]
    background = [row for row in candidates if row["role"] == "background"]
    assert len(psf) == 1
    assert abs(psf[0]["detector_x"] - 12.0) < 0.1
    assert abs(psf[0]["detector_y"] - 12.0) < 0.1
    assert len(background) == 4

    result = extract_calibration_evidence(
        build_calibration_manifest([str(exposure)]),
        output_dir=tmp_path / "automatic-evidence",
        psf_radius_px=3,
        background_radius_px=3,
        minimum_psf_snr=2.0,
        detection_sigma=5.0,
        max_psf_candidates=4,
        max_background_candidates=4,
    )

    assert result["selection_mode"] == "automatic_detector"
    assert {row["role"] for row in result["evidence"]} == {"psf", "background"}
    assert "catalog_path" not in result


def test_reviewed_catalog_fails_closed_on_unreviewed_rows(tmp_path: Path) -> None:
    catalog = tmp_path / "candidates.csv"
    catalog.write_text(
        "source_id,role,ra_deg,dec_deg,reviewed\nstar-1,psf,1,2,false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reviewed must be true"):
        read_reviewed_sky_catalog(catalog)
