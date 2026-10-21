from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from torchsky_harness import phase0
from torchsky_harness.bounded_campaigns import (
    execute_case,
    load_matrix,
    validate_result,
)
from torchsky_harness.comparator_adapters import (
    UnavailableEvidence,
    adapt_stcal_resample_output,
    load_external_association_manifest,
    resolve_external_association,
)
from torchsky_harness.comparator_metrics import (
    adaptive_metrics,
    ifu_cube_metrics,
    product_metrics,
    required_metric_failures,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "campaigns" / "phase0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bounded_campaign_matrices_are_frozen_bounded_and_metric_complete() -> None:
    peers, campaigns = phase0.validate_configs(CONFIG_DIR)
    bounded = [campaign for campaign in campaigns if campaign["status"] == "bounded"]

    assert {campaign["id"] for campaign in bounded} == {
        "phase0-adaptive-reprojection",
        "phase0-drizzle",
        "phase0-jwst-ifu",
        "astrometry-v1",
        "sphere-v1",
        "deblend-v1",
    }
    for campaign in bounded:
        matrix_path = CONFIG_DIR / campaign["matrix"]
        matrix = load_matrix(matrix_path, campaign_id=campaign["id"])
        observed = {
            metric for case in matrix["cases"] for metric in case["required_metrics"]
        }
        assert len(matrix["cases"]) <= campaign["max_cases"]
        assert set(campaign["required_metrics"]) <= observed
        assert {
            peer_id for case in matrix["cases"] for peer_id in case["required_peers"]
        } <= set(peers)


def test_data_manifests_pin_matrix_and_fixture_hashes() -> None:
    _, campaigns = phase0.validate_configs(CONFIG_DIR)

    for campaign in campaigns:
        if campaign["status"] != "bounded":
            continue
        records = []
        for relative in campaign["data_manifests"]:
            records.extend(
                json.loads((CONFIG_DIR / relative).read_text(encoding="utf-8"))
            )
        by_id = {record["id"]: record for record in records}
        assert {dataset["id"] for dataset in campaign["datasets"]} == set(by_id)
        for record in records:
            uri = record["uri"]
            assert uri.startswith("file:")
            path = ROOT / uri.removeprefix("file:")
            assert path.is_file()
            assert _sha256(path) == record["sha256"]


def test_external_ifu_manifests_fail_closed_without_reviewed_assets() -> None:
    for name in ("nirspec-association-v1.json", "miri-association-v1.json"):
        manifest = load_external_association_manifest(CONFIG_DIR / "external" / name)
        assert manifest["availability"] == "unavailable"
        with pytest.raises(UnavailableEvidence, match="No redistributable"):
            resolve_external_association(manifest)


def test_vendored_hst_fixture_is_compact_real_wfc3_data() -> None:
    import torchfits

    path = ROOT / "benchmarks" / "fixtures" / "real_wcs" / "sample_ipppssoot_flt.fits"
    header = torchfits.read_header(str(path), 0)
    science = torchfits.read(str(path), hdu=1, mmap=False)

    assert header["TELESCOP"] == "HST"
    assert header["INSTRUME"] == "WFC3"
    assert header["DETECTOR"] == "IR"
    assert science.shape == (10, 10)
    assert path.stat().st_size < 256 * 1024


def test_stcal_product_adapter_and_product_metrics_cover_all_planes() -> None:
    payload = {
        "data": np.array([[2.0, 3.0]]),
        "wht": np.array([[4.0, 5.0]]),
        "var_poisson": np.array([[0.2, 0.3]]),
        "var_rnoise": np.array([[0.1, 0.1]]),
        "var_flat": np.array([[0.01, 0.02]]),
        "dq": np.array([[1, 2]], dtype=np.uint32),
        "con": np.array([[[1, 3]]], dtype=np.int32),
        "nexp": np.array([[1, 2]], dtype=np.int32),
    }
    adapted = adapt_stcal_resample_output(payload)
    metrics = product_metrics(adapted, adapted)

    assert metrics["science_rmse"] == 0.0
    assert metrics["weight_rmse"] == 0.0
    assert metrics["variance_var_poisson_rmse"] == 0.0
    assert metrics["variance_var_rnoise_rmse"] == 0.0
    assert metrics["variance_var_flat_rmse"] == 0.0
    assert metrics["dq_mismatch_count"] == 0
    assert metrics["context_mismatch_count"] == 0
    assert metrics["nexp_mismatch_count"] == 0


def test_adaptive_metrics_include_error_and_adjoint_contract() -> None:
    truth = np.arange(16, dtype=np.float64).reshape(4, 4)
    actual = truth + 1.0e-4
    metrics = adaptive_metrics(
        actual,
        truth,
        truth=truth,
        adjoint_relative_error=2.0e-15,
    )

    assert metrics["adaptive_peer_max_abs_error"] == pytest.approx(1.0e-4)
    assert metrics["adaptive_truth_rmse"] == pytest.approx(1.0e-4)
    assert metrics["adaptive_adjoint_relative_error"] == 2.0e-15
    assert (
        required_metric_failures(
            metrics,
            ("adaptive_peer_rmse", "adaptive_adjoint_relative_error"),
        )
        == []
    )


def test_ifu_metrics_cover_support_line_shape_continuum_dq_and_covariance() -> None:
    wavelength = np.arange(9, dtype=np.float64) + 4996.0
    line = 4.0 * np.exp(-0.5 * ((wavelength - 5000.0) / 1.2) ** 2)
    reference = np.zeros((9, 3, 3), dtype=np.float64)
    reference[:, 1, 1] = 2.0 + line
    dq = np.zeros_like(reference, dtype=np.uint32)
    dq[4, 1, 1] = 4

    metrics = ifu_cube_metrics(
        reference,
        reference,
        spectral_axis=wavelength,
        continuum=2.0,
        spatial_position=(1, 1),
        dq=dq,
        reference_dq=dq,
        covariance_variance=3.5,
        reference_covariance_variance=3.5,
    )

    assert metrics["ifu_voxel_support_jaccard"] == 1.0
    assert metrics["ifu_line_flux_relative_error"] == 0.0
    assert metrics["ifu_line_centroid_error"] == 0.0
    assert metrics["ifu_line_fwhm_error"] == 0.0
    assert metrics["ifu_continuum_absolute_error"] < 0.1
    assert metrics["ifu_dq_mismatch_count"] == 0
    assert metrics["ifu_covariance_relative_error"] == 0.0


def test_required_peer_unavailability_is_case_evidence_not_a_skip() -> None:
    case = {
        "id": "required-peer",
        "kind": "adaptive_synthetic",
        "required_peers": ["peer"],
        "required_metrics": ["metric"],
        "parameters": {},
    }
    result = execute_case(
        case,
        {
            "peers": [
                {
                    "identity_type": "peer",
                    "id": "peer",
                    "kind": "python",
                    "status": "missing",
                }
            ]
        },
    )

    assert result["status"] == "unavailable"
    assert result["failures"] == ["required peer unavailable: peer"]


def test_bounded_result_validator_accepts_explicit_failed_certification() -> None:
    subject = {
        "identity_type": "subject",
        "status": "missing",
        "distribution": "torchsky",
    }
    payload = {
        "schema_version": 1,
        "campaign_id": "phase0-example",
        "generated_at": "2026-10-21T00:00:00+00:00",
        "matrix": {
            "id": "matrix",
            "path": "/matrix.json",
            "sha256": "a" * 64,
            "case_count": 1,
        },
        "provenance": {
            "schema_version": 1,
            "campaign_id": "phase0-example",
            "generated_at": "2026-10-21T00:00:00+00:00",
            "platform": "osx-arm64",
            "python": "3.13.14",
            "subject": subject,
            "peers": [],
            "data": [],
            "crds": None,
            "indices": [],
            "coverage": {"status": "fail", "failures": ["subject unavailable"]},
        },
        "cases": [
            {
                "id": "case",
                "kind": "external",
                "status": "unavailable",
                "metrics": {},
                "failures": ["asset unavailable"],
            }
        ],
        "certification": {"status": "fail", "failures": ["asset unavailable"]},
    }

    validate_result(payload)
