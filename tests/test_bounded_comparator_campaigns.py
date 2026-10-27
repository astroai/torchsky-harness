from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from torchsky_harness import phase0
from torchsky_harness import bounded_campaigns
from torchsky_harness.bounded_campaigns import (
    _real_hst_drizzle_metrics,
    execute_case,
    load_matrix,
    validate_result,
)
from torchsky_harness.comparator_adapters import (
    ComparatorProducts,
    UnavailableEvidence,
    adapt_astrodrizzle_product,
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


def test_astrodrizzle_adapter_retains_serializable_output_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torchfits

    product_path = tmp_path / "bounded_drz.fits"
    product_path.touch()
    science = np.arange(6, dtype=np.float32).reshape(2, 3)

    def fake_plane(path: Path, names: tuple[str, ...]) -> np.ndarray | None:
        assert path == product_path
        if "SCI" in names:
            return science
        if "WHT" in names:
            return np.ones_like(science)
        return None

    output_header = {
        "NAXIS1": np.int64(3),
        "NAXIS2": np.int64(2),
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": np.float64(2.0),
        "CRPIX2": np.float64(1.5),
        "BUNIT": "ELECTRONS/S",
    }
    monkeypatch.setattr(
        "torchsky_harness.comparator_adapters._read_optional_plane", fake_plane
    )
    monkeypatch.setattr(torchfits, "read_header", lambda path, hdu: output_header)

    product = adapt_astrodrizzle_product(product_path)

    assert product.metadata["target_shape"] == [2, 3]
    assert product.metadata["target_wcs"]["CRPIX1"] == 2.0
    assert product.metadata["output_header"]["BUNIT"] == "ELECTRONS/S"
    json.dumps(product.metadata, allow_nan=False)


def test_real_hst_metrics_use_only_shared_finite_support() -> None:
    actual = ComparatorProducts(
        science=np.array([[1.0, 2.0], [100.0, np.nan]]),
        weight=np.array([[1.0, 1.0], [0.0, 0.0]]),
    )
    reference = ComparatorProducts(
        science=np.array([[1.0, 3.0], [-100.0, 4.0]]),
        weight=np.array([[1.0, 1.0], [0.0, 1.0]]),
    )

    metrics = _real_hst_drizzle_metrics(actual, reference)

    assert metrics["real_hst_shared_support_rmse"] == pytest.approx(2**-0.5)
    assert metrics["real_hst_support_iou"] == pytest.approx(2 / 3)
    assert metrics["real_hst_shared_support_fraction"] == 0.5
    assert metrics["flux_relative_error"] == pytest.approx(0.25)
    assert metrics["centroid_error_px"] == pytest.approx(1 / 12)


def test_real_hst_runner_matches_controls_and_labels_unmatched_planes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    reference = ComparatorProducts(
        science=np.array([[1.0, 2.0], [3.0, 4.0]]),
        weight=np.ones((2, 2)),
        metadata={
            "output_header": {"BUNIT": "ELECTRONS/S"},
            "target_wcs": {"BUNIT": "ELECTRONS/S"},
        },
    )

    def fake_astrodrizzle(inputs, output_directory, *, output_name, options):
        captured["options"] = options
        return reference

    monkeypatch.setattr(bounded_campaigns, "run_astrodrizzle_adapter", fake_astrodrizzle)
    monkeypatch.setattr(
        bounded_campaigns,
        "_run_torchsky_hst_product",
        lambda fixture, product, pixfrac: ComparatorProducts(
            science=product.science.copy(), weight=product.weight.copy()
        ),
    )
    case = {
        "parameters": {
            "fixture": "fixture.fits",
            "output_name": "bounded",
            "pixfrac": 0.8,
        }
    }

    result = bounded_campaigns._run_astrodrizzle_real(case)

    options = captured["options"]
    assert isinstance(options, dict)
    assert options["final_kernel"] == "square"
    assert options["final_pixfrac"] == 0.8
    assert options["final_units"] == "cps"
    assert options["final_wht_type"] == "EXP"
    assert options["skysub"] is False
    assert options["driz_cr"] is False
    assert result["metrics"]["real_hst_shared_support_rmse"] == 0.0
    assert "weight_plane_parity" in result["metadata"]["unavailable_metrics"]
    assert "dq_rejection_context_parity" in result["metadata"]["unavailable_metrics"]


def test_real_hst_installed_subject_computes_bounded_diagnostic_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchfits

    def fake_astrodrizzle(inputs, output_directory, *, output_name, options):
        del output_directory, output_name, options
        fixture = Path(inputs[0])
        science, header = torchfits.read(
            str(fixture), hdu="SCI", mmap=False, return_header=True
        )
        dq = torchfits.read(str(fixture), hdu="DQ", mmap=False)
        science_array = science.detach().cpu().numpy()
        return ComparatorProducts(
            science=science_array,
            weight=(dq.detach().cpu().numpy() == 0).astype(np.float64),
            metadata={
                "output_header": dict(header),
                "target_wcs": dict(header),
            },
        )

    monkeypatch.setattr(
        bounded_campaigns, "run_astrodrizzle_adapter", fake_astrodrizzle
    )
    result = bounded_campaigns._run_astrodrizzle_real(
        {
            "parameters": {
                "fixture": (
                    "benchmarks/fixtures/real_wcs/sample_ipppssoot_flt.fits"
                ),
                "output_name": "bounded",
                "pixfrac": 1.0,
            }
        }
    )

    metrics = result["metrics"]
    assert np.isfinite(metrics["real_hst_shared_support_rmse"])
    assert np.isfinite(metrics["real_hst_shared_support_relative_error"])
    assert 0.0 < metrics["real_hst_shared_support_fraction"] <= 1.0
    assert 0.0 <= metrics["real_hst_support_iou"] <= 1.0
    assert np.isfinite(metrics["flux_relative_error"])
    assert np.isfinite(metrics["centroid_error_px"])


def test_real_hst_torchsky_subject_unavailability_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def unavailable_torchsky(name, *args, **kwargs):
        if name == "torchsky.map":
            raise ModuleNotFoundError("No module named 'torchsky'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_torchsky)

    with pytest.raises(UnavailableEvidence, match="installed Torchsky subject unavailable"):
        bounded_campaigns._run_torchsky_hst_product(
            Path("fixture.fits"),
            ComparatorProducts(
                science=np.ones((1, 1)),
                metadata={"target_wcs": {"BUNIT": "ELECTRONS/S"}},
            ),
            pixfrac=1.0,
        )


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
