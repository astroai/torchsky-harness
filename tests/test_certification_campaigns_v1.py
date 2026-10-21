from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from torchsky_harness import phase0
from torchsky_harness.bounded_campaigns import execute_case, load_matrix
from torchsky_harness.comparator_adapters import (
    UnavailableEvidence,
    load_external_asset_manifest,
    resolve_external_asset,
    run_supplied_centroid_adapter,
    run_supplied_source_adapter,
)
from torchsky_harness.comparator_metrics import (
    astrometry_certification_metrics,
    deblend_certification_metrics,
    pseudocl_certification_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "campaigns" / "phase0"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_configs_are_bounded_and_preserve_existing_map_ifu_campaigns() -> None:
    _, campaigns = phase0.validate_configs(CONFIG)
    by_id = {campaign["id"]: campaign for campaign in campaigns}

    assert {by_id[name]["status"] for name in ("astrometry-v1", "sphere-v1", "deblend-v1")} == {"bounded"}
    assert {by_id[name]["status"] for name in ("phase0-drizzle", "phase0-adaptive-reprojection", "phase0-jwst-ifu")} == {"bounded"}
    assert by_id["astrometry-v1"]["required_identities"] == ["subject", "peer", "data", "index"]
    assert by_id["deblend-v1"]["max_cases"] == 6
    assert by_id["sphere-v1"]["max_cases"] == 10


def test_v1_data_manifests_pin_every_declared_input() -> None:
    peers = phase0.load_peer_catalog(CONFIG / "peers.toml")
    for name in ("astrometry-v1", "sphere-v1", "deblend-v1"):
        campaign = phase0.load_campaign(CONFIG / f"{name}.toml", peers)
        records = json.loads((CONFIG / campaign["data_manifests"][0]).read_text(encoding="utf-8"))
        assert {record["id"] for record in records} == {dataset["id"] for dataset in campaign["datasets"]}
        for record in records:
            path = ROOT / record["uri"].removeprefix("file:")
            assert path.is_file()
            assert _hash(path) == record["sha256"]


def test_external_asset_schema_and_stubs_fail_closed() -> None:
    schema = json.loads((ROOT / "src" / "torchsky_harness" / "schemas" / "external-asset-manifest-v1.schema.json").read_text(encoding="utf-8"))
    assert set(schema["properties"]["family"]["enum"]) == {"astrometry-v1", "sphere-v1", "deblend-v1"}
    stubs = sorted((CONFIG / "external").glob("astrometry-*-v1.json"))
    stubs += sorted((CONFIG / "external").glob("sphere-*-v1.json"))
    stubs += sorted((CONFIG / "external").glob("deblend-*-v1.json"))
    assert len(stubs) == 17
    for path in stubs:
        manifest = load_external_asset_manifest(path)
        assert manifest["availability"] == "unavailable"
        with pytest.raises(UnavailableEvidence, match=manifest["reason"]):
            resolve_external_asset(manifest)


def test_supplied_centroid_adapter_requires_no_detection_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    product = tmp_path / "centroids.json"
    product.write_text(json.dumps({"observed_tangent_arcsec": [[0.0, 0.0]], "predicted_tangent_arcsec": [[0.0, 0.0]], "row_id": [7]}), encoding="utf-8")
    monkeypatch.setenv("CENTROID_PRODUCT", str(product))
    manifest = {
        "family": "astrometry-v1",
        "availability": "available",
        "asset_env": "CENTROID_PRODUCT",
        "asset_sha256": _hash(product),
        "required_identity_fields": ["peer", "version", "centroid_source", "fixture_sha256", "split_sha256"],
        "identity": {"peer": "scamp", "version": "2.15", "centroid_source": "supplied", "detection": False, "fixture_sha256": "a" * 64, "split_sha256": "b" * 64},
    }
    adapted = run_supplied_centroid_adapter("scamp", manifest)
    assert adapted["row_id"] == [7]
    manifest["identity"]["detection"] = True
    with pytest.raises(UnavailableEvidence, match="disable detection"):
        run_supplied_centroid_adapter("scamp", manifest)


def test_supplied_source_adapter_requires_fixed_catalog_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    product = tmp_path / "sources.npz"
    np.savez(product, component_id=np.array([1]), flux=np.ones((1, 2)), lon_deg=np.array([1.0]), lat_deg=np.array([2.0]), covariance=np.eye(2))
    monkeypatch.setenv("SOURCE_PRODUCT", str(product))
    manifest = {
        "family": "deblend-v1",
        "availability": "available",
        "asset_env": "SOURCE_PRODUCT",
        "asset_sha256": _hash(product),
        "required_identity_fields": ["peer", "version", "source_selection", "fixture_sha256", "split_sha256"],
        "identity": {"peer": "scarlet", "version": "0.5.1", "source_selection": "supplied", "detection": False, "fixture_sha256": "a" * 64, "split_sha256": "b" * 64},
    }
    adapted = run_supplied_source_adapter("scarlet", manifest)
    np.testing.assert_array_equal(adapted["component_id"], [1])
    manifest["identity"]["source_selection"] = "detected"
    with pytest.raises(UnavailableEvidence, match="source_selection"):
        run_supplied_source_adapter("scarlet", manifest)


def test_campaign_metrics_cover_holdouts_maps_covariance_and_no_detection() -> None:
    residual = np.zeros((12, 2))
    mask = np.arange(12) % 3 == 0
    astrometry = astrometry_certification_metrics(
        residual,
        exposure_holdout=mask,
        source_holdout=np.roll(mask, 1),
        epoch_holdout=np.roll(mask, 2),
        focal_xy=np.stack((np.linspace(-1, 1, 12), np.linspace(1, -1, 12)), axis=1),
        color=np.linspace(-0.5, 1.5, 12),
        epoch=np.linspace(2019, 2022, 12),
        covariance_arcsec2=np.broadcast_to(np.eye(2) * 0.01, (12, 2, 2)),
        gauge_residual_norm=0.0,
        gauge_count=2,
        runtime_seconds=0.25,
        peak_bytes=4096,
    )
    assert astrometry["astrometry_covariance_95_coverage"] == 1.0
    assert astrometry["astrometry_gauge_count"] == 2
    assert astrometry["astrometry_time_residual_map_max_bias_arcsec"] == 0.0

    pseudocl = pseudocl_certification_metrics(np.ones(4), np.ones(4), standard_b_power=2.0, pure_b_power=0.02, monte_carlo_coverage=0.75)
    assert pseudocl["pseudocl_pure_b_leakage_ratio"] == pytest.approx(0.01)
    assert pseudocl["pseudocl_monte_carlo_coverage"] == 0.75

    deblend = deblend_certification_metrics(
        np.array([[2.0, 1.0], [1.0, 2.0]]),
        np.array([[2.0, 1.0], [1.0, 2.0]]),
        position_error_arcsec=np.zeros(2),
        covariance=np.eye(4),
        reference_covariance=np.eye(4),
        catastrophic_swaps=np.zeros(2, dtype=bool),
        heldout_log_likelihood=-3.0,
        source_detection=False,
    )
    assert deblend["deblend_flux_max_relative_error"] == 0.0
    assert deblend["deblend_catastrophic_swap_count"] == 0
    assert deblend["deblend_source_detection"] is False


@pytest.mark.parametrize(
    ("campaign", "case_id"),
    [
        ("astrometry-v1", "synthetic-global-local-color-dcr-pm-parallax-robust-learned"),
        ("sphere-v1", "existing-healpix-index-bruteforce"),
        ("sphere-v1", "car-scalar-spin-analytic"),
        ("sphere-v1", "pseudocl-pure-eb-low-l-monte-carlo"),
        ("deblend-v1", "deterministic-crowded-fixed-template-no-detection"),
    ],
)
def test_internal_bounded_cases_execute_against_installed_subject(campaign: str, case_id: str) -> None:
    matrix = load_matrix(CONFIG / "matrices" / f"{campaign}.json", campaign_id=campaign)
    case = next(value for value in matrix["cases"] if value["id"] == case_id)
    result = execute_case(case, {"peers": []})
    assert result["status"] == "passed", result["failures"]
