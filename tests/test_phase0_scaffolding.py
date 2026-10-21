from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import tomllib

import pytest

from torchsky_harness import phase0


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "campaigns" / "phase0"
SCHEMA_DIR = ROOT / "src" / "torchsky_harness" / "schemas"
AVAILABLE_SUBJECT = {
    "identity_type": "subject",
    "status": "available",
    "distribution": "torchsky",
    "version": "0.4.0",
    "import_path": "/installed/torchsky/__init__.py",
    "package_content_sha256": "a" * 64,
}


def _data_identity(dataset: dict) -> dict:
    return {
        "identity_type": "data",
        "id": dataset["id"],
        "role": dataset["role"],
        "uri": "file:///fixtures/example.json",
        "sha256": "b" * 64,
    }


def test_all_phase0_campaign_stubs_validate() -> None:
    peers, campaigns = phase0.validate_configs(CONFIG_DIR)

    assert {campaign["id"] for campaign in campaigns} == {
        "phase0-adaptive-reprojection",
        "phase0-astrometry",
        "phase0-astrometry-supplied-centroids",
        "phase0-drizzle",
        "phase0-harmonics",
        "phase0-jwst-ifu",
        "phase0-photometry",
        "phase0-sparse-sphere",
        "phase0-wcs-graphs",
        "astrometry-v1",
        "sphere-v1",
        "deblend-v1",
    }
    assert all(campaign["scheduled_certification"] for campaign in campaigns)
    assert all(campaign["runs_large_peers"] is False for campaign in campaigns)
    assert "tractor" not in {peer.get("distribution") for peer in peers.values()}


def test_requested_peer_matrix_is_declared_required() -> None:
    peers, campaigns = phase0.validate_configs(CONFIG_DIR)
    by_id = {campaign["id"]: campaign for campaign in campaigns}

    assert set(by_id["phase0-wcs-graphs"]["required_peers"]) == {"gwcs", "pyast"}
    assert set(by_id["phase0-astrometry"]["required_peers"]) == {
        "scamp",
        "tweakreg",
        "astrometry-net",
    }
    assert by_id["phase0-astrometry-supplied-centroids"]["required_peers"] == [
        "astrometry-net"
    ]
    assert set(by_id["phase0-sparse-sphere"]["required_peers"]) == {
        "mhealpy",
        "healsparse",
        "mocpy",
    }
    assert set(by_id["phase0-harmonics"]["required_peers"]) == {
        "torch-harmonics",
        "ducc0",
        "s2fft",
        "pixell",
        "namaster",
    }
    assert set(by_id["phase0-drizzle"]["required_peers"]) == {
        "astrodrizzle",
        "stcal",
    }
    assert set(by_id["phase0-adaptive-reprojection"]["required_peers"]) == {
        "reproject-adaptive"
    }
    assert set(by_id["phase0-jwst-ifu"]["required_peers"]) == {
        "jwst-ifu",
        "crds",
    }
    assert set(by_id["phase0-photometry"]["required_peers"]) == {
        "tphot",
        "the-tractor",
        "scarlet",
    }
    assert set().union(
        *(set(campaign["required_peers"]) for campaign in campaigns)
    ) <= set(peers)

    assert set(by_id["astrometry-v1"]["required_peers"]) == {
        "scamp",
        "tweakreg",
        "astrometry-net",
    }
    assert set(by_id["sphere-v1"]["required_peers"]) == {
        "mhealpy",
        "healsparse",
        "torch-harmonics",
        "ducc0",
        "s2fft",
        "pixell",
        "namaster",
    }
    assert set(by_id["deblend-v1"]["required_peers"]) == {
        "tphot",
        "the-tractor",
        "scarlet",
    }


def test_scheduled_coverage_fails_explicitly_for_missing_required_peers() -> None:
    peers = phase0.load_peer_catalog(CONFIG_DIR / "peers.toml")
    campaign = phase0.load_campaign(CONFIG_DIR / "harmonics.toml", peers)
    manifest = {
        "subject": AVAILABLE_SUBJECT,
        "peers": [
            {
                "identity_type": "peer",
                "id": peer_id,
                "kind": peers[peer_id]["kind"],
                "status": "missing",
            }
            for peer_id in campaign["required_peers"]
        ],
        "data": [_data_identity(campaign["datasets"][0])],
        "crds": None,
        "indices": [],
    }

    failures = phase0.coverage_failures(campaign, manifest)

    assert len(failures) == len(campaign["required_peers"])
    assert all("required peer unavailable" in failure for failure in failures)


def test_astrometry_requires_index_identity() -> None:
    peers = phase0.load_peer_catalog(CONFIG_DIR / "peers.toml")
    campaign = phase0.load_campaign(CONFIG_DIR / "astrometry.toml", peers)
    manifest = {
        "subject": AVAILABLE_SUBJECT,
        "peers": [
            {
                "identity_type": "peer",
                "id": peer_id,
                "kind": peers[peer_id]["kind"],
                "status": "available",
                "version": "1.0",
                "path": f"/peers/{peer_id}",
            }
            for peer_id in campaign["required_peers"]
        ],
        "data": [_data_identity(campaign["datasets"][0])],
        "crds": None,
        "indices": [],
    }

    assert phase0.coverage_failures(campaign, manifest) == [
        "index identity unavailable"
    ]


def test_identity_schemas_define_all_shared_identity_types() -> None:
    identities = json.loads(
        (SCHEMA_DIR / "evidence-identities-v1.schema.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (SCHEMA_DIR / "evidence-manifest-v1.schema.json").read_text(encoding="utf-8")
    )

    assert set(identities["$defs"]) >= {
        "subject",
        "peer",
        "data",
        "crds",
        "index",
    }
    assert manifest["properties"]["subject"]["$ref"].endswith("#/$defs/subject")
    assert manifest["properties"]["coverage"]["properties"]["status"]["enum"] == [
        "pass",
        "fail",
    ]


def test_manifest_validator_accepts_explicit_failure() -> None:
    payload = {
        "schema_version": 1,
        "campaign_id": "phase0-example",
        "generated_at": "2026-10-21T00:00:00+00:00",
        "platform": "linux-64",
        "python": "3.13.14",
        "subject": {
            "identity_type": "subject",
            "status": "missing",
            "distribution": "torchsky",
        },
        "peers": [
            {
                "identity_type": "peer",
                "id": "peer",
                "kind": "python",
                "status": "missing",
            }
        ],
        "data": [],
        "crds": None,
        "indices": [],
        "coverage": {"status": "fail", "failures": ["peer unavailable"]},
    }

    phase0.validate_manifest(payload)


def test_python_peer_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> metadata.Distribution:
        raise metadata.PackageNotFoundError("absent")

    monkeypatch.setattr(phase0.metadata, "distribution", missing)
    peer = {
        "id": "example",
        "kind": "python",
        "distribution": "example",
        "import_name": "example",
        "platforms": ["linux-64"],
    }

    record = phase0.inspect_peer(peer, platform_key="linux-64")

    assert record["status"] == "missing"
    assert "absent" in record["reason"]


def test_binary_peer_version_timeout_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phase0.shutil, "which", lambda _: "/usr/bin/example")

    def timeout(*args: object, **kwargs: object) -> object:
        raise phase0.subprocess.TimeoutExpired(args[0], 15)

    monkeypatch.setattr(phase0.subprocess, "run", timeout)
    record = phase0.inspect_peer(
        {
            "id": "example",
            "kind": "binary",
            "command": "example",
            "version_args": ["--version"],
            "platforms": ["osx-arm64"],
        },
        platform_key="osx-arm64",
    )

    assert record["status"] == "missing"
    assert "version probe failed" in record["reason"]


def test_phase0_pixi_environments_and_tasks_are_declared() -> None:
    with (ROOT / "pixi.toml").open("rb") as stream:
        pixi = tomllib.load(stream)

    environments = pixi["environments"]
    tasks = pixi["tasks"]
    assert {
        "phase0-wcs",
        "phase0-astrometry",
        "phase0-astrometry-net",
        "phase0-adaptive",
        "phase0-sparse-sphere",
        "phase0-harmonics",
        "phase0-drizzle",
        "phase0-jwst-ifu",
        "phase0-photometry",
        "phase0-astrometry-portable",
        "phase0-harmonics-portable",
    } <= set(environments)
    assert "phase0-config-check" in tasks
    assert "phase0-provenance-all" in tasks
    assert "phase0-certify-all" in tasks
    assert "phase0-certify-adaptive" in tasks
    assert "phase0-run-drizzle" in tasks
    assert "phase0-run-adaptive" in tasks
    assert "phase0-run-jwst-ifu" in tasks
    assert "certify-astrometry-v1" in tasks
    assert "certify-sphere-v1" in tasks
    assert "certify-deblend-v1" in tasks
    assert "run-certification-v1" in tasks


def test_supplied_centroid_comparator_is_pinned_and_offline() -> None:
    campaign = phase0.load_campaign(
        CONFIG_DIR / "astrometry-supplied-centroids.toml",
        phase0.load_peer_catalog(CONFIG_DIR / "peers.toml"),
    )
    config_path = CONFIG_DIR / campaign["comparator_config"]
    with config_path.open("rb") as stream:
        comparator = tomllib.load(stream)

    assert comparator["version"] == "0.97"
    assert comparator["adapter"] == "torchsky.wcs.solve_with_astrometry_net"
    assert comparator["detection"] is False
    assert comparator["native_index"] is False
    assert comparator["runtime_index_downloads"] is False
    assert comparator["executable_sha256_required"] is True
    assert comparator["index_manifest_sha256_required"] is True
    assert campaign["verification_contract"].startswith("reserved supplied centroids")
