from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from torchsky_harness.cli import main
from torchsky_harness.release_performance import (
    assess_claims,
    build_artifact,
    load_claims,
    validate_claims,
    validate_configuration,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "src/torchsky_harness/schemas/release-performance-claims-v1.schema.json"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(claim: dict, role: str) -> dict:
    return {
        "runner": "measured-node-1",
        "cpu": "measured-test-cpu",
        "os": "measured-test-os",
        "python": "3.13.14",
        "torch": "2.13.0",
        "torchsky_version": "0.3.0" if role == "baseline" else "0.4.0",
        "torchsky_revision": "a" * 40 if role == "baseline" else "b" * 40,
        "dtype": claim["dtype"],
        "device": claim["device"],
        "task_command_sha256": claim["task_command_sha256"],
        "task_arguments": claim["effective_arguments"],
    }


def _artifact(
    root: Path,
    claim: dict,
    values: list[float],
    *,
    role: str,
    label: str,
    fingerprint: dict | None = None,
) -> dict:
    samples = []
    sources = []
    for index, value in enumerate(values):
        pair_id = f"pair-{index:03d}"
        raw_relative = Path("reports") / label / role / f"{pair_id}.json"
        raw = root / raw_relative
        _write(raw, [{**claim["row_selector"], claim["timing_field"]: value}])
        digest = _sha256(raw)
        envelope_relative = Path("envelopes") / label / role / f"{pair_id}.json"
        envelope = root / envelope_relative
        _write(
            envelope,
            {
                "pair_id": pair_id,
                "status": "OK",
                "correctness_status": "OK",
                "source_report": {
                    "path": raw_relative.as_posix(),
                    "sha256": digest,
                    "task": claim["task"],
                    "task_artifact": claim["task_artifact"],
                },
                "rows": [
                    {
                        **claim["row_selector"],
                        "status": "OK",
                        claim["timing_field"]: value,
                    }
                ],
            },
        )
        samples.append(
            {
                "pair_id": pair_id,
                "value_ms": value,
                "status": "OK",
                "source_report_sha256": digest,
            }
        )
        sources.append(
            {
                "pair_id": pair_id,
                "path": raw_relative.as_posix(),
                "sha256": digest,
                "status": "OK",
                "correctness_status": "OK",
                "task": claim["task"],
                "task_artifact": claim["task_artifact"],
                "envelope_path": envelope_relative.as_posix(),
                "envelope_sha256": _sha256(envelope),
            }
        )
    return {
        "schema_version": 1,
        "task": claim["task"],
        "evaluation": claim["evaluation"],
        "correctness_source": claim["correctness_source"],
        "fingerprint": dict(
            _fingerprint(claim, role) if fingerprint is None else fingerprint
        ),
        "source_reports": sources,
        "rows": [
            {
                **claim["row_selector"],
                "status": "OK",
                claim["timing_field"]: samples,
            }
        ],
    }


def _artifact_path(root: Path, claim: dict, field: str) -> Path:
    return root / claim[field]


def _configured_artifacts(tmp_path: Path) -> dict:
    payload = deepcopy(load_claims())
    for index, claim in enumerate(payload["claims"]):
        claim["baseline_artifact"] = f"artifacts/baseline-{index}.json"
        claim["candidate_artifact"] = f"artifacts/candidate-{index}.json"
        _write(
            _artifact_path(tmp_path, claim, "baseline_artifact"),
            _artifact(tmp_path, claim, [10.0] * 5, role="baseline", label=str(index)),
        )
        _write(
            _artifact_path(tmp_path, claim, "candidate_artifact"),
            _artifact(tmp_path, claim, [10.2] * 5, role="candidate", label=str(index)),
        )
    return payload


def test_release_performance_claim_schema_has_fail_closed_contract_fields() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    required = set(schema["$defs"]["claim"]["required"])
    assert {
        "row_id",
        "correctness_source",
        "baseline_artifact",
        "candidate_artifact",
        "task_artifact",
        "task_command_sha256",
        "effective_arguments",
        "dtype",
        "device",
        "paired_sampling_policy",
        "max_regression_fraction",
        "absolute_timing_floor_ms",
    } <= required
    properties = schema["$defs"]["claim"]["properties"]
    assert properties["max_regression_fraction"] == {"const": 0.05}
    assert (
        properties["paired_sampling_policy"]["properties"]["minimum_pairs"]["minimum"]
        == 5
    )


def test_release_performance_claims_are_finite_and_bound_to_exact_tasks() -> None:
    payload = load_claims()
    claims = payload["claims"]
    assert len(claims) == 6
    assert len({claim["row_id"] for claim in claims}) == len(claims)
    assert {claim["dtype"] for claim in claims[:4]} == {"float64"}
    assert {claim["dtype"] for claim in claims[4:]} == {"float32"}
    assert validate_configuration(payload, root=ROOT) == []
    errors = validate_claims(payload, root=ROOT)
    assert len(errors) == 12
    assert all("artifact absent" in error for error in errors)


def test_release_performance_rejects_task_argument_or_command_drift() -> None:
    payload = deepcopy(load_claims())
    payload["claims"][0]["effective_arguments"][0] = "--changed"
    payload["claims"][1]["task_command_sha256"] = "0" * 64
    errors = validate_configuration(payload, root=ROOT)
    assert any("effective_arguments do not match" in error for error in errors)
    assert any("task command hash does not match" in error for error in errors)


def test_release_performance_claims_pass_with_complete_paired_artifacts(
    tmp_path: Path,
) -> None:
    result = assess_claims(
        _configured_artifacts(tmp_path), root=ROOT, artifact_root=tmp_path
    )
    assert result["release_performance_ready"] is True
    assert result["errors"] == []
    assert len(result["claims"]) == 6
    assert result["claims"][0]["median_paired_ratio"] == pytest.approx(1.02)


def test_release_performance_tree_remains_valid_after_relocation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scratch-output"
    source.mkdir()
    payload = _configured_artifacts(source)
    _write(source / "release_performance_claims.json", payload)
    destination = tmp_path / "durable-copy"
    shutil.copytree(source, destination)
    relocated = json.loads(
        (destination / "release_performance_claims.json").read_text(encoding="utf-8")
    )
    result = assess_claims(relocated, root=ROOT, artifact_root=destination)
    assert result["release_performance_ready"] is True
    assert not any(
        str(source) in path.read_text(encoding="utf-8")
        for path in destination.rglob("*.json")
    )


def test_release_performance_claims_fail_closed_without_candidate(
    tmp_path: Path,
) -> None:
    payload = _configured_artifacts(tmp_path)
    claim = payload["claims"][0]
    _artifact_path(tmp_path, claim, "candidate_artifact").unlink()
    result = assess_claims(payload, root=ROOT, artifact_root=tmp_path)
    assert result["release_performance_ready"] is False
    assert any("candidate artifact absent" in error for error in result["errors"])


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "tampered",
        "nonhex",
        "absolute",
        "wrong-identity",
        "tampered-envelope",
    ],
)
def test_release_performance_authenticates_raw_reports(
    tmp_path: Path, mode: str
) -> None:
    payload = _configured_artifacts(tmp_path)
    claim = payload["claims"][0]
    artifact_path = _artifact_path(tmp_path, claim, "candidate_artifact")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    source = artifact["source_reports"][0]
    raw = tmp_path / source["path"]
    if mode == "missing":
        raw.unlink()
    elif mode == "tampered":
        raw.write_text("[]", encoding="utf-8")
    elif mode == "nonhex":
        source["sha256"] = "z" * 64
        artifact["rows"][0][claim["timing_field"]][0]["source_report_sha256"] = "z" * 64
        _write(artifact_path, artifact)
    elif mode == "absolute":
        source["path"] = str(raw)
        envelope_path = tmp_path / source["envelope_path"]
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        envelope["source_report"]["path"] = str(raw)
        _write(envelope_path, envelope)
        source["envelope_sha256"] = _sha256(envelope_path)
        _write(artifact_path, artifact)
    elif mode == "wrong-identity":
        source["task"] = "other-task"
        _write(artifact_path, artifact)
    else:
        (tmp_path / source["envelope_path"]).write_text("{}", encoding="utf-8")
    errors = validate_claims(payload, root=ROOT, artifact_root=tmp_path)
    expected = {
        "missing": "is absent",
        "tampered": "hash mismatch",
        "nonhex": "has no sha256",
        "absolute": "must stay within the artifact root",
        "wrong-identity": "identity mismatch",
        "tampered-envelope": "envelope hash mismatch",
    }[mode]
    assert any(expected in error for error in errors)


def test_release_performance_reselects_raw_row_and_timing(tmp_path: Path) -> None:
    payload = _configured_artifacts(tmp_path)
    claim = payload["claims"][0]
    artifact_path = _artifact_path(tmp_path, claim, "candidate_artifact")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    raw = tmp_path / artifact["source_reports"][0]["path"]
    _write(raw, [{**claim["row_selector"], claim["timing_field"]: 99.0}])
    digest = _sha256(raw)
    artifact["source_reports"][0]["sha256"] = digest
    artifact["rows"][0][claim["timing_field"]][0]["source_report_sha256"] = digest
    envelope_path = tmp_path / artifact["source_reports"][0]["envelope_path"]
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["source_report"]["sha256"] = digest
    _write(envelope_path, envelope)
    artifact["source_reports"][0]["envelope_sha256"] = _sha256(envelope_path)
    _write(artifact_path, artifact)
    errors = validate_claims(payload, root=ROOT, artifact_root=tmp_path)
    assert any("timing does not match" in error for error in errors)


def test_release_performance_claims_require_matching_measured_fingerprints(
    tmp_path: Path,
) -> None:
    payload = _configured_artifacts(tmp_path)
    claim = payload["claims"][0]
    path = _artifact_path(tmp_path, claim, "candidate_artifact")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["fingerprint"]["cpu"] = "other-cpu"
    _write(path, artifact)
    errors = validate_claims(payload, root=ROOT, artifact_root=tmp_path)
    assert any("runner fingerprint mismatch for cpu" in error for error in errors)


def test_release_performance_allows_distinct_subject_revisions(tmp_path: Path) -> None:
    payload = _configured_artifacts(tmp_path)
    first = payload["claims"][0]
    baseline = json.loads(
        _artifact_path(tmp_path, first, "baseline_artifact").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        _artifact_path(tmp_path, first, "candidate_artifact").read_text(
            encoding="utf-8"
        )
    )
    assert (
        baseline["fingerprint"]["torchsky_revision"]
        != candidate["fingerprint"]["torchsky_revision"]
    )
    assert assess_claims(payload, root=ROOT, artifact_root=tmp_path)[
        "release_performance_ready"
    ]


def test_release_performance_claims_reject_regression_above_floor(
    tmp_path: Path,
) -> None:
    payload = _configured_artifacts(tmp_path)
    claim = payload["claims"][0]
    _write(
        _artifact_path(tmp_path, claim, "candidate_artifact"),
        _artifact(tmp_path, claim, [10.6] * 5, role="candidate", label="0-regressed"),
    )
    result = assess_claims(payload, root=ROOT, artifact_root=tmp_path)
    assert result["release_performance_ready"] is False
    assert any("exceeds 1.05" in error for error in result["errors"])


def _extracted_reports(tmp_path: Path, claim: dict) -> list[Path]:
    reports = []
    for index in range(5):
        pair_id = f"pair-{index:03d}"
        raw_relative = Path("raw") / f"{pair_id}.json"
        raw = tmp_path / raw_relative
        _write(raw, [{**claim["row_selector"], claim["timing_field"]: 10.0 + index}])
        report = tmp_path / "extracted" / f"{pair_id}.json"
        _write(
            report,
            {
                "pair_id": pair_id,
                "status": "OK",
                "correctness_status": "OK",
                "source_report": {
                    "path": raw_relative.as_posix(),
                    "sha256": _sha256(raw),
                    "task": claim["task"],
                    "task_artifact": claim["task_artifact"],
                },
                "rows": [
                    {
                        **claim["row_selector"],
                        "status": "OK",
                        claim["timing_field"]: 10.0 + index,
                    }
                ],
            },
        )
        reports.append(report)
    return reports


def test_performance_artifact_builder_requires_and_collects_real_reports(
    tmp_path: Path,
) -> None:
    payload = load_claims()
    claim = payload["claims"][0]
    reports = _extracted_reports(tmp_path, claim)
    artifact = build_artifact(
        payload,
        claim_id=claim["row_id"],
        reports=reports,
        fingerprint=_fingerprint(claim, "baseline"),
        artifact_root=tmp_path,
    )
    assert len(artifact["rows"][0][claim["timing_field"]]) == 5
    assert all(
        not Path(source["path"]).is_absolute() for source in artifact["source_reports"]
    )
    with pytest.raises(ValueError, match="requires at least 5 reports"):
        build_artifact(
            payload,
            claim_id=claim["row_id"],
            reports=reports[:4],
            fingerprint=_fingerprint(claim, "baseline"),
            artifact_root=tmp_path,
        )


def test_performance_cli_is_a_blocking_readiness_entry_point(tmp_path: Path) -> None:
    payload = _configured_artifacts(tmp_path)
    config = tmp_path / "claims.json"
    output = tmp_path / "readiness.json"
    _write(config, payload)
    args = [
        "performance",
        "--config",
        str(config),
        "--root",
        str(ROOT),
        "--artifact-root",
        str(tmp_path),
        "--json-out",
        str(output),
    ]
    assert main(args) == 0
    claim = payload["claims"][0]
    _artifact_path(tmp_path, claim, "candidate_artifact").unlink()
    assert main(args[:-2]) == 1


def test_mandatory_release_readiness_task_includes_performance() -> None:
    import tomllib

    tasks = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))["tasks"]
    assert "performance-readiness" in tasks
    assert tasks["release-readiness"] == (
        "pixi run evidence-bundle && pixi run performance-readiness"
    )
