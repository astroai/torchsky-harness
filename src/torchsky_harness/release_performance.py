from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
from statistics import median
import tomllib
from typing import Any, Mapping

from torchsky_harness.evidence import CLAIMS


CONFIG_PATH = Path(__file__).with_name("release_performance_claims.json")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArtifactMeasurement:
    fingerprint: dict[str, Any]
    samples: dict[str, float]
    pair_ids: list[str]


def load_claims(path: Path = CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release performance claim configuration must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay within the artifact root")
    return path


def _artifact_path(root: Path, relative: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} resolves outside the artifact root")
    return resolved


def _task_command(tasks: Mapping[str, Any], task: str) -> str:
    definition = tasks.get(task)
    return (
        definition.get("cmd", "")
        if isinstance(definition, dict)
        else str(definition or "")
    )


def _effective_arguments(command: str) -> list[str]:
    tokens = shlex.split(command)
    if "--" not in tokens:
        return []
    return tokens[tokens.index("--") + 1 :]


def build_artifact(
    payload: Mapping[str, Any],
    *,
    claim_id: str,
    reports: list[Path],
    fingerprint: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    claims = payload.get("claims")
    matches = (
        [
            claim
            for claim in claims
            if isinstance(claim, dict) and claim.get("row_id") == claim_id
        ]
        if isinstance(claims, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError(f"claim {claim_id!r} must exist exactly once")
    claim = matches[0]
    minimum_pairs = claim["paired_sampling_policy"]["minimum_pairs"]
    if len(reports) < minimum_pairs:
        raise ValueError(
            f"claim {claim_id!r} requires at least {minimum_pairs} reports"
        )
    timing_samples = []
    source_reports = []
    pair_ids: set[str] = set()
    for index, report in enumerate(reports):
        envelope = json.loads(report.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict):
            raise ValueError(f"report {report} must be an extracted report object")
        pair_id = envelope.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"report {report} has no pair_id")
        if pair_id in pair_ids:
            raise ValueError(f"report {report} repeats pair_id {pair_id}")
        pair_ids.add(pair_id)
        if envelope.get("status") != "OK" or envelope.get("correctness_status") != "OK":
            raise ValueError(f"report {report} has no successful status")
        source_report = envelope.get("source_report")
        if not isinstance(source_report, dict) or not SHA256_PATTERN.fullmatch(
            str(source_report.get("sha256", ""))
        ):
            raise ValueError(f"report {report} has no source report identity")
        source_relative = _relative_path(
            source_report.get("path"), f"report {report} source report path"
        )
        source_path = _artifact_path(
            artifact_root, source_relative, f"report {report} source report path"
        )
        if not source_path.is_file():
            raise ValueError(
                f"report {report} source report is absent: {source_relative}"
            )
        source_digest = _sha256(source_path)
        if source_digest != source_report["sha256"]:
            raise ValueError(f"report {report} source report hash mismatch")
        if (
            source_report.get("task") != claim["task"]
            or source_report.get("task_artifact") != claim["task_artifact"]
        ):
            raise ValueError(f"report {report} source report identity mismatch")
        rows = envelope.get("rows")
        selected = (
            [
                row
                for row in rows
                if isinstance(row, dict) and _matches(row, claim["row_selector"])
            ]
            if isinstance(rows, list)
            else []
        )
        if len(selected) != 1:
            raise ValueError(
                f"report {report} requires exactly one selector match, found {len(selected)}"
            )
        selected_row = selected[0]
        if selected_row.get("status") != "OK":
            raise ValueError(f"report {report} selected row status is not OK")
        value = selected_row.get(claim["timing_field"])
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"report {report} has no finite positive timing")
        timing_samples.append(
            {
                "pair_id": pair_id,
                "value_ms": float(value),
                "status": "OK",
                "source_report_sha256": source_report["sha256"],
            }
        )
        source_reports.append(
            {
                "pair_id": pair_id,
                "path": source_relative.as_posix(),
                "sha256": source_digest,
                "status": "OK",
                "correctness_status": "OK",
                "task": claim["task"],
                "task_artifact": claim["task_artifact"],
                "envelope_path": report.resolve()
                .relative_to(artifact_root.resolve())
                .as_posix(),
                "envelope_sha256": _sha256(report),
            }
        )
    row = dict(claim["row_selector"])
    row["status"] = "OK"
    row[claim["timing_field"]] = timing_samples
    return {
        "schema_version": 1,
        "task": claim["task"],
        "evaluation": claim["evaluation"],
        "correctness_source": claim["correctness_source"],
        "fingerprint": dict(fingerprint),
        "source_reports": source_reports,
        "rows": [row],
    }


def _matches(row: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in selector.items())


def _load_artifact(path: Path, row_id: str, role: str) -> tuple[Any, list[str]]:
    if not path.is_file():
        return None, [f"{row_id}: {role} artifact absent: {path}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, ValueError) as exc:
        return None, [f"{row_id}: {role} artifact unreadable: {exc}"]


def _artifact_measurement(
    payload: Any,
    *,
    claim: Mapping[str, Any],
    role: str,
    minimum_pairs: int,
    pairing_keys: list[str],
    artifact_root: Path,
) -> tuple[ArtifactMeasurement | None, list[str]]:
    row_id = str(claim["row_id"])
    prefix = f"{row_id}: {role} artifact"
    if not isinstance(payload, dict):
        return None, [f"{prefix} must be an object"]
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"{prefix} schema_version must be 1")
    for field in ("task", "evaluation", "correctness_source"):
        if payload.get(field) != claim.get(field):
            errors.append(f"{prefix} {field} does not match the claim")
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, dict):
        errors.append(f"{prefix} fingerprint must be an object")
        fingerprint = {}
    for key in pairing_keys:
        if key not in fingerprint or fingerprint[key] in (None, "", []):
            errors.append(f"{prefix} fingerprint is missing pairing key {key}")
    for key in ("torchsky_version", "torchsky_revision"):
        if fingerprint.get(key) in (None, "", []):
            errors.append(f"{prefix} fingerprint is missing subject key {key}")
    expected_fingerprint = {
        "dtype": claim.get("dtype"),
        "device": claim.get("device"),
        "task_command_sha256": claim.get("task_command_sha256"),
        "task_arguments": claim.get("effective_arguments"),
    }
    for key, expected in expected_fingerprint.items():
        if fingerprint.get(key) != expected:
            errors.append(f"{prefix} fingerprint {key} does not match the claim")
    rows = payload.get("rows")
    selector = claim.get("row_selector")
    matches = (
        [row for row in rows if isinstance(row, dict) and _matches(row, selector)]
        if isinstance(rows, list) and isinstance(selector, dict)
        else []
    )
    if len(matches) != 1:
        errors.append(
            f"{prefix} requires exactly one selector match, found {len(matches)}"
        )
        return None, errors
    timing_field = claim.get("timing_field")
    raw_samples = matches[0].get(timing_field)
    if matches[0].get("status") != "OK":
        errors.append(f"{prefix} selected row status must be OK")
    if not isinstance(raw_samples, list):
        errors.append(f"{prefix} timing field {timing_field} must be a sample list")
        return None, errors
    samples: dict[str, float] = {}
    pair_ids: list[str] = []
    sample_hashes: dict[str, str] = {}
    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, dict):
            errors.append(f"{prefix} timing sample {index} must be an object")
            continue
        pair_id = sample.get("pair_id")
        value = sample.get("value_ms")
        if not isinstance(pair_id, str) or not pair_id:
            errors.append(f"{prefix} timing sample {index} has no pair_id")
            continue
        if pair_id in samples:
            errors.append(f"{prefix} has duplicate pair_id {pair_id}")
            continue
        if sample.get("status") != "OK":
            errors.append(f"{prefix} timing sample {pair_id} status must be OK")
        source_hash = sample.get("source_report_sha256")
        if not isinstance(source_hash, str) or not SHA256_PATTERN.fullmatch(
            source_hash
        ):
            errors.append(f"{prefix} timing sample {pair_id} has no source report hash")
        else:
            sample_hashes[pair_id] = source_hash
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            errors.append(
                f"{prefix} timing sample {pair_id} must be finite and positive"
            )
            continue
        samples[pair_id] = float(value)
        pair_ids.append(pair_id)
    if len(samples) < minimum_pairs:
        errors.append(
            f"{prefix} requires at least {minimum_pairs} valid paired samples"
        )
    raw_sources = payload.get("source_reports")
    source_hashes: dict[str, str] = {}
    if not isinstance(raw_sources, list):
        errors.append(f"{prefix} source_reports must be a list")
    else:
        source_pair_ids: list[str] = []
        for index, source in enumerate(raw_sources):
            if not isinstance(source, dict):
                errors.append(f"{prefix} source report {index} must be an object")
                continue
            source_pair_id = source.get("pair_id")
            source_hash = source.get("sha256")
            if not isinstance(source_pair_id, str) or not source_pair_id:
                errors.append(f"{prefix} source report {index} has no pair_id")
                continue
            if source_pair_id in source_hashes:
                errors.append(
                    f"{prefix} has duplicate source report pair_id {source_pair_id}"
                )
                continue
            if not isinstance(source_hash, str) or not SHA256_PATTERN.fullmatch(
                source_hash
            ):
                errors.append(f"{prefix} source report {source_pair_id} has no sha256")
                continue
            if source.get("status") != "OK" or source.get("correctness_status") != "OK":
                errors.append(
                    f"{prefix} source report {source_pair_id} status must be OK"
                )
            if (
                source.get("task") != claim["task"]
                or source.get("task_artifact") != claim["task_artifact"]
            ):
                errors.append(
                    f"{prefix} source report {source_pair_id} identity mismatch"
                )
            envelope_hash = source.get("envelope_sha256")
            if not isinstance(envelope_hash, str) or not SHA256_PATTERN.fullmatch(
                envelope_hash
            ):
                errors.append(
                    f"{prefix} source report {source_pair_id} has no envelope sha256"
                )
                continue
            try:
                envelope_relative = _relative_path(
                    source.get("envelope_path"),
                    f"{prefix} source report {source_pair_id} envelope path",
                )
                envelope_path = _artifact_path(
                    artifact_root,
                    envelope_relative,
                    f"{prefix} source report {source_pair_id} envelope path",
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not envelope_path.is_file():
                errors.append(
                    f"{prefix} source report {source_pair_id} envelope is absent: {envelope_relative}"
                )
                continue
            if _sha256(envelope_path) != envelope_hash:
                errors.append(
                    f"{prefix} source report {source_pair_id} envelope hash mismatch"
                )
                continue
            try:
                envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(
                    f"{prefix} source report {source_pair_id} envelope unreadable: {exc}"
                )
                continue
            envelope_source = (
                envelope.get("source_report") if isinstance(envelope, dict) else None
            )
            if (
                not isinstance(envelope, dict)
                or envelope.get("pair_id") != source_pair_id
                or envelope.get("status") != "OK"
                or envelope.get("correctness_status") != "OK"
                or not isinstance(envelope_source, dict)
                or envelope_source.get("path") != source.get("path")
                or envelope_source.get("sha256") != source_hash
                or envelope_source.get("task") != claim["task"]
                or envelope_source.get("task_artifact") != claim["task_artifact"]
            ):
                errors.append(
                    f"{prefix} source report {source_pair_id} envelope identity mismatch"
                )
                continue
            try:
                source_relative = _relative_path(
                    source.get("path"),
                    f"{prefix} source report {source_pair_id} path",
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            try:
                source_path = _artifact_path(
                    artifact_root,
                    source_relative,
                    f"{prefix} source report {source_pair_id} path",
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not source_path.is_file():
                errors.append(
                    f"{prefix} source report {source_pair_id} is absent: {source_relative}"
                )
                continue
            actual_hash = _sha256(source_path)
            if actual_hash != source_hash:
                errors.append(f"{prefix} source report {source_pair_id} hash mismatch")
                continue
            try:
                source_rows = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(
                    f"{prefix} source report {source_pair_id} unreadable: {exc}"
                )
                continue
            source_matches = (
                [
                    row
                    for row in source_rows
                    if isinstance(row, dict) and _matches(row, claim["row_selector"])
                ]
                if isinstance(source_rows, list)
                else []
            )
            if len(source_matches) != 1:
                errors.append(
                    f"{prefix} source report {source_pair_id} requires exactly one selector match, found {len(source_matches)}"
                )
                continue
            raw_value = source_matches[0].get(claim["timing_field"])
            raw_status = source_matches[0].get("status", "OK")
            if raw_status != "OK":
                errors.append(
                    f"{prefix} source report {source_pair_id} row status is not OK"
                )
                continue
            if (
                not isinstance(raw_value, (int, float))
                or isinstance(raw_value, bool)
                or not math.isfinite(float(raw_value))
                or samples.get(source_pair_id) != float(raw_value)
            ):
                errors.append(
                    f"{prefix} source report {source_pair_id} timing does not match"
                )
                continue
            source_pair_ids.append(source_pair_id)
            source_hashes[source_pair_id] = actual_hash
        if source_pair_ids != pair_ids:
            errors.append(f"{prefix} source report order does not match sample order")
        if source_hashes != sample_hashes:
            errors.append(f"{prefix} source report hashes do not match samples")
    if errors:
        return None, errors
    return ArtifactMeasurement(
        fingerprint=dict(fingerprint), samples=samples, pair_ids=pair_ids
    ), []


def _config_errors(payload: Mapping[str, Any], *, root: Path) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return [*errors, "claims must be a non-empty list"]
    try:
        tasks = tomllib.loads((root / "pixi.toml").read_text(encoding="utf-8"))["tasks"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        return [*errors, f"Pixi task registry unreadable: {exc}"]
    required = {
        "row_id",
        "task",
        "evaluation",
        "row_selector",
        "timing_field",
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
    }
    seen: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"claim {index}: must be an object")
            continue
        missing = sorted(required.difference(claim))
        if missing:
            errors.append(f"claim {index}: missing fields: {', '.join(missing)}")
            continue
        row_id = claim["row_id"]
        if not isinstance(row_id, str) or not row_id:
            errors.append(f"claim {index}: row_id must be a non-empty string")
            continue
        if row_id in seen:
            errors.append(f"{row_id}: duplicate row_id")
        seen.add(row_id)
        task = claim["task"]
        if not isinstance(task, str) or task not in tasks:
            errors.append(f"{row_id}: unknown Pixi task {task}")
        evaluation = claim["evaluation"]
        evidence_claim = CLAIMS.get(evaluation) if isinstance(evaluation, str) else None
        if evidence_claim is None:
            errors.append(f"{row_id}: evaluation has no correctness claim")
        task_command = _task_command(tasks, task) if isinstance(task, str) else ""
        if isinstance(evaluation, str) and f" run {evaluation} " not in task_command:
            errors.append(f"{row_id}: Pixi task does not run evaluation {evaluation}")
        if f"--json-out {claim['task_artifact']}" not in task_command:
            errors.append(f"{row_id}: Pixi task does not write task_artifact")
        command_hash = hashlib.sha256(task_command.encode("utf-8")).hexdigest()
        configured_command_hash = claim["task_command_sha256"]
        if (
            not isinstance(configured_command_hash, str)
            or not SHA256_PATTERN.fullmatch(configured_command_hash)
            or configured_command_hash != command_hash
        ):
            errors.append(f"{row_id}: task command hash does not match Pixi task")
        effective_arguments = claim["effective_arguments"]
        if (
            not isinstance(effective_arguments, list)
            or not effective_arguments
            or not all(isinstance(argument, str) for argument in effective_arguments)
        ):
            errors.append(
                f"{row_id}: effective_arguments must be a non-empty string list"
            )
        elif effective_arguments != _effective_arguments(task_command):
            errors.append(f"{row_id}: effective_arguments do not match Pixi task")
        if claim["dtype"] not in {"float32", "float64"}:
            errors.append(f"{row_id}: dtype must be float32 or float64")
        if claim["device"] != "cpu":
            errors.append(f"{row_id}: release performance device must be cpu")
        selector = claim["row_selector"]
        if not isinstance(selector, dict) or not selector:
            errors.append(f"{row_id}: row_selector must be a non-empty object")
        timing_field = claim["timing_field"]
        if not isinstance(timing_field, str) or not timing_field:
            errors.append(f"{row_id}: timing_field must be a non-empty string")
        for field in ("baseline_artifact", "candidate_artifact", "task_artifact"):
            if not isinstance(claim[field], str) or not claim[field]:
                errors.append(f"{row_id}: {field} must be a non-empty string")
            else:
                try:
                    _relative_path(claim[field], f"{row_id}: {field}")
                except ValueError as exc:
                    errors.append(str(exc))
        if claim["max_regression_fraction"] != 0.05:
            errors.append(f"{row_id}: max_regression_fraction must be 0.05")
        floor = claim["absolute_timing_floor_ms"]
        if (
            not isinstance(floor, (int, float))
            or isinstance(floor, bool)
            or not math.isfinite(float(floor))
            or float(floor) <= 0.0
        ):
            errors.append(
                f"{row_id}: absolute_timing_floor_ms must be finite and positive"
            )
        policy = claim["paired_sampling_policy"]
        if not isinstance(policy, dict):
            errors.append(f"{row_id}: paired_sampling_policy must be an object")
        else:
            if policy.get("mode") != "same_runner_alternating":
                errors.append(f"{row_id}: paired sampling must use the same runner")
            if policy.get("order") != "seeded_alternating":
                errors.append(
                    f"{row_id}: paired sampling order must be seeded alternating"
                )
            if policy.get("statistic") != "median_paired_ratio":
                errors.append(f"{row_id}: paired statistic must be median_paired_ratio")
            minimum_pairs = policy.get("minimum_pairs")
            if (
                not isinstance(minimum_pairs, int)
                or isinstance(minimum_pairs, bool)
                or minimum_pairs < 5
            ):
                errors.append(f"{row_id}: paired sampling requires at least five pairs")
            pairing_keys = policy.get("pairing_keys")
            if (
                not isinstance(pairing_keys, list)
                or not pairing_keys
                or len(pairing_keys) != len(set(pairing_keys))
                or not all(isinstance(key, str) and key for key in pairing_keys)
            ):
                errors.append(
                    f"{row_id}: pairing_keys must be unique non-empty strings"
                )
        source = claim["correctness_source"]
        if not isinstance(source, dict):
            errors.append(f"{row_id}: correctness_source must be an object")
        elif source.get("type") == "evidence_claim":
            if (
                evidence_claim is None
                or source.get("claim") != evaluation
                or source.get("required_row") not in evidence_claim.required_rows
            ):
                errors.append(f"{row_id}: correctness evidence is inconsistent")
        elif source.get("type") == "acceptance_artifact":
            artifact = root / str(source.get("artifact", ""))
            try:
                rows = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"{row_id}: correctness artifact unreadable: {exc}")
            else:
                source_selector = source.get("selector")
                matches = (
                    [
                        row
                        for row in rows
                        if isinstance(row, dict) and _matches(row, source_selector)
                    ]
                    if isinstance(rows, list) and isinstance(source_selector, dict)
                    else []
                )
                if len(matches) != 1:
                    errors.append(
                        f"{row_id}: correctness selector must match exactly once"
                    )
                if (
                    evidence_claim is None
                    or not isinstance(source_selector, dict)
                    or tuple(source_selector.items())
                    not in evidence_claim.required_row_selectors
                ):
                    errors.append(
                        f"{row_id}: correctness selector is not required by the evaluation"
                    )
        else:
            errors.append(f"{row_id}: unknown correctness source type")
    return errors


def validate_configuration(payload: Mapping[str, Any], *, root: Path) -> list[str]:
    return _config_errors(payload, root=root)


def _assess(
    payload: Mapping[str, Any], *, root: Path, artifact_root: Path
) -> tuple[list[str], list[dict[str, Any]]]:
    errors = _config_errors(payload, root=root)
    claims = payload.get("claims")
    if errors or not isinstance(claims, list):
        return errors, []
    results: list[dict[str, Any]] = []
    for claim in claims:
        row_id = str(claim["row_id"])
        policy = claim["paired_sampling_policy"]
        minimum_pairs = int(policy["minimum_pairs"])
        pairing_keys = list(policy["pairing_keys"])
        measurements: dict[str, ArtifactMeasurement] = {}
        claim_errors: list[str] = []
        for role, field in (
            ("baseline", "baseline_artifact"),
            ("candidate", "candidate_artifact"),
        ):
            try:
                artifact_relative = _relative_path(claim[field], f"{row_id}: {field}")
            except ValueError as exc:
                claim_errors.append(str(exc))
                continue
            try:
                artifact_path = _artifact_path(
                    artifact_root, artifact_relative, f"{row_id}: {field}"
                )
            except ValueError as exc:
                claim_errors.append(str(exc))
                continue
            artifact, load_errors = _load_artifact(artifact_path, row_id, role)
            claim_errors.extend(load_errors)
            if load_errors:
                continue
            measurement, artifact_errors = _artifact_measurement(
                artifact,
                claim=claim,
                role=role,
                minimum_pairs=minimum_pairs,
                pairing_keys=pairing_keys,
                artifact_root=artifact_root,
            )
            claim_errors.extend(artifact_errors)
            if measurement is not None:
                measurements[role] = measurement
        if claim_errors:
            errors.extend(claim_errors)
            continue
        baseline = measurements["baseline"]
        candidate = measurements["candidate"]
        fingerprint_mismatches = [
            key
            for key in pairing_keys
            if baseline.fingerprint[key] != candidate.fingerprint[key]
        ]
        if fingerprint_mismatches:
            errors.append(
                f"{row_id}: runner fingerprint mismatch for "
                + ", ".join(fingerprint_mismatches)
            )
            continue
        if set(baseline.samples) != set(candidate.samples):
            errors.append(f"{row_id}: baseline and candidate pair IDs do not match")
            continue
        if baseline.pair_ids != candidate.pair_ids:
            errors.append(f"{row_id}: baseline and candidate pair order does not match")
            continue
        pair_ids = baseline.pair_ids
        ratios = [
            candidate.samples[pair_id] / baseline.samples[pair_id]
            for pair_id in pair_ids
        ]
        baseline_median_ms = float(median(baseline.samples.values()))
        median_ratio = float(median(ratios))
        floor = float(claim["absolute_timing_floor_ms"])
        gated = baseline_median_ms >= floor
        passed = not gated or median_ratio <= 1.0 + float(
            claim["max_regression_fraction"]
        )
        result = {
            "row_id": row_id,
            "pair_ids": pair_ids,
            "paired_ratios": ratios,
            "median_paired_ratio": median_ratio,
            "baseline_median_ms": baseline_median_ms,
            "absolute_timing_floor_ms": floor,
            "regression_gate_applied": gated,
            "passed": passed,
        }
        results.append(result)
        if not passed:
            errors.append(
                f"{row_id}: median paired ratio {median_ratio:.6g} exceeds 1.05"
            )
    return errors, results


def validate_claims(
    payload: Mapping[str, Any], *, root: Path, artifact_root: Path | None = None
) -> list[str]:
    errors, _ = _assess(
        payload, root=root, artifact_root=(artifact_root or root).resolve()
    )
    return errors


def assess_claims(
    payload: Mapping[str, Any], *, root: Path, artifact_root: Path | None = None
) -> dict[str, Any]:
    errors, results = _assess(
        payload, root=root, artifact_root=(artifact_root or root).resolve()
    )
    return {
        "release_performance_ready": not errors,
        "errors": errors,
        "claims": results,
    }
