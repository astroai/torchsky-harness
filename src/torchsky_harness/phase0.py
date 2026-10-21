"""Phase-0 comparator configuration and evidence provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
from typing import Any, Mapping, Sequence


IDENTITY_KINDS = frozenset({"subject", "peer", "data", "crds", "index"})
PEER_KINDS = frozenset({"python", "binary", "source"})
STATUS_VALUES = frozenset({"available", "missing", "unsupported"})
CAMPAIGN_STATUS_VALUES = frozenset({"stub", "bounded"})


class Phase0ConfigError(ValueError):
    pass


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if not isinstance(payload, dict):
        raise Phase0ConfigError(f"{path}: TOML root must be a table")
    return payload


def _require_string(record: Mapping[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Phase0ConfigError(f"{context}: {key} must be a non-empty string")
    return value


def _string_list(record: Mapping[str, Any], key: str, context: str) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise Phase0ConfigError(f"{context}: {key} must be a string list")
    if len(value) != len(set(value)):
        raise Phase0ConfigError(f"{context}: {key} contains duplicates")
    return value


def load_peer_catalog(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_toml(path)
    if payload.get("schema_version") != 1:
        raise Phase0ConfigError(f"{path}: unsupported peer catalog schema")
    raw_peers = payload.get("peers")
    if not isinstance(raw_peers, list) or not raw_peers:
        raise Phase0ConfigError(f"{path}: peers must be a non-empty array")
    peers: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_peers):
        context = f"{path}: peers[{index}]"
        if not isinstance(raw, dict):
            raise Phase0ConfigError(f"{context} must be a table")
        peer_id = _require_string(raw, "id", context)
        if peer_id in peers:
            raise Phase0ConfigError(f"{path}: duplicate peer id {peer_id}")
        kind = _require_string(raw, "kind", context)
        if kind not in PEER_KINDS:
            raise Phase0ConfigError(f"{context}: unsupported kind {kind}")
        _string_list(raw, "platforms", context)
        if kind == "python":
            _require_string(raw, "distribution", context)
            _require_string(raw, "import_name", context)
        elif kind == "binary":
            _require_string(raw, "command", context)
        else:
            _require_string(raw, "path_env", context)
        bootstrap = raw.get("bootstrap")
        if bootstrap is not None:
            if not isinstance(bootstrap, dict):
                raise Phase0ConfigError(f"{context}: bootstrap must be a table")
            manager = _require_string(bootstrap, "manager", context)
            if manager not in {"pixi", "pip", "manual"}:
                raise Phase0ConfigError(
                    f"{context}: unsupported bootstrap manager {manager}"
                )
            _require_string(bootstrap, "spec", context)
        peers[peer_id] = dict(raw)
    return peers


def load_campaign(path: Path, peers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = _load_toml(path)
    context = str(path)
    if payload.get("schema_version") != 1:
        raise Phase0ConfigError(f"{path}: unsupported campaign schema")
    _require_string(payload, "id", context)
    _require_string(payload, "domain", context)
    status = payload.get("status")
    if status not in CAMPAIGN_STATUS_VALUES:
        raise Phase0ConfigError(
            f"{path}: Phase-0 campaign status must be stub or bounded"
        )
    if not isinstance(payload.get("scheduled_certification"), bool):
        raise Phase0ConfigError(f"{path}: scheduled_certification must be boolean")
    if payload.get("runs_large_peers") is not False:
        raise Phase0ConfigError(f"{path}: runs_large_peers must be false")
    environments = _string_list(payload, "environments", context)
    if not environments:
        raise Phase0ConfigError(f"{path}: environments cannot be empty")
    required = _string_list(payload, "required_peers", context)
    optional = _string_list(payload, "optional_peers", context)
    if set(required) & set(optional):
        raise Phase0ConfigError(f"{path}: required and optional peers overlap")
    unknown = sorted((set(required) | set(optional)) - set(peers))
    if unknown:
        raise Phase0ConfigError(f"{path}: unknown peers: {', '.join(unknown)}")
    if payload["scheduled_certification"] and not required:
        raise Phase0ConfigError(f"{path}: scheduled certification needs required peers")
    identities = _string_list(payload, "required_identities", context)
    invalid_identities = sorted(set(identities) - IDENTITY_KINDS)
    if invalid_identities:
        raise Phase0ConfigError(
            f"{path}: invalid identity kinds: {', '.join(invalid_identities)}"
        )
    for mandatory in ("subject", "peer", "data"):
        if mandatory not in identities:
            raise Phase0ConfigError(f"{path}: required_identities omits {mandatory}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise Phase0ConfigError(f"{path}: datasets must be a non-empty array")
    dataset_ids: set[str] = set()
    for index, dataset in enumerate(datasets):
        dataset_context = f"{path}: datasets[{index}]"
        if not isinstance(dataset, dict):
            raise Phase0ConfigError(f"{dataset_context} must be a table")
        dataset_id = _require_string(dataset, "id", dataset_context)
        _require_string(dataset, "role", dataset_context)
        if dataset_id in dataset_ids:
            raise Phase0ConfigError(f"{path}: duplicate dataset id {dataset_id}")
        dataset_ids.add(dataset_id)
    if status == "bounded":
        matrix = _require_string(payload, "matrix", context)
        matrix_path = (path.parent / matrix).resolve()
        if not matrix_path.is_file():
            raise Phase0ConfigError(f"{path}: matrix does not exist: {matrix}")
        data_manifests = _string_list(payload, "data_manifests", context)
        if not data_manifests:
            raise Phase0ConfigError(f"{path}: data_manifests cannot be empty")
        for manifest in data_manifests:
            if not (path.parent / manifest).resolve().is_file():
                raise Phase0ConfigError(
                    f"{path}: data manifest does not exist: {manifest}"
                )
        required_metrics = _string_list(payload, "required_metrics", context)
        if not required_metrics:
            raise Phase0ConfigError(f"{path}: required_metrics cannot be empty")
        max_cases = payload.get("max_cases")
        if (
            not isinstance(max_cases, int)
            or isinstance(max_cases, bool)
            or max_cases < 1
        ):
            raise Phase0ConfigError(f"{path}: max_cases must be a positive integer")
        matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        validate_campaign_matrix(matrix_payload, campaign_id=str(payload["id"]))
        if len(matrix_payload["cases"]) > max_cases:
            raise Phase0ConfigError(
                f"{path}: matrix has {len(matrix_payload['cases'])} cases, above max_cases={max_cases}"
            )
    return payload


def validate_campaign_matrix(
    payload: Mapping[str, Any], *, campaign_id: str | None = None
) -> None:
    if payload.get("schema_version") != 1:
        raise Phase0ConfigError("matrix: unsupported schema_version")
    matrix_id = _require_string(payload, "id", "matrix")
    declared_campaign = _require_string(payload, "campaign_id", "matrix")
    if campaign_id is not None and declared_campaign != campaign_id:
        raise Phase0ConfigError(
            f"matrix {matrix_id}: campaign_id {declared_campaign!r} does not match {campaign_id!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise Phase0ConfigError("matrix: cases must be a non-empty array")
    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        context = f"matrix: cases[{index}]"
        if not isinstance(case, dict):
            raise Phase0ConfigError(f"{context} must be an object")
        case_id = _require_string(case, "id", context)
        _require_string(case, "kind", context)
        peers = _string_list(case, "required_peers", context)
        metrics = _string_list(case, "required_metrics", context)
        if not metrics:
            raise Phase0ConfigError(f"{context}: required_metrics cannot be empty")
        if case_id in case_ids:
            raise Phase0ConfigError(f"matrix: duplicate case id {case_id}")
        case_ids.add(case_id)
        if not peers and case.get("certification") == "peer":
            raise Phase0ConfigError(
                f"{context}: peer certification needs required_peers"
            )
        parameters = case.get("parameters")
        if not isinstance(parameters, dict):
            raise Phase0ConfigError(f"{context}: parameters must be an object")


def discover_campaigns(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.glob("*.toml") if path.name != "peers.toml"
    )


def validate_configs(
    directory: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    peers = load_peer_catalog(directory / "peers.toml")
    campaign_paths = discover_campaigns(directory)
    if not campaign_paths:
        raise Phase0ConfigError(f"{directory}: no campaign configs found")
    campaigns = [load_campaign(path, peers) for path in campaign_paths]
    ids = [str(campaign["id"]) for campaign in campaigns]
    if len(ids) != len(set(ids)):
        raise Phase0ConfigError(f"{directory}: duplicate campaign ids")
    return peers, campaigns


def _platform_key() -> str:
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "darwin":
        return "osx-arm64" if machine in {"arm64", "aarch64"} else "osx-64"
    if system == "linux":
        return "linux-aarch64" if machine in {"arm64", "aarch64"} else "linux-64"
    return f"{system}-{machine}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_peer(peer: Mapping[str, Any]) -> dict[str, Any]:
    distribution = str(peer["distribution"])
    import_name = str(peer["import_name"])
    record: dict[str, Any] = {
        "identity_type": "peer",
        "id": peer["id"],
        "kind": "python",
        "distribution": distribution,
        "import_name": import_name,
    }
    try:
        installed = metadata.distribution(distribution)
        record["version"] = installed.version
        spec = importlib.util.find_spec(import_name)
    except (metadata.PackageNotFoundError, ImportError, AttributeError) as exc:
        record.update(status="missing", reason=str(exc))
        return record
    if spec is None:
        record.update(status="missing", reason=f"cannot resolve import {import_name}")
        return record
    direct_url = installed.read_text("direct_url.json")
    package_record = installed.read_text("RECORD")
    record.update(
        status="available",
        path=str(Path(spec.origin).resolve()) if spec.origin else None,
        direct_url=json.loads(direct_url) if direct_url else None,
        record_sha256=hashlib.sha256(package_record.encode()).hexdigest()
        if package_record
        else None,
    )
    return record


def _binary_peer(peer: Mapping[str, Any]) -> dict[str, Any]:
    command = str(peer["command"])
    path = shutil.which(command)
    record: dict[str, Any] = {
        "identity_type": "peer",
        "id": peer["id"],
        "kind": "binary",
        "command": command,
    }
    if path is None:
        record.update(status="missing", reason=f"{command} not found on PATH")
        return record
    version_args = peer.get("version_args", ["--version"])
    if not isinstance(version_args, list) or any(
        not isinstance(item, str) for item in version_args
    ):
        raise Phase0ConfigError(f"peer {peer['id']}: version_args must be strings")
    try:
        completed = subprocess.run(
            [path, *version_args],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        record.update(status="missing", reason=f"version probe failed: {exc}")
        return record
    output = "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    first_line = output.splitlines()[0] if output else f"exit={completed.returncode}"
    resolved = Path(path).resolve()
    record.update(
        status="available",
        path=str(resolved),
        version=first_line,
        executable_sha256=_sha256(resolved),
    )
    return record


def _source_peer(peer: Mapping[str, Any]) -> dict[str, Any]:
    path_env = str(peer["path_env"])
    value = os.environ.get(path_env)
    record: dict[str, Any] = {
        "identity_type": "peer",
        "id": peer["id"],
        "kind": "source",
        "path_env": path_env,
    }
    if not value:
        record.update(status="missing", reason=f"{path_env} is not set")
        return record
    path = Path(value).expanduser().resolve()
    if not path.exists():
        record.update(status="missing", reason=f"{path} does not exist")
        return record
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    record.update(
        status="available",
        path=str(path),
        version=revision.stdout.strip() if revision.returncode == 0 else "unversioned",
        revision=revision.stdout.strip() if revision.returncode == 0 else None,
        dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    )
    return record


def inspect_peer(
    peer: Mapping[str, Any], platform_key: str | None = None
) -> dict[str, Any]:
    selected_platform = platform_key or _platform_key()
    supported = peer.get("platforms", [])
    if selected_platform not in supported:
        return {
            "identity_type": "peer",
            "id": peer["id"],
            "kind": peer["kind"],
            "status": "unsupported",
            "reason": f"{selected_platform} not in {supported}",
        }
    if peer["kind"] == "python":
        return _python_peer(peer)
    if peer["kind"] == "binary":
        return _binary_peer(peer)
    return _source_peer(peer)


def _subject_identity() -> dict[str, Any]:
    record: dict[str, Any] = {
        "identity_type": "subject",
        "distribution": "torchsky",
    }
    try:
        distribution = metadata.distribution("torchsky")
    except metadata.PackageNotFoundError as exc:
        record.update(status="missing", reason=str(exc))
        return record
    module_files = [
        (str(path), Path(path.locate()).resolve())
        for path in distribution.files or []
        if str(path).startswith("torchsky/")
        and Path(path.locate()).is_file()
        and Path(path).suffix in {".py", ".so", ".pyd"}
    ]
    digest = hashlib.sha256()
    for relative, path in sorted(module_files):
        digest.update(relative.encode())
        digest.update(bytes.fromhex(_sha256(path)))
    spec = importlib.util.find_spec("torchsky")
    direct_url = distribution.read_text("direct_url.json")
    record.update(
        status="available",
        version=distribution.version,
        import_path=str(Path(spec.origin).resolve()) if spec and spec.origin else None,
        package_content_sha256=digest.hexdigest(),
        source=json.loads(direct_url) if direct_url else {"kind": "registry"},
    )
    return record


def _crds_identity() -> dict[str, Any] | None:
    context = os.environ.get("CRDS_CONTEXT")
    cache_path = os.environ.get("CRDS_PATH")
    if not context:
        return None
    record = {
        "identity_type": "crds",
        "observatory": "jwst",
        "context": context,
        "server_url": os.environ.get("CRDS_SERVER_URL"),
        "cache_path": str(Path(cache_path).expanduser().resolve())
        if cache_path
        else None,
    }
    context_file = os.environ.get("CRDS_CONTEXT_FILE")
    if context_file:
        path = Path(context_file).expanduser().resolve()
        if path.is_file():
            record["mapping_path"] = str(path)
            record["mapping_sha256"] = _sha256(path)
    return record


def _data_identities(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            if not isinstance(value, dict) or value.get("identity_type") != "data":
                raise Phase0ConfigError(f"{path}: expected data identity record(s)")
            records.append(value)
    return records


def _index_identities() -> list[dict[str, Any]]:
    value = os.environ.get("ASTROMETRY_INDEX_MANIFEST")
    if not value:
        return []
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    index_id = payload.get("id") if isinstance(payload, dict) else None
    version = payload.get("version") if isinstance(payload, dict) else None
    return [
        {
            "identity_type": "index",
            "id": str(index_id or path.stem),
            "format": "astrometry.net",
            "version": str(version or "unknown"),
            "manifest_path": str(path),
            "manifest_sha256": _sha256(path),
        }
    ]


def coverage_failures(
    campaign: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    peer_records = {
        str(record.get("id")): record
        for record in manifest.get("peers", [])
        if isinstance(record, dict)
    }
    for peer_id in campaign["required_peers"]:
        record = peer_records.get(peer_id)
        if record is None:
            failures.append(f"missing required peer identity: {peer_id}")
        elif record.get("status") != "available":
            failures.append(
                f"required peer unavailable: {peer_id} ({record.get('status', 'missing')})"
            )
        elif not record.get("version") or not record.get("path"):
            failures.append(f"required peer identity incomplete: {peer_id}")
    required_identities = set(campaign["required_identities"])
    subject = manifest.get("subject")
    if "subject" in required_identities:
        if not isinstance(subject, dict) or subject.get("status") != "available":
            failures.append("subject identity unavailable")
        elif any(
            not subject.get(key)
            for key in ("version", "import_path", "package_content_sha256")
        ):
            failures.append("subject identity incomplete")
    data_records = {
        str(record.get("id")): record
        for record in manifest.get("data", [])
        if isinstance(record, dict)
    }
    for dataset in campaign["datasets"]:
        record = data_records.get(dataset["id"])
        if record is None:
            failures.append(f"missing data identity: {dataset['id']}")
        elif any(not record.get(key) for key in ("role", "uri", "sha256")):
            failures.append(f"incomplete data identity: {dataset['id']}")
    crds = manifest.get("crds")
    if "crds" in required_identities:
        if not isinstance(crds, dict):
            failures.append("CRDS identity unavailable")
        elif not crds.get("context"):
            failures.append("CRDS identity incomplete")
    indices = manifest.get("indices")
    if "index" in required_identities:
        if not indices:
            failures.append("index identity unavailable")
        elif any(
            not isinstance(record, dict)
            or not record.get("version")
            or not record.get("manifest_sha256")
            for record in indices
        ):
            failures.append("index identity incomplete")
    return failures


def build_manifest(
    campaign: Mapping[str, Any],
    peers: Mapping[str, Mapping[str, Any]],
    *,
    data_manifest_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    selected_ids = [*campaign["required_peers"], *campaign["optional_peers"]]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": campaign["id"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": _platform_key(),
        "python": sys.version.split()[0],
        "subject": _subject_identity(),
        "peers": [inspect_peer(peers[peer_id]) for peer_id in selected_ids],
        "data": _data_identities(data_manifest_paths),
        "crds": _crds_identity(),
        "indices": _index_identities(),
    }
    failures = coverage_failures(campaign, manifest)
    manifest["coverage"] = {
        "status": "fail" if failures else "pass",
        "failures": failures,
    }
    return manifest


def validate_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise Phase0ConfigError("manifest: unsupported schema_version")
    _require_string(payload, "campaign_id", "manifest")
    _require_string(payload, "generated_at", "manifest")
    subject = payload.get("subject")
    if not isinstance(subject, dict) or subject.get("identity_type") != "subject":
        raise Phase0ConfigError("manifest: invalid subject identity")
    if subject.get("status") not in STATUS_VALUES:
        raise Phase0ConfigError("manifest: invalid subject status")
    for key, identity_type in (
        ("peers", "peer"),
        ("data", "data"),
        ("indices", "index"),
    ):
        records = payload.get(key)
        if not isinstance(records, list):
            raise Phase0ConfigError(f"manifest: {key} must be an array")
        if any(
            not isinstance(record, dict) or record.get("identity_type") != identity_type
            for record in records
        ):
            raise Phase0ConfigError(f"manifest: invalid {identity_type} identity")
    crds = payload.get("crds")
    if crds is not None and (
        not isinstance(crds, dict) or crds.get("identity_type") != "crds"
    ):
        raise Phase0ConfigError("manifest: invalid CRDS identity")
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("status") not in {"pass", "fail"}:
        raise Phase0ConfigError("manifest: invalid coverage record")
    if not isinstance(coverage.get("failures"), list):
        raise Phase0ConfigError("manifest: coverage failures must be an array")


def bootstrap_missing(
    campaign: Mapping[str, Any], peers: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    unavailable: list[str] = []
    for peer_id in [*campaign["required_peers"], *campaign["optional_peers"]]:
        peer = peers[peer_id]
        if inspect_peer(peer).get("status") == "available":
            continue
        bootstrap = peer.get("bootstrap")
        if not isinstance(bootstrap, dict) or bootstrap.get("manager") != "pip":
            unavailable.append(peer_id)
            continue
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(bootstrap["spec"])],
            check=False,
        )
        if completed.returncode != 0 or inspect_peer(peer).get("status") != "available":
            unavailable.append(peer_id)
    return unavailable


def _campaign_path(directory: Path, value: str) -> Path:
    path = Path(value)
    if path.suffix == ".toml":
        return path
    return directory / f"{value}.toml"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("campaigns/phase0"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-configs")
    for command in ("bootstrap", "provenance", "certify"):
        child = subparsers.add_parser(command)
        child.add_argument("campaign")
        child.add_argument("--json-out", type=Path)
        child.add_argument("--data-manifest", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    peers, campaigns = validate_configs(args.config_dir)
    if args.command == "validate-configs":
        print(f"validated {len(campaigns)} Phase-0 campaign configs")
        return 0
    path = _campaign_path(args.config_dir, args.campaign)
    campaign = load_campaign(path, peers)
    if args.command == "bootstrap":
        unavailable = bootstrap_missing(campaign, peers)
        if unavailable:
            print("unavailable peers: " + ", ".join(unavailable))
    manifest = build_manifest(campaign, peers, data_manifest_paths=args.data_manifest)
    validate_manifest(manifest)
    text = json.dumps(manifest, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    if args.command == "certify" and manifest["coverage"]["failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
