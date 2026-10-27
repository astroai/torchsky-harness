from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
from typing import Any, Callable, Mapping, Sequence

from torchsky_harness.release_performance import load_claims, validate_configuration


SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
TASK_ENVIRONMENTS = {
    "bench-wcs-release-gate": "bench-all",
    "bench-healpix-release-gate": "sphere-bench",
    "bench-map-drizzle-required": "map-compat",
}


@dataclass(frozen=True)
class SubjectSpec:
    role: str
    revision: str | None
    wheel: Path | None

    @property
    def immutable_identity(self) -> dict[str, str]:
        if self.revision is not None:
            return {"kind": "git", "revision": self.revision}
        if self.wheel is None:
            raise ValueError(f"{self.role} subject is unspecified")
        return {
            "kind": "wheel",
            "filename": self.wheel.name,
            "sha256": sha256(self.wheel),
        }


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        text=True,
        capture_output=capture_output,
        check=True,
    )


def _require_exact_sha(value: str, label: str) -> str:
    if not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be an exact lowercase 40-character SHA")
    return value


def _subject_spec(role: str, revision: str | None, wheel: Path | None) -> SubjectSpec:
    if (revision is None) == (wheel is None):
        raise ValueError(f"{role} requires exactly one SHA or wheel path")
    if revision is not None:
        return SubjectSpec(role, _require_exact_sha(revision, f"{role} SHA"), None)
    assert wheel is not None
    return SubjectSpec(role, None, wheel.resolve(strict=True))


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _verify_checkout(root: Path, revision: str, label: str) -> None:
    actual = _git_output(root, "rev-parse", "HEAD")
    dirty = _git_output(root, "status", "--porcelain")
    if actual != revision:
        raise RuntimeError(f"{label} checkout is {actual}, expected {revision}")
    if dirty:
        raise RuntimeError(f"{label} checkout is dirty")


def _clone_checkout(repository: str, revision: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing existing checkout: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(destination),
        ]
    )
    _run(["git", "checkout", "--detach", revision], cwd=destination)
    _verify_checkout(destination, revision, str(destination))


def measure_environment(device: str) -> dict[str, Any]:
    import torch

    from torchsky_harness.subject import identify_subject

    if device != "cpu":
        raise ValueError("release performance measurement requires cpu")
    torch.empty(1, device=device)
    subject = identify_subject()
    source = subject.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("installed subject has no source identity")
    revision = source.get("revision") or source.get("sha256")
    if not revision:
        revision = subject.get("package_content_sha256")
    cpu = platform.processor().strip()
    if not cpu and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "runner": socket.gethostname(),
        "cpu": cpu or f"unknown-{os.cpu_count() or 0}-logical-cpus",
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchsky_version": str(subject["version"]),
        "torchsky_revision": str(revision),
        "device": device,
    }


def extract_claim_report(
    *,
    claim: Mapping[str, Any],
    source_report: Path,
    extracted_report: Path,
    pair_id: str,
    role: str,
    artifact_root: Path,
) -> dict[str, Any]:
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else None
    selector = claim["row_selector"]
    selected = (
        [
            row
            for row in rows
            if isinstance(row, dict)
            and all(row.get(key) == value for key, value in selector.items())
        ]
        if rows is not None
        else []
    )
    if len(selected) != 1:
        raise ValueError(
            f"{source_report} requires exactly one row for {claim['row_id']}, found {len(selected)}"
        )
    source_row = selected[0]
    status = source_row.get("status", "OK")
    if status != "OK":
        raise ValueError(f"{source_report} selected row status is {status!r}")
    timing = source_row.get(claim["timing_field"])
    if (
        not isinstance(timing, (int, float))
        or isinstance(timing, bool)
        or not math.isfinite(float(timing))
        or float(timing) <= 0.0
    ):
        raise ValueError(f"{source_report} selected row has no finite positive timing")
    row = {
        **selector,
        "status": "OK",
        claim["timing_field"]: float(timing),
    }
    source_relative = source_report.resolve().relative_to(artifact_root.resolve())
    record = {
        "schema_version": 1,
        "pair_id": pair_id,
        "role": role,
        "status": "OK",
        "correctness_status": "OK",
        "source_report": {
            "path": source_relative.as_posix(),
            "sha256": sha256(source_report),
            "task": claim["task"],
            "task_artifact": claim["task_artifact"],
        },
        "rows": [row],
    }
    extracted_report.parent.mkdir(parents=True, exist_ok=True)
    extracted_report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


class PairedPerformanceRunner:
    def __init__(
        self,
        *,
        harness_revision: str,
        baseline: SubjectSpec,
        candidate: SubjectSpec,
        output_root: Path,
        work_root: Path | None,
        pairs: int,
        harness_repository: str,
        subject_repository: str,
        command_runner: RunCommand = _run,
    ) -> None:
        self.harness_revision = _require_exact_sha(harness_revision, "harness SHA")
        self.subjects = {"baseline": baseline, "candidate": candidate}
        self.output_root = output_root.resolve()
        self.work_root = (
            work_root.resolve()
            if work_root is not None
            else self.output_root.parent / f".{self.output_root.name}-work"
        )
        self.pairs = pairs
        self.harness_repository = harness_repository
        self.subject_repository = subject_repository
        self.command_runner = command_runner
        self.roots: dict[str, Path] = {}
        self.identities: dict[str, dict[str, Any]] = {}
        self.environment_identities: dict[str, dict[str, dict[str, Any]]] = {}
        self.environment_measurements: dict[str, dict[str, dict[str, Any]]] = {}
        self.execution: list[dict[str, Any]] = []

    def _command(
        self, command: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return self.command_runner(command, **kwargs)

    def prepare(self) -> None:
        if self.pairs < 5:
            raise ValueError("paired performance evidence requires at least five pairs")
        if self.output_root.exists():
            raise FileExistsError(f"output root already exists: {self.output_root}")
        self.output_root.mkdir(parents=True)
        if self.work_root.exists():
            raise FileExistsError(f"work root already exists: {self.work_root}")
        self.work_root.mkdir(parents=True)
        required_environments = sorted(set(TASK_ENVIRONMENTS.values()) | {"test"})
        for role, subject in self.subjects.items():
            harness_root = self.work_root / role / "harness"
            _clone_checkout(
                self.harness_repository, self.harness_revision, harness_root
            )
            self.roots[role] = harness_root
            subject_root: Path | None = None
            if subject.revision is not None:
                subject_root = self.work_root / role / "torchsky"
                _clone_checkout(self.subject_repository, subject.revision, subject_root)
            for environment in required_environments:
                self._command(
                    [
                        "pixi",
                        "install",
                        "--manifest-path",
                        str(harness_root / "pixi.toml"),
                        "-e",
                        environment,
                    ]
                )
                install_target = subject.wheel or subject_root
                assert install_target is not None
                install = [
                    "pixi",
                    "run",
                    "--manifest-path",
                    str(harness_root / "pixi.toml"),
                    "-e",
                    environment,
                    "--",
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                ]
                if subject.revision is not None:
                    install.append("--no-build-isolation")
                install.append(str(install_target))
                self._command(install)
                self._command(
                    [
                        "pixi",
                        "run",
                        "--manifest-path",
                        str(harness_root / "pixi.toml"),
                        "-e",
                        environment,
                        "--",
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--no-build-isolation",
                        "-e",
                        str(harness_root),
                    ]
                )
            environment_identities = {}
            environment_measurements = {}
            for environment in required_environments:
                identity_result = self._command(
                    [
                        "pixi",
                        "run",
                        "--manifest-path",
                        str(harness_root / "pixi.toml"),
                        "-e",
                        environment,
                        "--",
                        "python",
                        "-m",
                        "torchsky_harness.cli",
                        "subject",
                    ],
                    capture_output=True,
                )
                identity = json.loads(identity_result.stdout)
                self._verify_subject_identity(subject, identity)
                environment_identities[environment] = identity
                measurement_result = self._command(
                    [
                        "pixi",
                        "run",
                        "--manifest-path",
                        str(harness_root / "pixi.toml"),
                        "-e",
                        environment,
                        "--",
                        "python",
                        "-c",
                        (
                            "import json; from torchsky_harness.paired_performance "
                            "import measure_environment; "
                            "print(json.dumps(measure_environment('cpu')))"
                        ),
                    ],
                    capture_output=True,
                )
                measurement = json.loads(measurement_result.stdout)
                if not isinstance(measurement, dict):
                    raise RuntimeError(f"{role} {environment} measurement is invalid")
                environment_measurements[environment] = measurement
            self.environment_identities[role] = environment_identities
            self.environment_measurements[role] = environment_measurements
            self.identities[role] = environment_identities["test"]
        baseline_import = self.identities["baseline"]["import_path"]
        candidate_import = self.identities["candidate"]["import_path"]
        if baseline_import == candidate_import:
            raise RuntimeError(
                "baseline and candidate resolved to the same import path"
            )

    def _verify_subject_identity(
        self, subject: SubjectSpec, identity: Mapping[str, Any]
    ) -> None:
        source = identity.get("source")
        if not isinstance(source, dict):
            raise RuntimeError(f"{subject.role} subject identity has no source")
        if subject.revision is not None:
            if (
                source.get("kind") != "source_tree"
                or source.get("revision") != subject.revision
                or source.get("dirty") is not False
            ):
                raise RuntimeError(
                    f"{subject.role} subject identity does not match its SHA"
                )
        else:
            assert subject.wheel is not None
            if source.get("kind") != "archive" or source.get("sha256") != sha256(
                subject.wheel
            ):
                raise RuntimeError(
                    f"{subject.role} subject identity does not match its wheel"
                )

    def execute(self) -> Path:
        claims_payload = load_claims(
            self.roots["baseline"]
            / "src/torchsky_harness/release_performance_claims.json"
        )
        config_errors = validate_configuration(
            claims_payload, root=self.roots["baseline"]
        )
        if config_errors:
            raise RuntimeError(
                "invalid release performance configuration: " + "; ".join(config_errors)
            )
        claims = claims_payload["claims"]
        tasks = list(dict.fromkeys(claim["task"] for claim in claims))
        reports: dict[tuple[str, str], list[Path]] = {
            (role, claim["row_id"]): [] for role in self.subjects for claim in claims
        }
        clean_env = dict(os.environ)
        clean_env["PYTHONNOUSERSITE"] = "1"
        clean_env.pop("PYTHONPATH", None)
        for pair_index in range(self.pairs):
            pair_id = f"pair-{pair_index:03d}"
            order = (
                ("baseline", "candidate")
                if pair_index % 2 == 0
                else ("candidate", "baseline")
            )
            for task in tasks:
                for role in order:
                    root = self.roots[role]
                    task_claims = [claim for claim in claims if claim["task"] == task]
                    task_artifacts = {claim["task_artifact"] for claim in task_claims}
                    if len(task_artifacts) != 1:
                        raise RuntimeError(
                            f"task {task} has inconsistent task artifacts"
                        )
                    relative_artifact = Path(task_artifacts.pop())
                    source_artifact = root / relative_artifact
                    source_artifact.unlink(missing_ok=True)
                    self._command(
                        [
                            "pixi",
                            "run",
                            "--manifest-path",
                            str(root / "pixi.toml"),
                            task,
                        ],
                        cwd=root,
                        env=clean_env,
                    )
                    if not source_artifact.is_file():
                        raise RuntimeError(
                            f"task {task} did not write {relative_artifact}"
                        )
                    raw_destination = (
                        self.output_root / "reports" / pair_id / role / f"{task}.json"
                    )
                    raw_destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_artifact, raw_destination)
                    event = {
                        "sequence": len(self.execution),
                        "pair_id": pair_id,
                        "task": task,
                        "role": role,
                        "source_report": raw_destination.relative_to(
                            self.output_root
                        ).as_posix(),
                        "source_report_sha256": sha256(raw_destination),
                        "status": "OK",
                    }
                    self.execution.append(event)
                    for claim in task_claims:
                        extracted = (
                            self.output_root
                            / "extracted"
                            / claim["row_id"]
                            / role
                            / f"{pair_id}.json"
                        )
                        extract_claim_report(
                            claim=claim,
                            source_report=raw_destination,
                            extracted_report=extracted,
                            pair_id=pair_id,
                            role=role,
                            artifact_root=self.output_root,
                        )
                        reports[(role, claim["row_id"])].append(extracted)
        return self._build_artifacts(claims_payload, reports)

    def _build_artifacts(
        self,
        claims_payload: Mapping[str, Any],
        reports: Mapping[tuple[str, str], list[Path]],
    ) -> Path:
        generated = json.loads(json.dumps(claims_payload))
        generated_by_id = {claim["row_id"]: claim for claim in generated["claims"]}
        fingerprint_files: list[dict[str, str]] = []
        for claim in claims_payload["claims"]:
            for role, field in (
                ("baseline", "baseline_artifact"),
                ("candidate", "candidate_artifact"),
            ):
                output = (
                    self.output_root
                    / "artifacts"
                    / role
                    / (claim["row_id"].replace(".", "_") + ".json")
                )
                environment = TASK_ENVIRONMENTS[claim["task"]]
                fingerprint = dict(self.environment_measurements[role][environment])
                fingerprint.update(
                    {
                        "dtype": claim["dtype"],
                        "device": claim["device"],
                        "task_command_sha256": claim["task_command_sha256"],
                        "task_arguments": claim["effective_arguments"],
                    }
                )
                fingerprint_path = (
                    self.output_root / "fingerprints" / claim["row_id"] / f"{role}.json"
                )
                fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
                fingerprint_path.write_text(
                    json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8"
                )
                fingerprint_files.append(
                    {
                        "path": fingerprint_path.relative_to(
                            self.output_root
                        ).as_posix(),
                        "sha256": sha256(fingerprint_path),
                    }
                )
                command = [
                    "pixi",
                    "run",
                    "--manifest-path",
                    str(self.roots[role] / "pixi.toml"),
                    "-e",
                    "test",
                    "--",
                    "python",
                    "-m",
                    "torchsky_harness.cli",
                    "performance-artifact",
                    "--claim-id",
                    claim["row_id"],
                    "--fingerprint-json",
                    str(fingerprint_path),
                    "--artifact-root",
                    str(self.output_root),
                    "--json-out",
                    str(output),
                ]
                for report in reports[(role, claim["row_id"])]:
                    command.extend(["--report", str(report)])
                self._command(command, cwd=self.roots[role])
                if not output.is_file():
                    raise RuntimeError(f"artifact builder did not write {output}")
                generated_by_id[claim["row_id"]][field] = output.relative_to(
                    self.output_root
                ).as_posix()
        generated_config = self.output_root / "release_performance_claims.json"
        generated_config.write_text(
            json.dumps(generated, indent=2) + "\n", encoding="utf-8"
        )
        readiness = self.output_root / "release_performance_readiness.json"
        self._command(
            [
                "pixi",
                "run",
                "--manifest-path",
                str(self.roots["candidate"] / "pixi.toml"),
                "-e",
                "test",
                "--",
                "python",
                "-m",
                "torchsky_harness.cli",
                "performance",
                "--config",
                str(generated_config),
                "--root",
                str(self.roots["candidate"]),
                "--artifact-root",
                str(self.output_root),
                "--json-out",
                str(readiness),
            ],
            cwd=self.roots["candidate"],
        )
        manifest = {
            "schema_version": 1,
            "harness_revision": self.harness_revision,
            "subjects": {
                role: {
                    "requested": subject.immutable_identity,
                    "installed": self._portable_identity(self.identities[role]),
                    "environments": {
                        environment: self._portable_identity(identity)
                        for environment, identity in self.environment_identities[
                            role
                        ].items()
                    },
                }
                for role, subject in self.subjects.items()
            },
            "measured_fingerprints": fingerprint_files,
            "pairs": self.pairs,
            "execution": self.execution,
            "generated_config": generated_config.relative_to(
                self.output_root
            ).as_posix(),
            "generated_config_sha256": sha256(generated_config),
            "readiness": readiness.relative_to(self.output_root).as_posix(),
            "readiness_sha256": sha256(readiness),
        }
        manifest_path = self.output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest_path

    @staticmethod
    def _portable_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
        source = identity.get("source")
        portable_source = (
            {
                key: value
                for key, value in source.items()
                if key in {"kind", "revision", "dirty", "sha256"}
            }
            if isinstance(source, dict)
            else None
        )
        return {
            "distribution": identity.get("distribution"),
            "version": identity.get("version"),
            "package_content_sha256": identity.get("package_content_sha256"),
            "source": portable_source,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paired release-performance evidence in isolated subject environments."
    )
    parser.add_argument("--baseline-sha")
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-sha")
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--harness-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument(
        "--harness-repository",
        default="https://github.com/astroai/torchsky-harness.git",
    )
    parser.add_argument(
        "--subject-repository", default="https://github.com/astroai/torchsky.git"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = _subject_spec("baseline", args.baseline_sha, args.baseline_wheel)
    candidate = _subject_spec("candidate", args.candidate_sha, args.candidate_wheel)
    claims = load_claims()["claims"]
    if args.pairs < 5:
        raise ValueError("paired performance evidence requires at least five pairs")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "harness_revision": _require_exact_sha(
                        args.harness_sha, "harness SHA"
                    ),
                    "baseline": baseline.immutable_identity,
                    "candidate": candidate.immutable_identity,
                    "output_root": str(args.output_root.resolve()),
                    "work_root": str(
                        args.work_root.resolve()
                        if args.work_root is not None
                        else args.output_root.resolve().parent
                        / f".{args.output_root.resolve().name}-work"
                    ),
                    "pairs": args.pairs,
                    "task_count": len({claim["task"] for claim in claims}),
                    "claim_count": len(claims),
                    "fingerprint_source": "measured independently in each subject environment",
                },
                indent=2,
            )
        )
        return 0
    runner = PairedPerformanceRunner(
        harness_revision=args.harness_sha,
        baseline=baseline,
        candidate=candidate,
        output_root=args.output_root,
        work_root=args.work_root,
        pairs=args.pairs,
        harness_repository=args.harness_repository,
        subject_repository=args.subject_repository,
    )
    runner.prepare()
    print(runner.execute())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
