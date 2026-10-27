"""Installed-package entry point; suite arguments follow a -- separator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

from torchsky_harness.evidence import build_bundle, write_bundle
from torchsky_harness.release_performance import (
    assess_claims,
    build_artifact,
    load_claims,
)
from torchsky_harness.registry import domain, evaluations
from torchsky_harness.runner import run
from torchsky_harness.subject import identify_subject, sha256


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in {"calibration-manifest", "calibration-extract"}:
        # Calibration commands are the only harness CLI surface that imports
        # the installed subject. Keep list/subject/run usable without it.
        from torchsky_harness.calibration_command import (
            register_calibration_parser,
        )

        parser = argparse.ArgumentParser(prog="torchsky-harness")
        commands = parser.add_subparsers(dest="command", required=True)
        register_calibration_parser(commands)
        args = parser.parse_args(raw)
        return args.handler(args)

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument(
        "--domain", choices=["wcs", "sphere", "map", "catalogs", "cross_domain"]
    )
    commands.add_parser("subject")
    bundle = commands.add_parser(
        "evidence", help="Collect run records into a release evidence bundle."
    )
    bundle.add_argument("--runs-dir", type=Path, default=Path("runs"))
    bundle.add_argument("--json-out", type=Path, default=None)
    bundle.add_argument("--markdown-out", type=Path, default=None)
    bundle.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero unless every declared claim has an established run",
    )
    bundle.add_argument("--performance-config", type=Path, default=None)
    bundle.add_argument("--root", type=Path, default=Path("."))
    bundle.add_argument("--performance-artifact-root", type=Path, default=None)
    performance = commands.add_parser("performance")
    performance.add_argument("--config", type=Path, default=None)
    performance.add_argument("--root", type=Path, default=Path("."))
    performance.add_argument("--artifact-root", type=Path, default=None)
    performance.add_argument("--json-out", type=Path, default=None)
    artifact = commands.add_parser("performance-artifact")
    artifact.add_argument("--config", type=Path, default=None)
    artifact.add_argument("--claim-id", required=True)
    artifact.add_argument("--report", type=Path, action="append", required=True)
    artifact.add_argument("--fingerprint-json", type=Path, required=True)
    artifact.add_argument("--json-out", type=Path, required=True)
    artifact.add_argument("--artifact-root", type=Path, required=True)
    commands.add_parser(
        "calibration-manifest",
        help="Build a split-safe JWST/NIRCam calibration-evidence manifest.",
    )
    commands.add_parser(
        "calibration-extract",
        help="Extract PSF/background targets from JWST/NIRCam detector files.",
    )
    running = commands.add_parser("run")
    running.add_argument("evaluation")
    running.add_argument("--run-dir", type=Path)
    running.add_argument("--subject-wheel", type=Path)
    split = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:split])
    child_args = raw[split + 1 :] if split < len(raw) else []
    if args.command == "list":
        for name in evaluations():
            if args.domain is None or domain(name) == args.domain:
                print(f"{domain(name):12s} {name}")
        return 0
    if args.command == "subject":
        print(json.dumps(identify_subject(), indent=2))
        return 0
    if args.command == "evidence":
        run_dirs = (
            sorted(path for path in args.runs_dir.iterdir() if path.is_dir())
            if args.runs_dir.is_dir()
            else []
        )
        evidence_bundle = build_bundle(run_dirs)
        performance_payload = (
            load_claims(args.performance_config)
            if args.performance_config
            else load_claims()
        )
        performance_result = assess_claims(
            performance_payload,
            root=args.root.resolve(),
            artifact_root=(
                args.performance_artifact_root.resolve()
                if args.performance_artifact_root is not None
                else None
            ),
        )
        evidence_bundle["scientific_release_ready"] = evidence_bundle["release_ready"]
        evidence_bundle["performance"] = performance_result
        evidence_bundle["release_ready"] = (
            evidence_bundle["scientific_release_ready"]
            and performance_result["release_performance_ready"]
        )
        print(
            write_bundle(
                evidence_bundle,
                json_out=args.json_out,
                markdown_out=args.markdown_out,
            )
        )
        if args.fail_on_missing and not evidence_bundle["release_ready"]:
            return 1
        return 0
    if args.command == "performance":
        performance_payload = load_claims(args.config) if args.config else load_claims()
        result = assess_claims(
            performance_payload,
            root=args.root.resolve(),
            artifact_root=(
                args.artifact_root.resolve() if args.artifact_root is not None else None
            ),
        )
        rendered = json.dumps(result, indent=2)
        print(rendered)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        return 0 if result["release_performance_ready"] else 1
    if args.command == "performance-artifact":
        performance_payload = load_claims(args.config) if args.config else load_claims()
        fingerprint = json.loads(args.fingerprint_json.read_text(encoding="utf-8"))
        if not isinstance(fingerprint, dict):
            raise ValueError("fingerprint JSON must be an object")
        result = build_artifact(
            performance_payload,
            claim_id=args.claim_id,
            reports=args.report,
            fingerprint=fingerprint,
            artifact_root=args.artifact_root.resolve(),
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0
    wheel = None
    if args.subject_wheel:
        wheel = args.subject_wheel.resolve(strict=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-c", "import torchsky; import torchsky._cpp"], check=True
        )
        print(f"Installed subject artifact {wheel.name} sha256={sha256(wheel)}")
    run_dir = args.run_dir or Path("runs") / uuid.uuid4().hex
    code = run(args.evaluation, child_args, run_dir, subject_artifact=wheel)
    print(f"Execution record: {run_dir.resolve() / 'execution.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
