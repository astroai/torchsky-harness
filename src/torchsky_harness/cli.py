"""Installed-package entry point; suite arguments follow a -- separator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

from torchsky_harness.registry import domain, evaluations
from torchsky_harness.runner import run
from torchsky_harness.subject import identify_subject, sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--domain", choices=["wcs", "sphere", "map", "catalogs", "cross_domain"])
    commands.add_parser("subject")
    running = commands.add_parser("run")
    running.add_argument("evaluation")
    running.add_argument("--run-dir", type=Path)
    running.add_argument("--subject-wheel", type=Path)
    raw = list(sys.argv[1:] if argv is None else argv)
    split = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:split])
    child_args = raw[split + 1:] if split < len(raw) else []
    if args.command == "list":
        for name in evaluations():
            if args.domain is None or domain(name) == args.domain:
                print(f"{domain(name):12s} {name}")
        return 0
    if args.command == "subject":
        print(json.dumps(identify_subject(), indent=2))
        return 0
    wheel = None
    if args.subject_wheel:
        wheel = args.subject_wheel.resolve(strict=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
        subprocess.run([sys.executable, "-c", "import torchsky; import torchsky.cpp"], check=True)
        print(f"Installed subject artifact {wheel.name} sha256={sha256(wheel)}")
    run_dir = args.run_dir or Path("runs") / uuid.uuid4().hex
    code = run(args.evaluation, child_args, run_dir, subject_artifact=wheel)
    print(f"Execution record: {run_dir.resolve() / 'execution.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
