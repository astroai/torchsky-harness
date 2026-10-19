"""Run existing suites with durable, atomic execution provenance."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from torchsky_harness import __version__
from torchsky_harness.benchmarks.config import benchmark_subprocess_env
from torchsky_harness.evidence import verdict
from torchsky_harness.registry import resolve
from torchsky_harness.subject import git_identity, identify_subject, sha256


def write_record(path: Path, record: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def run(name: str, args: list[str], run_dir: Path, subject_artifact: Path | None = None) -> int:
    module = resolve(name)
    subject = identify_subject()
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", module, *args]
    inputs = []
    for value in args:
        path = Path(value)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
    if subject_artifact is not None:
        subject["selected_artifact"] = {"path": str(subject_artifact), "sha256": sha256(subject_artifact)}
    record = {
        "schema_version": 1, "evaluation": name, "command": command,
        "working_directory": str(Path.cwd()), "subject": subject,
        "harness": {"version": __version__, "source": git_identity(Path(__file__).parent)},
        "inputs": inputs, "execution": "running", "returncode": None,
        "scientific_validity": "not_assessed", "comparison_eligibility": "not_assessed",
        "verdict_reasons": ["execution not finished"],
        "metrics": "suite_owned", "device": "suite_owned",
    }
    record_path = run_dir / "execution.json"
    write_record(record_path, record)
    try:
        with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
            env = benchmark_subprocess_env()
            env["TORCHSKY_HARNESS_OUTPUT_ROOT"] = str((run_dir / "outputs").resolve())
            completed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False)
        code = completed.returncode
    except BaseException as exc:
        record.update(execution="interrupted", error=repr(exc))
        record["verdict_reasons"] = ["execution interrupted"]
        write_record(record_path, record)
        raise
    record.update(execution="completed" if code == 0 else "failed", returncode=code)
    record.update(
        verdict(
            name,
            returncode=code,
            command=command,
            working_directory=Path.cwd(),
        )
    )
    record["logs"] = [{"path": name, "sha256": sha256(run_dir / name)} for name in ("stdout.log", "stderr.log")]
    write_record(record_path, record)
    return code
