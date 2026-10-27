import json
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from torchsky_harness.paired_performance import (
    PairedPerformanceRunner,
    SubjectSpec,
    TASK_ENVIRONMENTS,
    extract_claim_report,
    main,
)
from torchsky_harness.cli import main as cli_main
from torchsky_harness.release_performance import assess_claims
from torchsky_harness.release_performance import load_claims


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
OTHER_SHA = "b" * 40


def _raw_rows(task: str) -> list[dict]:
    claims = [claim for claim in load_claims()["claims"] if claim["task"] == task]
    rows: list[dict] = []
    for claim in claims:
        row = next(
            (
                existing
                for existing in rows
                if all(
                    existing.get(key) == value
                    for key, value in claim["row_selector"].items()
                )
            ),
            None,
        )
        if row is None:
            row = dict(claim["row_selector"])
            rows.append(row)
        row[claim["timing_field"]] = 10.0
    return rows


def test_extract_claim_report_preserves_status_and_source_hash(tmp_path: Path) -> None:
    claim = load_claims()["claims"][0]
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([{**claim["row_selector"], claim["timing_field"]: 2.5}]),
        encoding="utf-8",
    )
    destination = tmp_path / "extracted.json"
    result = extract_claim_report(
        claim=claim,
        source_report=source,
        extracted_report=destination,
        pair_id="pair-000",
        role="baseline",
        artifact_root=tmp_path,
    )
    assert result["status"] == "OK"
    assert result["correctness_status"] == "OK"
    assert result["rows"][0]["status"] == "OK"
    assert len(result["source_report"]["sha256"]) == 64


@pytest.mark.parametrize("status", ["FAILED", "SKIPPED"])
def test_extract_claim_report_fails_closed_on_bad_status(
    tmp_path: Path, status: str
) -> None:
    claim = load_claims()["claims"][0]
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    **claim["row_selector"],
                    claim["timing_field"]: 2.5,
                    "status": status,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selected row status"):
        extract_claim_report(
            claim=claim,
            source_report=source,
            extracted_report=tmp_path / "extracted.json",
            pair_id="pair-000",
            role="candidate",
            artifact_root=tmp_path,
        )


def test_runner_executes_complete_pairs_in_alternating_order(tmp_path: Path) -> None:
    output = tmp_path / "output"
    calls: list[tuple[str, str]] = []
    roots = {
        "baseline": tmp_path / "baseline",
        "candidate": tmp_path / "candidate",
    }
    for root in roots.values():
        (root / "src/torchsky_harness").mkdir(parents=True)
        shutil.copy2(
            ROOT / "src/torchsky_harness/release_performance_claims.json",
            root / "src/torchsky_harness/release_performance_claims.json",
        )
        shutil.copy2(ROOT / "pixi.toml", root / "pixi.toml")
        acceptance = root / "benchmarks/baselines/map_drizzle_acceptance.json"
        acceptance.parent.mkdir(parents=True)
        shutil.copy2(
            ROOT / "benchmarks/baselines/map_drizzle_acceptance.json", acceptance
        )

    def fake_command(
        command: list[str],
        *,
        cwd: Path | None = None,
        env: object = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd is not None
        task = command[-1]
        role = "baseline" if cwd == roots["baseline"] else "candidate"
        calls.append((task, role))
        claim = next(item for item in load_claims()["claims"] if item["task"] == task)
        artifact = cwd / claim["task_artifact"]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(_raw_rows(task)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    runner = PairedPerformanceRunner(
        harness_revision=SHA,
        baseline=SubjectSpec("baseline", SHA, None),
        candidate=SubjectSpec("candidate", OTHER_SHA, None),
        output_root=output,
        work_root=tmp_path / "work",
        pairs=5,
        harness_repository="unused",
        subject_repository="unused",
        command_runner=fake_command,
    )
    output.mkdir()
    runner.roots = roots
    expected = output / "done.json"
    runner._build_artifacts = lambda claims_payload, reports: expected
    assert runner.execute() == expected
    tasks = list(dict.fromkeys(claim["task"] for claim in load_claims()["claims"]))
    for pair_index in range(5):
        expected_order = (
            ["baseline", "candidate"]
            if pair_index % 2 == 0
            else ["candidate", "baseline"]
        )
        offset = pair_index * len(tasks) * 2
        for task_index, task in enumerate(tasks):
            assert calls[offset + task_index * 2 : offset + task_index * 2 + 2] == [
                (task, expected_order[0]),
                (task, expected_order[1]),
            ]
    assert len(runner.execution) == 5 * len(tasks) * 2


def test_dry_run_accepts_wheels_without_creating_output(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.whl"
    candidate = tmp_path / "candidate.whl"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    output = tmp_path / "output"
    assert (
        main(
            [
                "--baseline-wheel",
                str(baseline),
                "--candidate-wheel",
                str(candidate),
                "--harness-sha",
                SHA,
                "--output-root",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output.exists()


def test_runner_rejects_incomplete_pair_count(tmp_path: Path) -> None:
    wheel = tmp_path / "subject.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(ValueError, match="at least five pairs"):
        main(
            [
                "--baseline-wheel",
                str(wheel),
                "--candidate-sha",
                SHA,
                "--harness-sha",
                OTHER_SHA,
                "--output-root",
                str(tmp_path / "output"),
                "--pairs",
                "4",
                "--dry-run",
            ]
        )


def test_generated_evidence_tree_is_relocatable(tmp_path: Path) -> None:
    output = tmp_path / "scratch-output"
    output.mkdir()
    claims_payload = load_claims()
    reports: dict[tuple[str, str], list[Path]] = {
        (role, claim["row_id"]): []
        for role in ("baseline", "candidate")
        for claim in claims_payload["claims"]
    }
    for pair_index in range(5):
        pair_id = f"pair-{pair_index:03d}"
        for role in ("baseline", "candidate"):
            for task in dict.fromkeys(
                claim["task"] for claim in claims_payload["claims"]
            ):
                raw = output / "reports" / pair_id / role / f"{task}.json"
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_text(json.dumps(_raw_rows(task)), encoding="utf-8")
                for claim in (
                    item for item in claims_payload["claims"] if item["task"] == task
                ):
                    extracted = (
                        output
                        / "extracted"
                        / claim["row_id"]
                        / role
                        / f"{pair_id}.json"
                    )
                    extract_claim_report(
                        claim=claim,
                        source_report=raw,
                        extracted_report=extracted,
                        pair_id=pair_id,
                        role=role,
                        artifact_root=output,
                    )
                    reports[(role, claim["row_id"])].append(extracted)

    def fake_command(
        command: list[str],
        *,
        cwd: Path | None = None,
        env: object = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        cli_index = next(
            index
            for index, value in enumerate(command)
            if value in {"performance-artifact", "performance"}
        )
        assert cli_main(command[cli_index:]) == 0
        return subprocess.CompletedProcess(command, 0, "", "")

    runner = PairedPerformanceRunner(
        harness_revision=SHA,
        baseline=SubjectSpec("baseline", SHA, None),
        candidate=SubjectSpec("candidate", OTHER_SHA, None),
        output_root=output,
        work_root=tmp_path / "work",
        pairs=5,
        harness_repository="unused",
        subject_repository="unused",
        command_runner=fake_command,
    )
    runner.roots = {"baseline": ROOT, "candidate": ROOT}
    runner.identities = {
        role: {
            "distribution": "torchsky",
            "version": "0.4.0",
            "package_content_sha256": role * 8,
            "source": {
                "kind": "source_tree",
                "revision": SHA if role == "baseline" else OTHER_SHA,
                "dirty": False,
            },
        }
        for role in ("baseline", "candidate")
    }
    runner.environment_identities = {
        role: {
            environment: runner.identities[role]
            for environment in set(TASK_ENVIRONMENTS.values()) | {"test"}
        }
        for role in ("baseline", "candidate")
    }
    runner.environment_measurements = {
        role: {
            environment: {
                "runner": "measured-node",
                "cpu": "measured-cpu",
                "os": "measured-os",
                "python": "3.13.14",
                "torch": "2.13.0",
                "torchsky_version": "0.3.0" if role == "baseline" else "0.4.0",
                "torchsky_revision": SHA if role == "baseline" else OTHER_SHA,
                "device": "cpu",
            }
            for environment in set(TASK_ENVIRONMENTS.values()) | {"test"}
        }
        for role in ("baseline", "candidate")
    }
    manifest_path = runner._build_artifacts(claims_payload, reports)
    assert manifest_path == output / "manifest.json"
    relocated = tmp_path / "durable"
    shutil.copytree(output, relocated)
    manifest = json.loads((relocated / "manifest.json").read_text(encoding="utf-8"))
    for field in ("generated_config", "readiness"):
        assert not Path(manifest[field]).is_absolute()
        assert (relocated / manifest[field]).is_file()
    assert (
        hashlib.sha256(
            (relocated / manifest["generated_config"]).read_bytes()
        ).hexdigest()
        == manifest["generated_config_sha256"]
    )
    assert (
        hashlib.sha256((relocated / manifest["readiness"]).read_bytes()).hexdigest()
        == manifest["readiness_sha256"]
    )
    for fingerprint in manifest["measured_fingerprints"]:
        fingerprint_path = relocated / fingerprint["path"]
        assert fingerprint_path.is_file()
        assert (
            hashlib.sha256(fingerprint_path.read_bytes()).hexdigest()
            == fingerprint["sha256"]
        )
    for extracted in relocated.glob("extracted/**/*.json"):
        envelope = json.loads(extracted.read_text(encoding="utf-8"))
        source = envelope["source_report"]["path"]
        assert not Path(source).is_absolute()
        assert (relocated / source).is_file()
    config = json.loads(
        (relocated / manifest["generated_config"]).read_text(encoding="utf-8")
    )
    assert assess_claims(config, root=ROOT, artifact_root=relocated)[
        "release_performance_ready"
    ]
