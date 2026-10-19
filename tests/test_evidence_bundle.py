from __future__ import annotations

import json
from pathlib import Path

import pytest

from torchsky_harness.cli import main
from torchsky_harness.evidence import (
    CLAIMS,
    build_bundle,
    render_bundle_markdown,
    write_bundle,
)

SUBJECT = {
    "distribution": "torchsky",
    "version": "0.4.0",
    "import_path": "/installed/torchsky/__init__.py",
    "package_content_sha256": "a" * 64,
    "source": {"revision": "b" * 40, "dirty": False},
}
HARNESS = {"version": "0.1.0", "source": {"revision": "c" * 40}}


def _run_dir(
    root: Path,
    name: str,
    evaluation: str,
    *,
    validity: str = "established",
    execution: str = "completed",
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    report = run_dir / "report.json"
    report.write_text(json.dumps([{"case": "row", "status": "OK"}]), encoding="utf-8")
    (run_dir / "stdout.log").write_text("ok\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    record = {
        "schema_version": 1,
        "evaluation": evaluation,
        "execution": execution,
        "returncode": 0,
        "scientific_validity": validity,
        "comparison_eligibility": "baseline_compared",
        "verdict_reasons": [],
        "subject": SUBJECT,
        "harness": HARNESS,
        "evidence_report": str(report),
        "logs": [
            {"path": "stdout.log", "sha256": "0" * 64},
            {"path": "stderr.log", "sha256": "0" * 64},
        ],
    }
    (run_dir / "execution.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def _all_claims(root: Path) -> list[Path]:
    return [
        _run_dir(root, f"run-{index}", evaluation)
        for index, evaluation in enumerate(sorted(CLAIMS))
    ]


def test_bundle_is_release_ready_when_every_claim_is_established(tmp_path: Path) -> None:
    bundle = build_bundle(_all_claims(tmp_path))

    assert bundle["release_ready"] is True
    assert bundle["missing_claims"] == []
    assert bundle["verdict_counts"]["established"] == len(CLAIMS)
    assert bundle["subject_consistent"] is True
    assert bundle["unreadable_runs"] == []
    entry = bundle["entries"]["bench_map_ifu"]
    assert entry["evidence_report"]["sha256"] is not None
    assert entry["logs"][0]["sha256"] is not None


def test_bundle_reports_missing_claim(tmp_path: Path) -> None:
    runs = _all_claims(tmp_path)[:-1]
    bundle = build_bundle(runs)

    assert bundle["release_ready"] is False
    assert bundle["missing_claims"] == [sorted(CLAIMS)[-1]]


def test_bundle_reports_unreadable_run_directory(tmp_path: Path) -> None:
    runs = _all_claims(tmp_path)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "execution.json").write_text("{not json", encoding="utf-8")
    bundle = build_bundle([*runs, tmp_path / "broken"])

    assert bundle["release_ready"] is False
    assert bundle["unreadable_runs"] == [str(tmp_path / "broken")]


def test_bundle_flags_inconsistent_subjects(tmp_path: Path) -> None:
    runs = _all_claims(tmp_path)
    other = _run_dir(tmp_path, "run-other", "bench_wcs")
    record = json.loads((other / "execution.json").read_text(encoding="utf-8"))
    record["subject"] = {**SUBJECT, "package_content_sha256": "d" * 64}
    (other / "execution.json").write_text(json.dumps(record), encoding="utf-8")
    bundle = build_bundle([*runs, other])

    assert bundle["subject_consistent"] is False
    assert bundle["release_ready"] is False


def test_bundle_prefers_newest_established_run(tmp_path: Path) -> None:
    older = _run_dir(tmp_path, "run-older", "bench_map_ifu")
    newer = _run_dir(tmp_path, "run-newer", "bench_map_ifu", validity="not_established")
    bundle = build_bundle([older, newer])

    selected = bundle["entries"]["bench_map_ifu"]
    assert selected["run_dir"] == str(older.resolve())
    assert bundle["superseded_runs"] == {}


def test_write_bundle_emits_json_and_markdown(tmp_path: Path) -> None:
    bundle = build_bundle(_all_claims(tmp_path))
    json_out = tmp_path / "out" / "bundle.json"
    markdown_out = tmp_path / "out" / "bundle.md"
    markdown = write_bundle(bundle, json_out=json_out, markdown_out=markdown_out)

    assert json.loads(json_out.read_text(encoding="utf-8"))["release_ready"] is True
    assert markdown_out.read_text(encoding="utf-8") == markdown
    assert "release ready: **True**" in markdown
    assert render_bundle_markdown(bundle) == markdown


def test_cli_evidence_fails_on_missing_claim(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for index, evaluation in enumerate(sorted(CLAIMS)[:-1]):
        _run_dir(runs_dir, f"run-{index}", evaluation)
    json_out = tmp_path / "bundle.json"

    assert main(["evidence", "--runs-dir", str(runs_dir), "--json-out", str(json_out)]) == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))["release_ready"] is False
    assert (
        main(
            [
                "evidence",
                "--runs-dir",
                str(runs_dir),
                "--fail-on-missing",
            ]
        )
        == 1
    )


def test_cli_evidence_passes_with_complete_bundle(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for path in _all_claims(runs_dir):
        assert path.is_dir()

    assert (
        main(["evidence", "--runs-dir", str(runs_dir), "--fail-on-missing"]) == 0
    )


@pytest.mark.parametrize("evaluation", sorted(CLAIMS))
def test_claim_records_scope_and_limitations(evaluation: str) -> None:
    record = CLAIMS[evaluation].as_record()
    assert record["scope"] == "synthetic_truth"
    assert record["required_rows"]
    assert record["limitations"]
