from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import torchsky_harness.cli as cli_module
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
    subject: dict | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    report = run_dir / "report.json"
    claim = CLAIMS.get(evaluation)
    if claim is not None and claim.summary_failures_key is not None:
        payload: object = {claim.summary_failures_key: []}
    elif claim is not None:
        rows = [
            {"case": required, "status": "OK"} for required in claim.required_rows
        ]
        rows.extend(
            {**dict(selector), "status": "OK"}
            for selector in claim.required_row_selectors
        )
        payload = rows
    else:
        payload = [{"case": "row", "status": "OK"}]
    report.write_text(json.dumps(payload), encoding="utf-8")
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
        "subject": subject if subject is not None else SUBJECT,
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
    assert bundle["invalid_reports"] == {}
    for evaluation in CLAIMS:
        entry = bundle["entries"][evaluation]
        assert entry["evidence_report"]["sha256"] is not None
        assert entry["evidence_report"]["revalidated"] is True
        assert entry["evidence_report"]["validation_errors"] == []
        assert entry["logs"][0]["sha256"] is not None


def test_deleted_declared_claim_report_blocks_release_readiness(tmp_path: Path) -> None:
    runs = _all_claims(tmp_path)
    report = runs[0] / "report.json"
    report.unlink()

    bundle = build_bundle(runs)

    evaluation = sorted(CLAIMS)[0]
    assert bundle["release_ready"] is False
    assert "SHA-256 unavailable" in " ".join(bundle["invalid_reports"][evaluation])
    assert bundle["entries"][evaluation]["evidence_report"]["sha256"] is None


def test_missing_declared_claim_report_path_blocks_release_readiness(
    tmp_path: Path,
) -> None:
    runs = _all_claims(tmp_path)
    record_path = runs[0] / "execution.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop("evidence_report")
    record_path.write_text(json.dumps(record), encoding="utf-8")

    bundle = build_bundle(runs)

    evaluation = sorted(CLAIMS)[0]
    assert bundle["release_ready"] is False
    assert bundle["invalid_reports"][evaluation] == ["evidence report path missing"]


def test_corrupted_declared_claim_report_blocks_release_readiness(tmp_path: Path) -> None:
    runs = _all_claims(tmp_path)
    (runs[0] / "report.json").write_text("{not json", encoding="utf-8")

    bundle = build_bundle(runs)

    evaluation = sorted(CLAIMS)[0]
    assert bundle["release_ready"] is False
    assert bundle["entries"][evaluation]["evidence_report"]["sha256"] is not None
    assert any(
        "evidence report unreadable" in reason
        for reason in bundle["invalid_reports"][evaluation]
    )


def test_undeclared_runs_at_another_subject_do_not_block_readiness(
    tmp_path: Path,
) -> None:
    runs = _all_claims(tmp_path)
    other = dict(SUBJECT)
    other["package_content_sha256"] = "d" * 64
    supporting = _run_dir(tmp_path, "run-support", "bench_map_compat", subject=other)
    bundle = build_bundle([*runs, supporting])

    assert bundle["release_ready"] is True
    assert bundle["subject_consistent"] is True
    assert bundle["claim_subject_identities"] == [("a" * 64, None, None, "b" * 40)]
    assert len(bundle["subject_identities"]) == 2


def test_a_declared_claim_at_another_subject_blocks_readiness(tmp_path: Path) -> None:
    # Every declared claim at SUBJECT except bench_wcs, which is only present at
    # another subject (no same-evaluation run, so selection never depends on mtime).
    others = [
        _run_dir(tmp_path, f"run-{index}", evaluation)
        for index, evaluation in enumerate(sorted(CLAIMS))
        if evaluation != "bench_wcs"
    ]
    other = dict(SUBJECT)
    other["package_content_sha256"] = "d" * 64
    replaced = _run_dir(tmp_path, "run-other", "bench_wcs", subject=other)
    bundle = build_bundle([*others, replaced])

    assert bundle["release_ready"] is False
    assert bundle["subject_consistent"] is False


def test_bundle_reports_missing_claim(tmp_path: Path) -> None:
    runs = _all_claims(tmp_path)[:-1]
    bundle = build_bundle(runs)

    assert bundle["release_ready"] is False
    assert bundle["missing_claims"] == [sorted(CLAIMS)[-1]]


def test_undeclared_failed_run_does_not_block_release_readiness(tmp_path: Path) -> None:
    runs = _all_claims(tmp_path)
    extra = _run_dir(
        tmp_path,
        "run-extra",
        "bench_map_compat",
        validity="not_assessed",
        execution="failed",
    )
    bundle = build_bundle([*runs, extra])

    assert bundle["release_ready"] is True
    assert bundle["incomplete_runs"] == ["bench_map_compat"]
    assert bundle["incomplete_claims"] == []


def test_incomplete_declared_claim_blocks_release_readiness(tmp_path: Path) -> None:
    missing = sorted(CLAIMS)[-1]
    runs = _all_claims(tmp_path)[:-1]
    failed = _run_dir(
        tmp_path,
        "run-failed",
        missing,
        validity="not_established",
        execution="failed",
    )
    bundle = build_bundle([*runs, failed])

    assert bundle["release_ready"] is False
    assert bundle["incomplete_claims"] == [missing]


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
    record_path = other / "execution.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    # The newest run per evaluation wins; coarse filesystem mtimes (scratch
    # volumes) can tie with the original bench_wcs run, so order them explicitly.
    newest = max(path.stat().st_mtime for path in tmp_path.rglob("execution.json")) + 10.0
    os.utime(record_path, (newest, newest))
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


def test_cli_evidence_fails_with_complete_science_but_missing_performance(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for path in _all_claims(runs_dir):
        assert path.is_dir()

    output = tmp_path / "bundle.json"
    assert (
        main(
            [
                "evidence",
                "--runs-dir",
                str(runs_dir),
                "--json-out",
                str(output),
                "--fail-on-missing",
            ]
        )
        == 1
    )
    bundle = json.loads(output.read_text(encoding="utf-8"))
    assert bundle["scientific_release_ready"] is True
    assert bundle["performance"]["release_performance_ready"] is False
    assert bundle["release_ready"] is False


def test_cli_evidence_passes_only_with_science_and_performance_ready(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for path in _all_claims(runs_dir):
        assert path.is_dir()
    monkeypatch.setattr(
        cli_module,
        "assess_claims",
        lambda *_args, **_kwargs: {
            "release_performance_ready": True,
            "errors": [],
            "claims": [],
        },
    )
    assert main(["evidence", "--runs-dir", str(runs_dir), "--fail-on-missing"]) == 0


@pytest.mark.parametrize("evaluation", sorted(CLAIMS))
def test_claim_records_scope_and_limitations(evaluation: str) -> None:
    record = CLAIMS[evaluation].as_record()
    assert record["scope"] in {
        "synthetic_truth",
        "reference_comparison",
        "real_data",
        "contract",
    }
    assert record["required_rows"] or record["summary_failures_key"]
    assert record["limitations"]
