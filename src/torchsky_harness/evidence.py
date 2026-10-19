"""Declared evaluation claims and the verdicts an execution record may carry.

An execution record never guesses scientific validity. A run can only be
``established`` for a claim the harness has declared here, and only when the
declared required rows are present and passing. Evaluations without a declared
claim stay ``not_assessed``; the record says why.

Claim scopes:

``synthetic_truth``
    The evaluation compares against an injected or analytic truth on synthetic
    data. It says nothing about real detector data, real sky, or other regimes.
``reference_comparison``
    The evaluation compares the subject against an independent reference
    implementation on the declared cases.
``real_data``
    The evaluation uses real observations with an explicit, checked reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from torchsky_harness.subject import sha256

ClaimScope = Literal["synthetic_truth", "reference_comparison", "real_data"]


@dataclass(frozen=True)
class EvaluationClaim:
    """The bounded claim a release gate is allowed to certify."""

    scope: ClaimScope
    required_rows: tuple[str, ...]
    comparators: tuple[str, ...] = ()
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "required_rows": list(self.required_rows),
            "comparators": list(self.comparators),
            "limitations": list(self.limitations),
        }


CLAIMS: dict[str, EvaluationClaim] = {
    "bench_map_cube": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("cube_constant_bands",),
        limitations=(
            "synthetic 2D-band cube assembly only; not a 3D IFU or detector cube",
            "no dense spectral interpolation claim",
        ),
    ),
    "bench_map_ifu": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("ifu_matched_sparse_spectrum",),
        limitations=(
            "matched spectral sampling only; dense interpolation is not claimed",
            "synthetic wavelength-map IFU, not real IFU data",
        ),
    ),
    "bench_map_healpix_multiband": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=(
            "healpix_multiband_flat_multiscale",
            "healpix_multiband_sed_registration",
            "healpix_multiband_plane_photometry",
            "healpix_multiband_roundtrip_cube",
        ),
        limitations=(
            "coadd_healpix samples image values at HEALPix centres; flux-conserving"
            " reprojection is not claimed",
            "no variance/weight product, harmonic/SHT resampling, or healpy"
            " throughput claim",
        ),
    ),
    "bench_native_response": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("native_gaussian_star",),
        limitations=(
            "linear pixel-integrated Gaussian profile only",
            "not a full likelihood, joint, or real-detector claim",
        ),
    ),
    "bench_native_likelihood": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("native_gaussian_likelihood",),
        limitations=(
            "diagonal native Gaussian likelihood on a synthetic stamp",
            "evidence-staleness rejection is contract behaviour, not calibration",
        ),
    ),
    "bench_native_nuisance": EvaluationClaim(
        scope="synthetic_truth",
        required_rows=("native_nuisance_precision",),
        limitations=(
            "profiled/calibrated nuisance precision against dense references",
            "no real-detector nuisance calibration claim",
        ),
    ),
}


def _argument(command: Sequence[str], flag: str) -> str | None:
    values = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == flag
    ]
    return values[-1] if values else None


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], f"evidence report unreadable: {exc}"
    if not isinstance(payload, list):
        return [], "evidence report is not a row list"
    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        return [], "evidence report has no rows"
    return rows, None


def verdict(
    name: str,
    *,
    returncode: int | None,
    command: Sequence[str],
    working_directory: Path,
) -> dict[str, Any]:
    """Derive the only verdict fields an execution record may carry."""
    claim = CLAIMS.get(name)
    if claim is None:
        return {
            "scientific_validity": "not_assessed",
            "comparison_eligibility": "not_assessed",
            "verdict_reasons": ["no declared claim for this evaluation"],
        }

    reasons: list[str] = []
    comparison_notes: list[str] = []
    record: dict[str, Any] = {
        "claim": claim.as_record(),
    }
    if returncode != 0:
        reasons.append(f"evaluation exited with return code {returncode}")

    report_argument = _argument(command, "--json-out")
    if report_argument is None:
        reasons.append("evaluation declared no --json-out evidence report")
    else:
        report_path = Path(report_argument)
        if not report_path.is_absolute():
            report_path = working_directory / report_path
        rows, error = _load_rows(report_path)
        if error is not None:
            reasons.append(error)
        else:
            by_case = {str(row.get("case")): row for row in rows}
            for required in claim.required_rows:
                row = by_case.get(required)
                if row is None:
                    reasons.append(f"missing required row {required}")
                elif str(row.get("status", "OK")) != "OK":
                    reasons.append(
                        f"required row {required} reported status {row.get('status')}"
                    )
            record["evidence_report"] = str(report_path.resolve())

    comparison = "suite_owned"
    baseline_argument = _argument(command, "--baseline-json")
    if baseline_argument is not None:
        if "--require-complete-baseline" in command:
            comparison = "baseline_compared"
        else:
            comparison_notes.append(
                "baseline supplied without --require-complete-baseline, so"
                " baseline coverage is not enforced"
            )
    record["scientific_validity"] = "established" if not reasons else "not_established"
    record["comparison_eligibility"] = comparison
    record["verdict_reasons"] = reasons
    record["comparison_notes"] = comparison_notes
    return record


BUNDLE_SCHEMA_VERSION = 1


def _read_record(run_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(
            (run_dir / "execution.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _digest(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return sha256(path)


def _run_entry(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    report_value = record.get("evidence_report")
    report_path = Path(report_value) if isinstance(report_value, str) else None
    subject = record.get("subject") if isinstance(record.get("subject"), dict) else {}
    source = subject.get("source") if isinstance(subject.get("source"), dict) else {}
    logs = record.get("logs") if isinstance(record.get("logs"), list) else []
    entry = {
        "run_dir": str(run_dir.resolve()),
        "evaluation": record.get("evaluation"),
        "execution": record.get("execution"),
        "returncode": record.get("returncode"),
        "scientific_validity": record.get("scientific_validity", "not_assessed"),
        "comparison_eligibility": record.get("comparison_eligibility", "not_assessed"),
        "verdict_reasons": record.get("verdict_reasons", []),
        "claim": record.get("claim"),
        "subject": {
            "distribution": subject.get("distribution"),
            "version": subject.get("version"),
            "import_path": subject.get("import_path"),
            "package_content_sha256": subject.get("package_content_sha256"),
            "revision": source.get("revision"),
            "dirty": source.get("dirty"),
        },
        "harness": record.get("harness"),
        "execution_record": {
            "path": str((run_dir / "execution.json").resolve()),
            "sha256": _digest(run_dir / "execution.json"),
        },
        "evidence_report": {
            "path": str(report_path.resolve()) if report_path else None,
            "sha256": _digest(report_path),
        },
        "logs": [
            {
                "path": str(item.get("path")),
                "sha256": _digest(run_dir / str(item.get("path")))
                if isinstance(item, dict) and item.get("path")
                else None,
            }
            for item in logs
            if isinstance(item, dict)
        ],
    }
    return entry


def build_bundle(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Collect run records into a release evidence bundle.

    For each declared claim the newest `established` run is selected. Runs that
    cannot be read are reported, never silently dropped.
    """
    entries: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for run_dir in run_dirs:
        record = _read_record(run_dir)
        if record is None:
            unreadable.append(str(run_dir))
            continue
        entries.append(_run_entry(run_dir, record))

    by_evaluation: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_evaluation.setdefault(str(entry.get("evaluation")), []).append(entry)

    selected: dict[str, dict[str, Any]] = {}
    superseded: dict[str, list[str]] = {}
    for evaluation, evaluation_entries in by_evaluation.items():
        established = [
            entry
            for entry in evaluation_entries
            if entry.get("scientific_validity") == "established"
        ]
        pool = established or evaluation_entries
        chosen = max(
            pool,
            key=lambda entry: (
                Path(entry["execution_record"]["path"]).stat().st_mtime
                if Path(entry["execution_record"]["path"]).is_file()
                else 0.0
            ),
        )
        selected[evaluation] = chosen
        others = [
            entry["run_dir"] for entry in pool if entry["run_dir"] != chosen["run_dir"]
        ]
        if others:
            superseded[evaluation] = sorted(others)

    declared = sorted(CLAIMS)
    missing_claims = [
        evaluation
        for evaluation in declared
        if selected.get(evaluation, {}).get("scientific_validity") != "established"
    ]
    extra_established = sorted(
        evaluation
        for evaluation, entry in selected.items()
        if evaluation not in CLAIMS
        and entry.get("scientific_validity") == "established"
    )
    subjects = {
        (
            entry["subject"]["package_content_sha256"],
            entry["subject"]["revision"],
        )
        for entry in selected.values()
    }
    harness_revisions = {
        (entry.get("harness") or {}).get("source", {}).get("revision")
        for entry in selected.values()
    }
    incomplete = sorted(
        evaluation
        for evaluation, entry in selected.items()
        if entry.get("execution") != "completed"
    )
    counts: dict[str, int] = {"established": 0, "not_established": 0, "not_assessed": 0}
    for entry in selected.values():
        key = str(entry.get("scientific_validity", "not_assessed"))
        counts[key] = counts.get(key, 0) + 1
    release_ready = (
        not missing_claims
        and not incomplete
        and len(subjects) <= 1
        and not unreadable
    )
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release_ready": release_ready,
        "declared_claims": declared,
        "missing_claims": missing_claims,
        "extra_established": extra_established,
        "subject_identities": [
            {"package_content_sha256": content, "revision": revision}
            for content, revision in sorted(subjects, key=str)
        ],
        "subject_consistent": len(subjects) <= 1,
        "harness_revisions": sorted(str(item) for item in harness_revisions),
        "incomplete_runs": incomplete,
        "unreadable_runs": unreadable,
        "superseded_runs": superseded,
        "verdict_counts": counts,
        "entries": selected,
    }


def render_bundle_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# Evidence bundle",
        "",
        f"- release ready: **{bundle['release_ready']}**",
        f"- generated: {bundle['generated_at']}",
        f"- subject consistent: {bundle['subject_consistent']}",
        f"- verdicts: {bundle['verdict_counts']}",
    ]
    if bundle["missing_claims"]:
        lines.append(f"- missing claims: {', '.join(bundle['missing_claims'])}")
    if bundle["incomplete_runs"]:
        lines.append(f"- incomplete runs: {', '.join(bundle['incomplete_runs'])}")
    if bundle["unreadable_runs"]:
        lines.append(f"- unreadable run dirs: {', '.join(bundle['unreadable_runs'])}")
    lines += [
        "",
        "| evaluation | validity | comparison | subject | run |",
        "|---|---|---|---|---|",
    ]
    for evaluation in sorted(bundle["entries"]):
        entry = bundle["entries"][evaluation]
        subject = entry["subject"]
        identity = subject.get("package_content_sha256") or "n/a"
        lines.append(
            f"| {evaluation} | {entry['scientific_validity']} |"
            f" {entry['comparison_eligibility']} | {str(identity)[:12]} |"
            f" {entry['run_dir']} |"
        )
    return "\n".join(lines) + "\n"


def write_bundle(
    bundle: dict[str, Any],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> str:
    markdown = render_bundle_markdown(bundle)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    return markdown
