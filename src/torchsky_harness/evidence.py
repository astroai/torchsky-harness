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
import json
from pathlib import Path
from typing import Any, Literal, Sequence

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
