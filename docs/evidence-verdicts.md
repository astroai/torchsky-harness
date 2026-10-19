# Execution records and evidence verdicts

Every `torchsky-harness run` writes `<run-dir>/execution.json`. The record is the
only place the harness states what a run did and what it is allowed to claim.
`stdout.log`, `stderr.log`, and any `--json-out` report sit beside it.

## Record fields

| Field | Meaning |
|---|---|
| `schema_version` | record layout version (currently `1`) |
| `evaluation`, `command`, `working_directory` | what ran, exactly |
| `subject` | installed subject distribution, version, import path, package-content SHA-256, VCS revision/dirty state, Python, platform. The subject is never a checkout on `sys.path`. |
| `harness` | harness version and source identity |
| `inputs` | SHA-256 of every command argument that is an existing file at launch |
| `execution` | `running`, `completed`, `failed`, or `interrupted` |
| `returncode` | child process status once finished |
| `logs`, `evidence_report` | SHA-256 of the logs; absolute path of the `--json-out` report |
| `scientific_validity` | `not_assessed`, `established`, or `not_established` |
| `comparison_eligibility` | `not_assessed`, `suite_owned`, or `baseline_compared` |
| `verdict_reasons` | why validity is not `established` (empty when established) |
| `comparison_notes` | baseline supplied but not enforced, or similar |
| `claim` | the declared bounded claim this evaluation may certify (absent when none) |
| `metrics`, `device` | `suite_owned`: rows carry their own metrics/device declaration |

## Verdict vocabulary

A run never guesses. Validity is derived only from declarations in
`src/torchsky_harness/evidence.py` plus observable facts:

* `not_assessed` — the evaluation has no declared claim (most evaluations), or
  the run did not finish.
* `established` — the evaluation has a declared claim, the child exited `0`, and
  every declared required row is present in the evidence report with
  `status=OK`.
* `not_established` — a declared claim exists but one of the above failed; each
  failure is listed in `verdict_reasons`.

`comparison_eligibility` is independent of scientific validity:

* `suite_owned` — the evaluation reports its own metrics; no baseline coverage
  was enforced.
* `baseline_compared` — a `--baseline-json` was supplied together with
  `--require-complete-baseline`, so missing rows, missing metrics, and
  regressions are failures.

## Claim scopes

| Scope | What a pass certifies |
|---|---|
| `synthetic_truth` | the declared synthetic/analytic cases only; says nothing about real detector data, real sky, or other regimes |
| `reference_comparison` | parity against the named independent comparators on the declared cases |
| `real_data` | real observations with an explicit, checked reference |

Each declared claim also lists `limitations`. Those limitations are part of the
claim: for example the HEALPix multi-band gate does not claim flux-conserving
reprojection, harmonic resampling, variance propagation, or comparator
throughput.

## Declared release gates

| Evaluation | Rows | Task |
|---|---|---|
| `bench_map_cube` | `cube_constant_bands` | `bench-map-cube-gate` |
| `bench_map_ifu` | `ifu_matched_sparse_spectrum` | `bench-map-ifu-gate` |
| `bench_map_healpix_multiband` | `healpix_multiband_flat_multiscale`, `healpix_multiband_sed_registration`, `healpix_multiband_plane_photometry`, `healpix_multiband_roundtrip_cube` | `bench-map-healpix-multiband-gate` |
| `bench_native_response` | `native_gaussian_star` | `bench-native-response-gate` |
| `bench_native_likelihood` | `native_gaussian_likelihood` | `bench-native-likelihood-gate` |
| `bench_native_nuisance` | `native_nuisance_precision` | `bench-native-nuisance-gate` |

Adding an evaluation to `CLAIMS` is a deliberate act: it widens what a passing
run may claim. Do not add one without a committed baseline, a completeness
check, and a limitations list.

## Reading a bundle

For each gate run keep: `execution.json` (including `subject.source.revision`
and `package_content_sha256`), the `--json-out` report, `stdout.log`, and the
task command. A release bundle is the set of runs whose `scientific_validity` is
`established` for a declared claim and whose subject identity matches the release
artifact. Anything else is supporting evidence, not a pass.
