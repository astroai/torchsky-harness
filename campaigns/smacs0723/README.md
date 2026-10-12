# SMACS 0723 evaluation

`history/20261011/` preserves the job specifications, ledger and compact reports
available at extraction. They are historical snapshots, not live session status.
`historical-artifacts.json` records original local paths and byte hashes. Frozen
job specifications retain their original remote source/calibration identities.

Existing remote fits/scorers keep running against their frozen sources. Their
large products and source archives remain at the existing CANFAR paths; the
original ignored local results are retained as well. No jobs were submitted or
retired for extraction. Fetch an existing artifact using the installed Pixi
environment and `fetch_frozen_artifact.py <run-id> <relative-artifact-path>`.

New evaluation jobs use the harness runners, with separate
`TORCHSKY_HARNESS_COMMIT` and `TORCHSKY_COMMIT` pins, plus either
`TORCHSKY_SUBJECT_ROOT` or `TORCHSKY_WHEEL`. The harness must be installed in each
selected environment. Do not reuse old frozen specifications for a new source.

The five priorities remain SNR/recovery, resolution, reconstruction, artifacts,
and flux conservation. Compare paired cohorts and units. Published UPDC products
support static image diagnostics; they do not provide pipeline injection metrics.
Historical scorecards include unavailable and unfinished measurements and must
not be treated as a completed all-method acceptance matrix.
