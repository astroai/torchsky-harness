# Extraction status

Torchsky-harness now owns the existing WCS, sphere, map/SMACS, and catalog
evaluation code. Benchmark module names and numerical acceptance files were
preserved under the installed `torchsky_harness.benchmarks` namespace. No PSF,
detection, segmentation or SSL placeholders were created. Library kernels,
compiled builds and bounded package tests remain in torchsky.

The extraction removed 145 benchmark/evidence files, 55 evaluator test modules,
and 28 evaluation scripts from the original tree, and split 20 mixed test
modules. Original byte hashes and function ownership are recorded in
`migration-files.json`, `migration-tests.json`, and `migration-scripts.json`.
The existing uncommitted library/model changes were preserved and not published
as part of this migration. The tested subject wheel includes that local state;
its SHA256 is in `source-snapshot.json`.

## Verification

- Torchsky preflight (Ruff and compileall): passed.
- Harness preflight and pure-Python wheel build: passed.
- Core collection: 2,132 tests collected without import errors after extraction.
- Harness collection: 982 tests collected without import errors before the final
  small fixture/metadata corrections; no full test or benchmark suite was run.
- Core boundary/docs/catalog/PCG tests: 13 passed, 1 skipped.
- Focused geometry core: 47 passed, 7 skipped; harness counterparts: 6 passed,
  18 skipped; cylindrical core contract: 1 passed; cylindrical oracle cases:
  8 passed. Optional oracle skips remain visible and do not certify acceptance.
- Map/depth/runner/preparation/scorer selection: 120 cases accounted for,
  117 passed and 3 skipped after repairing extraction errors. The obsolete
  quicklook metadata assertion also failed in the original tree; the migrated
  test checks the same shapes through the current metadata schema.
- Additional targeted SMACS/helper tests and catalog/pipeline smoke are recorded
  in the three domain verification notes. Runner infrastructure: 6 passed.
- Both wheels installed into the harness's clean locked environment and
  evaluated from `/private/tmp`: WCS, map, catalog, and native sphere probes
  passed. The required-healpy probe failed visibly because the lean test profile
  lacks healpy; use `sphere-bench` for that acceptance contract.

The wheel probes initially inherited an editable subject import hook through a
system-site-packages virtual environment. The import-path assertion caught the
wrong subject. Verification then used the separately installed harness Pixi
environment and confirmed both imports resolve to installed wheel packages.

Torchfits 1.0.0 cannot write a data-less TensorHDU through its public HDUList
writer. Converted MEF fixtures therefore use a zero-length primary tensor,
preserving an empty primary and the original science/weight/mask arrays; its
structural header has NAXIS1=0. SCI extension names and keyed HDU access were
checked explicitly. No astronomy I/O implementation was added to the harness.

## Execution and CI

CANFAR runners select separate harness and torchsky revisions or an explicit
subject wheel. Frozen existing jobs and artifact locations were left untouched.
The historical campaign ledger is a snapshot, not live status.

Infrastructure CI runs on PRs. Scientific workflows are manual until a
`SUBJECT_READ_TOKEN` with access to the private torchsky repository is configured;
the dispatcher must select an exact subject revision. Their former scheduled
triggers are deliberately inactive during the cross-repository transition.
Required comparator gates and their tolerances remain in the harness; they were
not recomputed by the small migration smoke tests.

Historical FITS cold-read/GPU-loading and isolated pipeline-read rows are
torchfits-owned evidence. They were preserved with that ownership rather than
being removed or described as torchsky performance.

## Local handoff

The local repository is on codex/extract-scientific-evaluation. Torchsky migration edits remain uncommitted to preserve the pre-existing development changes. Verification wheels and logs are preserved under the ignored dist/verification directory. No remote repository or PR was created: automatic approval review rejected publishing the internal code and historical campaign payload without explicit payload authorization.
