# Extraction status

`main` owns scientific evaluation, reference adapters, campaign reports, and
CANFAR evaluation orchestration for the `torchsky` subject. The core package
retains production implementations and bounded unit, API, regression,
adjoint, gradient, and device-contract tests. FITS/table parsing and I/O gates
belong to `torchfits`.

The initial extraction preserved the benchmark modules, acceptance files,
campaign records, and migrated tests under the installed
`torchsky_harness` namespace. File and test mappings are recorded in
[`migration-files.json`](migration-files.json),
[`migration-tests.json`](migration-tests.json), and
[`migration-scripts.json`](migration-scripts.json). Existing project-specific
verification notes are snapshots of checks made during the extraction; they
are not claims that every scientific gate has been rerun on the current
subject. See the [geometry](geometry-migration-verification.md),
[catalog](catalog-migration-verification.md), and
[map](map-migration-verification.md) notes for their scopes and limits.

No PSF, detection, segmentation, or SSL placeholder suites were created.
Moved FITS fixtures and scripts use `torchfits`; the harness does not contain a
second FITS implementation.

## Revision identity

Every evaluation records the `torchsky-harness` revision and the selected
`torchsky` subject revision separately. The harness SHA identifies evaluation
code; `TORCHSKY_COMMIT` or the subject wheel SHA identifies the implementation
under test. Do not infer one revision from the other or treat an editable
source checkout as an installed subject without recording its identity.

## Execution and CI

Infrastructure checks can run on ordinary CI. Scientific workflows require
an explicitly selected subject revision and the required comparator tools;
missing required comparators remain visible as missing coverage. Remote
campaigns require their own task authorization. Existing campaign records and
their artifacts are historical evidence, not live job status.

FITS cold-read and isolated FITS-loading measurements are owned by
`torchfits`. They are not torchsky FITS-throughput claims.
