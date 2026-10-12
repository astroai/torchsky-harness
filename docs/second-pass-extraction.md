# Second extraction pass, 2026-10-12

Scientific comparison code is owned by this repository. The second pass moved
map reconstruction, flux, frontier, uncertainty, acceptance, and benchmark-row
reporting functions into `torchsky_harness.metrics`. Detector evidence discovery
and extraction, and the `calibration-manifest` / `calibration-extract` commands,
also moved here. The core retains its validated calibration loader, uncertainty
estimator, renderers, geometry, solver diagnostics, and bounded runtime tests.

The three experiment notebooks and the TweakReg campaign helper and its tests
are now here. Core examples continue to demonstrate native APIs. Their remaining
FITS operations use public torchfits interfaces. A legacy FITS-read timing gate
is retained here as a facade boundary measurement; FITS performance ownership
belongs to torchfits.

Campaign ledgers, comparison protocols, measured performance history, and the
scientific agent runbooks were extracted from core documentation. Runtime API
guides, mathematical product formulation, and coding-agent memory stay in core.
Complete prior versions of mixed documentation are preserved in
[`archive/core-before-second-pass`](archive/core-before-second-pass). Those
snapshots retain their original paths and commands and are historical records.
The duplicate SMACS bakeoff document was consolidated into the existing
[`smacs_inr_autonomous_bakeoff.md`](smacs_inr_autonomous_bakeoff.md).

[`second-pass-code.json`](second-pass-code.json) records 28 verified function
placements, original AST hashes, the core path changes, and tested wheel hashes.
The moved calculations and tolerance defaults were preserved. Notebook/helper
file hashes are in [`../examples-migration-manifest.md`](../examples-migration-manifest.md).

## Verification

- Core Ruff and compileall preflight passed.
- Harness Ruff and compileall passed; generated build copies are excluded from lint.
- Core typecheck: zero errors and one existing `__all__` warning.
- Core API, I/O-boundary, and calibration-loader checks: 14 passed.
- Real SIP and synthetic TPV multi-extension image checks: 4 passed.
- Calibration helper and retained catalog application checks: 13 passed.
- Installed harness metrics, calibration, boundary, and runner checks against
  the earlier installed subject: 20 passed.
- Current core and harness wheels, installed in an independent Pixi environment:
  map product, field diagnostics, API, loader, metric, and extraction checks:
  38 passed, 3 expected skips. Import paths were verified inside site-packages.
- Relocated TweakReg guards: 5 passed.
- Retained image-MEF and table-interoperability examples ran successfully against
  the installed subject. CLI listing does not import the subject; it lists 48 map
  evaluations. Core active Markdown relative links were checked.

These are extraction and runtime checks, not a rerun of the complete scientific
comparison campaign or a new SNR/resolution/depth claim. No new remote jobs were
submitted by this pass.

## Local cleanup

The local ignored archive `.local-artifacts/torchsky-before-cleanup-20261012`
preserves 4,473,814,387 bytes of prior results, FITS products, external tools,
build artifacts, and scratch scripts. Its local manifest records original paths,
destinations, sizes, and SHA-256 values. These large artifacts are excluded from
Git. Historical control files and session IDs were preserved; new launches must
use the current harness runners.

Eight unused materialized Pixi environments, an obsolete virtual environment,
generated node dependencies, and caches were removed from core. The default and
test environments remain available, and supported optional environments can be
installed again from the manifest. The broken data symlink was removed after
recording its target. Existing library working-tree changes remain preserved;
the local subject wheel records that working state rather than a clean commit.
