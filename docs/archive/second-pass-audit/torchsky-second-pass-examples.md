# Torchsky second-pass examples/scripts audit

Audit date: 2026-10-12. Read-only audit of `/Users/fabbros/src/astroai/torchsky` and the sibling `/Users/fabbros/src/astroai/torchsky-harness`. I read the torchsky `AGENTS.md`, `.cursor/harness/playbook.md`, `.cursor/harness/config.json`, and harness `AGENTS.md`. No source/destination files were changed; this report is the only output.

## Harness-owned content already moved

- The core working tree has staged-for-commit deletions of the benchmark tree and campaign scripts/tests; the `benchmarks/` directory is absent from the live checkout. The sibling harness has `benchmarks/`, `scripts/`, and the migrated weekly workflows.
- `map_compat_weekly.yml` and `wcs_sphere_weekly.yml` are present under `torchsky-harness/.github/workflows/`. The core currently retains `ci.yml` and `build_wheels.yml` only.
- The harness already contains `scripts/build_smacs0723_minicutout.py`, which is why the remaining real-data notebook can continue there after its location and references are updated.

## Examples and notebooks

### Strong core/tutorial candidates

These are torchsky API demonstrations, use synthetic or small data, and do not import benchmark comparators:

- Coordinate/map API examples: `example_healpix.py`, `example_spectral_sht.py`, WCS examples (`example_wcs_transform.py`, `example_wcs_extended_projections.py`), image/cube/cutout/dataset examples, table recipes, and Polars/table interop examples.
- The 14 `example_map_*.py` scripts document native map/field/harmonic solver use and are listed in `docs/examples.md`/`docs/map.md`; keep them if they remain supported API tutorials. `example_map_field_survey_split.py` imports the private `torchsky.map.field._BaseFieldModel`, so it is the exception to a clean public-API tutorial surface.
- `example_field_multiband_quality.py` is a short synthetic field-fit walkthrough. Its quality measurements make it a borderline report demo, but its small size and direct `torchsky.field` API usage make it viable as a core tutorial.
- `scripts/gen_radio_mm_fixture_fits.py` is a deterministic, small native integration-fixture generator. It writes through `torchsky.write`/torchfits and should remain core.

### Move or decide explicitly

- All three `examples/notebooks/*.ipynb` are experiments rather than API reference tutorials: synthetic single-band fitting, synthetic hyperspectral fitting/export, and a real SMACS workflow. The real notebook says its wavelength metadata “matches benchmark smoke specs.” These fit the stated harness ownership and should move together.
- `examples/catalogue_photoz.py` is a 681-line pinned-catalogue photometric-redshift pilot/training pipeline, not a small API tutorial. It has a core test (`tests/test_catalogue_photoz_example.py`) and user guide (`docs/photometry-encoders.md`), so either keep it deliberately as a supported application example with matching optional dependencies, or move it with its test/docs as an experiment. Do not move only the source.
- `example_field_multiband_quality.py` and the advanced map solver examples are borderline only if “report demos/experiments” is intended to include all synthetic end-to-end applications. Current docs make them explicit product examples and they contain no comparator or benchmark imports.
- `example_image.py` only prints a message; the actual `torchsky.read` calls are commented out and `torchsky` is not imported. It is an orphan placeholder rather than a working tutorial.

### FITS/API violations in retained tutorials/helpers

These are concrete policy debt under the current root `AGENTS.md` (“Never import `astropy.io.fits`” and use torchfits at file boundaries):

- `examples/example_image_mef.py` imports `astropy.io.fits` and builds its MEF fixture with Astropy; it also uses Astropy Table. Convert the fixture to torchfits or replace the example with a supported torchfits API flow.
- `examples/example_table_interop.py` calls Astropy `Table.write(..., format="fits")`. This should use `torchfits.table` for FITS and demonstrate interop in memory if the point is Astropy conversion.
- `scripts/apply_wcs_to_mef.py` imports `astropy.io.fits` for production MEF header edits.
- `scripts/astrometric_calibrate.py` imports `astropy.io.fits` and is coupled to `apply_wcs_to_mef.py`. These were deliberately categorized as production calibration helpers, so preserve their purpose but migrate file operations to torchfits before treating them as compliant.
- `scripts/tweakreg_align_exposures.py` correctly uses torchfits for header reads, but its module docstring still names the now-moved `benchmarks/bench_map_smacs0723_real.py`. The harness's UPDC README also calls `scripts/tweakreg_align_exposures.py`; the helper is not in the harness checkout. It is strongly campaign-specific (JWST SMACS naming/output contract), so move it with the SMACS harness workflow or deliberately retain and document it as a production calibration CLI.

## Script inventory/classification

- Keep core maintenance: `scripts/install_hooks.py`, `scripts/verify_setup.py`, `scripts/audit_refactor_cohesion.py` (also wired to a Pixi task).
- Keep small fixture utility: `scripts/gen_radio_mm_fixture_fits.py`.
- Keep production calibration purpose, with the FITS migration above: `scripts/apply_wcs_to_mef.py`, `scripts/astrometric_calibrate.py`.
- Campaign/helper candidate for harness: `scripts/tweakreg_align_exposures.py` (see stale path above).
- `scripts/autonomous_refactor_status.py` still points at the autonomous-refactor launch branch and `docs/reviews/autonomous_refactor_log.md`; that log exists, so this is stale workflow tooling only if the project has ended that launch, not a broken path today.

`examples/test_examples.py` is a separate subprocess smoke runner and is not part of pytest collection (`pyproject.toml` uses `testpaths = ["tests"]`). It is not invoked by the retained CI or Pixi tasks. It selects `example_image.py`, `example_table.py`, `example_wcs_transform.py`, and `example_table_interop.py`; the latter requires Astropy and carries the FITS-writing violation above. Decide whether to wire this intentionally or remove it as an orphan runner.

## Broken/stale references after extraction

- `examples/notebooks/03_smacs0723_real_minicutout.ipynb` tells users to run `scripts/build_smacs0723_minicutout.py`; that path no longer exists in torchsky. The helper exists at the same relative path in torchsky-harness, so update the notebook/docs when moving it.
- `docs/examples.md`, `docs/field.md`, `docs/photometry-encoders.md`, `docs/map.md`, and `docs/projects/hyperspectral-fusion.md` still link to or give commands for notebooks/scripts under core `examples/` or `scripts/`. Repoint links to the harness for moved experiments while retaining links for true core tutorials.
- The harness migration inventory itself notes the moved SMACS mini-cutout builder also backs a tutorial and says to decide whether to move it or keep a thin core entry point. Current harness contains the full builder; no thin delegating entry point remains in core.
- The notebook Pixi task remains in core (`pixi.toml:81`, `jupyter lab`), but Jupyter is not installed by the default environment or declared by the `examples` extra. If notebooks move to harness, remove/move this task or wire a real notebook environment there.
- The core `.gitignore` retains many now-orphaned `benchmarks/`, `benchmarks/baselines/`, and `benchmarks/scorecards/` rules and scorecard allow-lists; the live `benchmarks/` directory is absent. These can be reviewed after the extraction commit.

## CI/Pixi/package wiring

- Runtime packaging is already lean and aligned with policy: `pyproject.toml` runtime dependencies are torch, numpy, torchfits; Pixi production feature adds torchfits; no harness-private runtime import was found in remaining examples/scripts.
- Core Pixi has no benchmark/comparator environment after the edit: only `default`, `test`, `cuda-bench` (native package tooling), and `astrometry` remain. The harness owns its comparator and evaluation environments.
- Core CI is package CI. Weekly comparator workflows are in harness. The retained `map-mosaic-bench` CI job name is misleading: it runs bounded native map tests only; consider renaming it to `map-contracts`. The `sphere-spectral` job is likewise a native contract suite, not an upstream comparator benchmark.
- `pyproject.toml` `[project.optional-dependencies].examples` is out of sync with examples: it contains `torchvision` (“MNIST example”) and `requests` (“SDSS download example”) that no remaining `.py` example imports; it contains matplotlib used only by notebooks likely moving to harness; it omits Polars and PyArrow needed by retained `example_polars.py` and `catalogue_photoz.py`, and Astropy required by `example_table_interop.py`/`example_image_mef.py`.
- The test extra and Pixi test feature still include Astropy, PyArrow, Polars and related packages for retained optional interop/parity/API tests; this is consistent with the repository policy. No bench/sphere-comparator extras remain in pyproject.
- `tests/test_perf_gate.py` still carries the `performance` marker and is retained core hot-path smoke coverage, consistent with the policy. `integration` is also still used.

## Ignored local data and folders (inventory only; do not delete)

Sizes are current `du` output, rounded by the filesystem. These are excluded from Git and were not opened or changed.

- `.pixi/` — 15 GB environment/cache.
- `.benchmarks/` — 544 MB benchmark-local state.
- `benchmarks_results/` — 389 MB results; `benchmark_results/` — 968 KB.
- `build/` — 88 MB; `dist/` — 2.1 MB; `extern/cfitsio/` — 14 MB generated vendor dependency (core build input; preserve).
- `.opencode/node_modules/` — 61 MB; `.cursor/harness/` — 932 KB; `.dsh/` — 32 KB; `.jules/` — 20 KB; `.freebuff/` — 4 KB (untracked, not ignored; contents not inspected).
- `.pixi-cache/` — 0 B; `.pytest_cache/` — 292 KB; `.ruff_cache/` — 368 KB; `.venv/` — 76 KB; `scratch/` — 8 KB.
- `examples/data_simple_fits/` — 200 KB and `examples/data_wcs_examples/` — 88 KB. These are ignored generated outputs: their examples create the files with torchfits, so they are not source fixtures that need migration.
- Root ignored FITS data: `coadd.weight.fits` (1.4 GB), `mosaic_g_test.fits` (1.8 GB), `coadd.fits` (8.4 KB), `tmp.fits` and `tmp_w.fits` (262 KB each), plus `tmp_0..4.fits` and `tmp_0_w..4_w.fits` (4 MB each). Preserve and identify ownership before any relocation; total is about 3.25 GB.
- Ignored generated example FITS include `examples/cube_example.fits`, `cutout_example.fits`, `mef_example.fits`, and `table_example.fits`. The corresponding examples create them locally.
- `data` at repository root is a symlink to `/Users/fabbros/src/torchsky/data` outside this checkout; preserve the symlink/target and clarify ownership before moving it.
- Other ignored state includes `.DS_Store`, LaTeX build products in `docs/`, `astrometry.cfg`, native `__pycache__` trees, and test-generated FITS. `tests/data/radio_mm_stub.fits` is the deliberate small fixture exception and is written by `gen_radio_mm_fixture_fits.py`.
- Untracked tracked-by-human-looking items in this checkout include `docs/plans/`, `docs/projects/smacs_inr_autonomous_bakeoff.md`, `docs/roadmap/ecosystem_parity_and_uv_ir_plan.md`, and `.freebuff/`; preserve as unrelated until ownership is decided. The SMACS bakeoff note appears harness/evaluation-owned by name, but it was not read as part of this inventory.

## Verification

Inventory only; no tests or source edits were run. The report was written to `/private/tmp/torchsky-second-pass-examples.md`.
