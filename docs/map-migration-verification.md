# Map and SMACS migration verification

This note records the focused checks run while moving map, depth, field, and
SMACS evaluation into `torchsky-harness`. The stage uses the torchsky Pixi test
environment with the harness editable-installed; no CANFAR job or full suite
was launched.

FITS fixture and campaign I/O now use `torchfits` public APIs. Synthetic MEFs
are built with `HDUList` and `TensorHDU`; image reads use `open`/`read`; single
image writes use `write`. Astropy remains available to parity code for WCS,
but no migrated map/SMACS module imports `astropy.io.fits`. Campaign tests and
scripts import through `torchsky_harness.benchmarks` and
`torchsky_harness.scripts`; the copied map/SMACS tests no longer insert
checkout paths into `sys.path`.

The migrated SMACS sampling test originally imported fixture helpers from the
core-only `test_field_artifacts.py` and `test_map_field_streamed_tiles.py`.
Those small deterministic fixtures now live in `tests/_map_field_fixtures.py`
beside the harness evaluation tests. The implementation tests remain in
torchsky and were not moved.

Focused checks completed:

```text
pixi run --frozen -e test -- pytest /private/tmp/torchsky-harness-20261011/tests/test_smacs_sampling.py
20 passed in 17.63s

pixi run --frozen -e test -- pytest \
  /private/tmp/torchsky-harness-20261011/tests/test_smacs_product_visualization.py \
  /private/tmp/torchsky-harness-20261011/tests/test_score_smacs0723_foam.py \
  /private/tmp/torchsky-harness-20261011/tests/test_score_smacs0723_native_classical.py \
  /private/tmp/torchsky-harness-20261011/tests/test_prepare_updc_inputs.py \
  /private/tmp/torchsky-harness-20261011/tests/test_download_updec_webb.py
18 passed in 13.71s

pixi run --frozen -e test -- pytest /private/tmp/torchsky-harness-20261011/tests/test_score_field_models.py
7 passed in 1.59s

pixi run --frozen -e test -- pytest \
  /private/tmp/torchsky-harness-20261011/tests/test_download_jwst_smacs0723_mast.py \
  /private/tmp/torchsky-harness-20261011/tests/test_download_smacs0723_multiband_public.py \
  /private/tmp/torchsky-harness-20261011/tests/test_bootstrap_map_compat_tools.py \
  /private/tmp/torchsky-harness-20261011/tests/test_compare_smacs_grid_mapping.py \
  /private/tmp/torchsky-harness-20261011/tests/test_svo_passband_bootstrap.py \
  /private/tmp/torchsky-harness-20261011/tests/test_smacs0723_full_mosaic_pipeline.py \
  /private/tmp/torchsky-harness-20261011/tests/test_stage_smacs0723_canfar.py
25 passed in 1.71s
```

The first combined run before adding the harness-local fixture module reported
31 passed and 7 failed. Every failure was a missing import of one of the two
core-only test helper modules; the isolated 20-test sampling rerun above passes
after that dependency was removed. The five other modules were rerun and pass
as shown.

Ruff also passed for the edited staged map/SMACS Python files using the core
Pixi default environment. A targeted static scan found no `sys.path` path
injections, dynamic file loaders, or Astropy FITS imports in the migrated
map/SMACS files. The only `astropy.io` text left in those tests is a negative
assertion guarding the torchfits-only campaign path.

The root separately verified its six-file map set: the initial 120-case run
had 110 passed, 3 skipped, and 7 failed; focused reruns repaired all 7. The
last failure was also present in the original core test: it expected the stale
`summary["baseline_shape"]` field, while current visualization output records
per-band shapes in `run_metadata.json`. The migrated assertion now checks the
two baseline products and confirms each is 16×16, preserving the science
assertion against the current metadata contract. No visualization algorithm
changed.

The torchfits fixture conversion preserves the science arrays. Empty primary
HDU fixtures use supported zero-length tensors (so their FITS `NAXIS1` is 0),
and converted science image HDUs retain `EXTNAME=SCI`. The minicutout reader
selects a named `SCI` HDU by keyed lookup and falls back to the primary HDU;
`HDUList` membership does not support string-key checks. These are fixture and
I/O API compatibility details, not benchmark numerical changes.
