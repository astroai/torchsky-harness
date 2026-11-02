# Map, SMACS, depth, and field extraction inventory

Inventory date: 2026-10-11. The initial inventory was read-only and inspected `/Users/fabbros/src/astroai/torchsky`; the core checkout remains untouched by this staged extraction. Its worktree is heavily dirty. In particular, the following relevant files already have local changes and must be preserved: `benchmarks/bench_map_smacs0723_field_fit_joint.py`, `benchmarks/bench_multiband_depth.py`, `benchmarks/score_field_models.py`, `docs/map.md`, `tests/test_map_smacs0723_field_fit_joint.py`, `tests/test_multiband_depth_benchmark.py`, `tests/test_score_field_models.py`; the 2026-10-06 SMACS validation scorecard and other untracked project notes are also present. The stage was copied from the working tree, not a clean revision.

## Move to the harness

All of these are scientific evaluation, performance, comparator, calibration, scorecard, campaign, or data-preparation surfaces. Keep their implementation dependencies on public torchsky APIs; do not promote benchmark helpers to torchsky public API.

- `benchmarks/bench_map_*.py` (including compat, ground-truth, image-plane, multifield, reducers, overlap/background, projection/TAN-SIP, Foam, and all `bench_map_smacs0723_*` files); also `benchmarks/bench_operator_multiexposure.py`, `benchmarks/bench_inr_perf.py`, `benchmarks/bench_strainer_phase2.py`, and `benchmarks/bench_map_smacs0723_strainer_transfer.py`.
- Field/map diagnostics and evaluations: `benchmarks/diagnose_field_fit_*.py`, `benchmarks/diagnose_map_*.py`, `benchmarks/run_ground_truth_competitive_large_union.py`, and `benchmarks/bench_multiband_depth.py`.
- SMACS/UPDC scoring and calibration: `benchmarks/bench_updec_webb_real.py`, `benchmarks/compare_smacs_grid_mapping.py`, `benchmarks/fit_smacs0723_operator_calibration.py`, `benchmarks/export_smacs0723_calibration_inspection.py`, `benchmarks/score_field_models.py`, `benchmarks/score_smacs0723_foam.py`, `benchmarks/score_smacs0723_native_classical.py`, `benchmarks/stage_smacs0723_canfar.py`, and all of `benchmarks/updc/`.
- Move the map subset of `benchmarks/baselines/`: all `map_*` files (compat, drizzle, field-fit, ground-truth, multifield, radio, SWarp, Montage). Move map/SMACS records from `benchmarks/scorecards/`, including `20260831_map_overlap_background.*`, `20260928_smacs_grid_mapping_validation.*`, and `20261006_smacs_linear_grid_solver_validation.*`. The scorecard index/archive tooling is shared evaluation infrastructure and belongs with the harness.
- Campaign/evaluation scripts: `scripts/bootstrap_map_compat_tools.py`, `scripts/bench_cfht_mef_mosaic.py`, `scripts/download_jwst_smacs0723_mast.py`, `scripts/download_smacs0723_multiband_public.py`, `scripts/download_updec_webb.py`, `scripts/build_updc_manifest_flexible.py`, `scripts/build_smacs0723_stack_baseline.py`, `scripts/smacs0723_full_mosaic_pipeline.py`, `scripts/smacs0723_field_fit_full_reference.py`, `scripts/visualize_smacs0723_products.py`, and `scripts/run_canfar_adass_depth.sh`. `scripts/build_smacs0723_minicutout.py` is campaign data staging but also backs a tutorial; decide whether to move it or retain a thin example entry point that delegates to the harness.
- Move scientific campaign/design documents `docs/adass_updec_depth_week.md`, `docs/smacs_inr_autonomous_bakeoff.md`, `docs/wcs-aware-smacs0723-pipeline.md`, and `docs/inr_foam_problem_formulation.md`; move `benchmarks/updc/README.md` with its code. Move benchmark methodology/results from `docs/benchmarks.md` and map/SMACS evidence in roadmap pages, then leave short links in torchsky docs. Keep `docs/map.md`, `docs/field.md`, `docs/examples.md`, and roadmap history as package/maintainer docs, editing their claims and commands rather than relocating whole files.

`scripts/export_stpsf_kernels.py` is a boundary case: it produces nominal PSF inputs consumed by the torchsky field API. It can remain as a supported data-preparation utility if it is documented as an opt-in producer, while PSF response/recovery evaluations move with the harness. No empty PSF, detection, segmentation, or SSL suite directories should be scaffolded; the current depth experiment already includes detection and effective-PSF measurements.

## Companion tests and split points

Move the evaluation-only modules with their benchmark code: `test_map_compat_bench.py`, `test_map_external_tools_smoke.py`, `test_map_dfreproject_parity.py`, `test_map_drizzle_external_parity.py`, `test_map_conservative_external_parity.py`, `test_map_swarp_optional_parity.py`, `test_map_megapipe_scale.py`, `test_map_ground_truth.py`, `test_map_multifield_truth.py`, `test_map_reducer_scorecard.py`, `test_map_field_fit.py`, `test_map_smacs0723_field_fit.py`, `test_map_smacs0723_field_fit_band_sweep.py`, `test_map_smacs0723_field_fit_joint.py`, `test_map_smacs0723_real.py`, `test_map_smacs0723_reproduce.py`, `test_map_smacs0723_stackers.py`, `test_map_smacs0723_stackers_band_sweep.py`, `test_multiband_depth_benchmark.py`, `test_inr_perf.py`, `test_score_field_models.py`, `test_score_smacs0723_foam.py`, `test_score_smacs0723_native_classical.py`, `test_smacs_calibration_inspection.py`, `test_smacs_foam_flux_recovery.py`, `test_smacs_measured_adapter.py`, `test_smacs_operator_calibration_fit.py`, `test_smacs_sampling.py`, `test_stage_smacs0723_canfar.py`, `test_updec_webb_real.py`, `test_prepare_updc_inputs.py`, `test_compare_catalogs_qa.py`, and `test_build_updc_manifest_flexible.py`. `test_map_smacs0723_field_fit_joint.py` is a large mixed experiment-contract suite; before deletion, retain focused mathematical/API regressions (deterministic initialization, invalid held-out/mixed-unit rejection, gradient flow, footprint validation, exact native-pixel refinement) only where they test a stable torchsky contract rather than a private benchmark protocol.

Keep package correctness tests that exercise torchsky directly: the core `test_map_image_plane.py` suite, `test_map_adjoints.py`, operator execution/chunking/noise/bundle tests, deterministic geometry and map-kernel tests, and field model/render/serialization contract tests. Split two known mixed files: remove the `test_coadd_image_plane_clamps_negative_resampled_weights_for_high_order_kernels` synthetic-ground-truth benchmark case from `test_map_image_plane.py` into the harness; and split the three benchmark-private helper tests in `test_inr_pcg.py` from its torchsky PCG mathematical tests. `test_map_image_plane.py` has one test-only benchmark import; its remaining API, adjoint, variance, device, invalid-input, and fixture regressions are implementation-owned.

Most companion modules import private functions from flat benchmark scripts after inserting the checkout's `benchmarks/` directory into `sys.path`; other tests use `spec_from_file_location`. Direct examples include the field-fit, joint-SMACS, depth, scorer, stacker, map-compat, real-data, and INR tests above. `scripts/smacs0723_full_mosaic_pipeline.py` and `scripts/smacs0723_field_fit_full_reference.py` also import private benchmark helpers and inject `src`/`benchmarks`. These imports must become harness-internal package imports. Benchmark entry points themselves commonly inject `src` and/or `benchmarks`; remove that checkout dependence and bind each run to the explicitly selected installed wheel/source subject. `benchmarks/config.py` is a shared `src`/`PYTHONPATH` injector and should be removed from harness execution paths. No direct loading of torchsky implementation files was found in the inspected map suite.

## FITS ownership findings

Forbidden `astropy.io.fits` imports exist in the relevant evaluation and test surfaces and must be replaced by torchfits APIs (`torchfits.open/read/get_header/write`, public HDU/TensorHDU/Header objects):

- Evaluation: `benchmarks/bench_map_compat.py`, `benchmarks/bench_map_image_plane.py`, and `benchmarks/bench_map_smacs0723_reproduce.py`.
- Package tests that must remain independent of Astropy FITS: `tests/test_map_image_plane.py`, `tests/test_map_smacs0723_real.py`, and `tests/test_map_reproject_fft_parity.py`; these are package-side FITS policy debt even when associated parity coverage moves.
- Harness tests: `tests/test_map_compat_bench.py`, `tests/test_map_dfreproject_parity.py`, `tests/test_multiband_depth_benchmark.py`, `tests/test_updec_webb_real.py`, and `tests/test_build_smacs0723_minicutout.py`.
- Data helpers: `scripts/build_smacs0723_minicutout.py` and the local optional import in `scripts/build_smacs0723_stack_baseline.py`.

`tests/test_map_smacs0723_real.py` already has an explicit test asserting the MEF/aux-file path uses torchfits, so it is a useful migration contract. No Python `fitsio` import appeared in the inspected `benchmarks/`, `tests/`, or `scripts/` tree. Additional non-campaign FITS-policy debt exists in `scripts/apply_wcs_to_mef.py`, `scripts/astrometric_calibrate.py`, `benchmarks/bench_pipeline_table_sphere.py`, `benchmarks/bench_gpu_memory.py`, `examples/example_image_mef.py`, and `tests/test_cli_calibration_manifest.py`; these should be tracked separately if outside this extraction slice. The staged `scripts/bench_cfht_mef_mosaic.py` now reads SWarp output with `torchfits.read`.

## Staged migration status

The evaluation files and companions listed above are copied into the harness
stage while preserving the original dirty contents. Benchmark and script
imports now use the installed `torchsky_harness.benchmarks` and
`torchsky_harness.scripts` namespaces; the migrated tests no longer insert a
checkout path into `sys.path` or load benchmark files with
`spec_from_file_location`. The map/SMACS FITS fixture and campaign boundaries
use `torchfits` public HDU and read/write APIs. The root maintainer owns the
remaining core test trims and any source-tree file deletion after its copy
manifest check; no files have been removed from the core checkout in this
subtask.

Focused results and exact test commands are recorded in
[`map-migration-verification.md`](map-migration-verification.md).
