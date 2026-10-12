# Torchsky second-pass code ownership audit

Read-only audit of the current core and sibling harness trees. No tracked files were changed. Core currently has extensive pre-existing dirty/untracked changes; in particular, `src/torchsky/map/__init__.py` is modified and its diff was inspected so proposed API cleanup does not erase unrelated map-facade edits.

## High-confidence harness extraction: map product evaluation/reporting

`src/torchsky/map/product.py` still contains two harness-owned groups. Neither has callers from production `src/torchsky` code.

### Scientific metrics

These signatures currently compare prediction/truth or score uncertainty, and harness code calls them:

```python
evaluate_map_product(
    prediction: MapProductResult | Tensor,
    truth: MapProductResult | Tensor,
    *, weight: Tensor | None = None,
    header: dict[str, Any] | None = None,
    value_semantics: MapValueSemantics | None = None,
) -> dict[str, float | list[float] | None]

evaluate_map_frontier_regions(
    prediction: MapProductResult | Tensor,
    truth: MapProductResult | Tensor,
    *, support_weight: Tensor,
    overlap_count: Tensor | None = None,
    fringe_kernel_size: int = 9,
) -> dict[str, float]

evaluate_map_uncertainty(
    inverse_variance: Tensor,
    prediction: MapProductResult | Tensor,
    truth: MapProductResult | Tensor,
    *, support_weight: Tensor | None = None,
) -> dict[str, float]
```

Actual consumers are in `torchsky-harness`: `bench_map_field_fit.py`, `bench_map_ground_truth.py`, `bench_map_multifield_truth.py`, `bench_map_projection_mosaic_suite.py`, `diagnose_field_fit_residuals.py`, `diagnose_field_fit_sweep.py`, `diagnose_field_fit_stochastic_chunks.py`, `diagnose_map_large_union_reprojection.py`, and `tests/test_map_ground_truth_image_plane.py`. `evaluate_map_uncertainty` and `evaluate_map_frontier_regions` are used by `bench_map_field_fit.py`; `evaluate_map_product` has the broader list above. `diagnose_field_fit_residuals.py` calls the evaluator through its imported benchmark module (`field_bench.evaluate_map_product`).

Core consumers are limited to five tests in `tests/test_map_product.py` (two product metric cases, one uncertainty metric case, one frontier metric case, plus the benchmark-regression case) and public facade wiring. There are no `examples/` or `scripts/` callers. Current docs do call the short alias: `docs/map.md` lists it at line 19, documents an example around lines 173–181, and lists it again around line 195. The current API inventory in `docs/roadmap/ecosystem_parity_and_uv_ir_plan.md` also names it. `docs/archive/roadmap-sky-measure-scene-plan.md` has historical references only. `tests/test_api.py` expects `evaluate_product` in the promoted facade.

Safe extraction shape: move the three metric implementations unchanged into an existing harness benchmark/evaluation helper area, update the listed harness callers to import that local helper, then remove the runtime exports. Keep `MapProductSpec`, `MapProductResult`, `MapUncertaintyProduct`, WCS/pixel-area geometry, product planning/rendering/serialization, and `estimate_map_inverse_variance` in core. Do not add a core fallback import from the harness.

### Benchmark rows and artifact helpers

Move this cluster as one unit to the harness:

```python
build_map_benchmark_markdown(rows: Sequence[Mapping[str, Any]]) -> str
map_benchmark_row_key(row: Mapping[str, Any]) -> str
check_map_benchmark_regressions(
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *, reconstruction_rmse_tolerance: float = 0.10,
    signal_to_residual_db_tolerance: float = 0.10,
    total_flux_relative_error_tolerance: float = 0.10,
    time_tolerance: float = 0.15,
) -> list[str]
write_map_benchmark_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *, json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> str
```

Private closure: `_row_duration_ms` and `_row_label` are used only by `build_map_benchmark_markdown`; `map_benchmark_row_key` is used only by `check_map_benchmark_regressions`. Move those three helpers with the cluster. `json` is only used by `write_map_benchmark_artifacts`, so its core import can go. `Mapping`, `Sequence`, `Any`, and `Path` remain needed elsewhere in `product.py`.

Harness callers: `bench_map_field_fit.py`, `bench_map_ground_truth.py`, and `bench_map_multifield_truth.py` use the row checker/writer; `tests/test_map_field_fit.py` calls the writer through `_field_fit_bench`. The formatter and key helper have no independent external call sites. Core callers are the single `test_check_map_benchmark_regressions_respects_skip_time_flag` case and lazy facade mappings. No active docs/examples/scripts depend on these names.

### Private/runtime dependency boundary

- `_as_cube` is also used by `estimate_map_inverse_variance`, so keep it core. The moved evaluators may carry their own private 2D/3D shape helper; do not expose a new core public helper just for the harness.
- `_mean_pool_cube` is also used by `estimate_map_inverse_variance`, so keep it core. The moved frontier evaluator can implement its own local pooling helper.
- `WCS` and `pixel_area_sr_map` are referenced only inside `evaluate_map_product`; once that function moves, their imports from `product.py` appear removable. The geometry implementation stays core and the harness evaluator can use the installed public WCS/pixel-area surface.
- `math` remains needed by bandpass/product validation even after metric extraction. `F` remains needed by the core `_mean_pool_cube`; `torch`, `Tensor`, `nn`, and map/spec types remain widely used by production code.

### Public API closure to update when authorized

The current dirty `src/torchsky/map/__init__.py` adds the promoted alias `evaluate_product` in `_EXPORTS`, `TYPE_CHECKING`, and `__all__`; keep the rest of that dirty diff intact, but remove this alias as part of the extraction. The lazy `_EXPORTS` mapping also contains `evaluate_map_product`, `evaluate_map_frontier_regions`, `evaluate_map_uncertainty`, `build_map_benchmark_markdown`, `map_benchmark_row_key`, `check_map_benchmark_regressions`, and `write_map_benchmark_artifacts`; remove those mappings. Remove matching imports from `tests/test_map_product.py`, move the five evaluation/regression tests to the harness, and remove `evaluate_product` from the `tests/test_api.py` expectation. The active `docs/map.md` example/API list needs a corresponding owner update. Public API removal needs the targeted core tests and core typecheck described by AGENTS.md.

## Keep in core despite evaluation/validation names

- `estimate_map_inverse_variance` is an inference/uncertainty construction algorithm used as a product utility; its tests stay with map product runtime tests.
- `field_image_quality_metrics` is called from core training (`map/field/training.py`) and powers the public field quality renderer; `render_field_quality`/`render_field_fit_quality` are user-facing product diagnostics. `field_quality_scorecard` is called by `render_field_quality` when requested and is exposed in `torchsky.field`; keep these APIs and bounded tests.
- `operator_calibration_score_report` is called inside the fit path in `map/operators/calibration.py` (initial/final train reports) and is documented in `docs/field.md`; keep it. A “validation” name alone is not grounds to move code.
- Product specs/results, renderers, calibration input validators, and the field linear inverse path remain runtime ownership.

## Calibration evidence: split instead of deleting the whole dependency

`src/torchsky/cli/calibration_command.py` is a JWST/NIRCam-specific `calibration-manifest`/`calibration-extract` application: it hardcodes PATT_NUM train/validation/test splits, reads detector exposure metadata, attaches target evidence, and writes hashed manifests. `src/torchsky/calibration_evidence.py` contains detector candidate discovery, reviewed-catalog parsing, PSF/background target extraction, and the manifest artifact pipeline. Those are evaluation/data-preparation candidates for harness ownership.

However, `load_operator_calibration_split(...)` in `calibration_evidence.py` is called by `src/torchsky/map/field/inverse.py` at two runtime sites (around lines 2493 and 2570) while loading and validating trained operators. Keep that validated runtime input adapter, either as a slim core submodule or by changing the inverse API to consume validated tensor records. Do not move the entire module until those dependencies are severed. The generic operator calibration algorithms and their bounded unit/gradient tests remain in core. `tests/test_cli_calibration_manifest.py` currently mixes manifest/extraction integration tests with the split-loader schema/hash test; split those cases along with the implementation. This file also uses `astropy.io.fits` to write fixtures, which violates the core FITS policy; replacing those fixture writes with torchfits is needed if any tests remain in core.

## Remaining scientific scripts

`scripts/tweakreg_align_exposures.py` explicitly says its output naming exists for the old `bench_map_smacs0723_real.py` consumer. `tests/test_tweakreg_wrapper.py` tests that benchmark glob plus preflight guards. Move the script and its test with the harness only after updating its current project-doc references. The script is also described in `docs/projects/wcs-aware-smacs0723-pipeline.md`, so it is presently a documented workflow, not an unreferenced orphan.

`astrometric_calibrate.py` and `apply_wcs_to_mef.py` are operational astrometry/coadd tools documented in `docs/astrometric_calibration.md`; their current content is a user workflow rather than an evaluator/scorecard. Do not move them solely because they use astronomy tools. Both still import `astropy.io.fits`, contrary to the core I/O policy and requiring a separate torchfits migration or ownership decision.

## Test retention boundary

Keep `tests/test_perf_gate.py` (the core instructions explicitly assign core hot-path smoke gates there), the sampled `test_sphere_high_nside.py` checks (NSIDE 1024 but only a few pixels), streamed-tile memory/gradient contracts, device and gradient tests, and bounded synthetic/oracle regressions. Do not move every test that imports Astropy/Healpy/reproject: the remaining examples are small API/semantic checks (`test_wcs_api.py`, `test_wcs_frames.py`, catalog covering) or one bounded HEALPix-to-TAN comparison, not exhaustive external parity suites. `tests/test_field_diagnostics.py` includes a profiling-return-shape check, not a timing threshold or benchmark gate.

## Extraction closure summary

1. Extract the map metric and benchmark-row helpers with math preserved; move/retarget only the five evaluation-oriented product tests and update the active public API test/docs owner.
2. Keep runtime uncertainty estimation, geometry, product APIs, public quality/training helpers, and the calibration split loader that core inverse loading calls.
3. Split NIRCam evidence generation/CLI from the core loader only after moving that runtime dependency; move the benchmark-tied TweakReg wrapper/test with its current docs references handled.
4. Preserve the user's existing dirty map facade edits except for the intentional removal of harness-owned names.
