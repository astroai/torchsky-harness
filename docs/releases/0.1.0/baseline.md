# Release baseline — torchsky 0.4.0 + torchsky-harness 0.1.0

Status: Phase 0 baseline recorded; Phase 1 audits partially integrated.
Date: 2026-10-18.

This file is the release evidence anchor. It records the exact state that later
gate results must reference; it is not a claim that any scientific gate passed.

## Repositories

| Repository | State |
|---|---|
| `torchsky` | `3f13770585faff59860edbff3be42392818f176d`; release branch `release/0.4.0` created from the dirty worktree (404 pre-existing/working changes preserved); `main` ahead 3 / behind 2 of `upstream/main` |
| `torchsky-harness` | `356fb7b`; release branch `release/0.1.0` created from the existing dirty worktree (15 pre-existing changes preserved, not amended) |

## Subject identity

Installed explicitly as an editable source build in the harness `test` environment;
no checkout path was injected into the harness.

```json
{
  "distribution": "torchsky",
  "version": "0.4.0.dev0",
  "import_path": "/Users/fabbros/src/astroai/torchsky/src/torchsky/__init__.py",
  "package_content_sha256": "44b955d9a1d2da18267a1830d95567c3df0eee61a685e57c64cb5a0357f462ca",
  "direct_url": {"dir_info": {"editable": true}, "url": "file:///Users/fabbros/src/astroai/torchsky"},
  "source": {"revision": "3f13770585faff59860edbff3be42392818f176d", "dirty": true,
             "tracked_diff_sha256": "4da6ed99fcf900c40083abf95fc9a0b5494ce6712bcd73f137815ab902dc8aa4"}
}
```

The release candidate must be a clean, tagged source build; this dirty record is
baseline evidence only.

## Environment

- macOS 14.8.2 arm64
- Python 3.13.14 (Pixi `test` environment)
- torch 2.13.0, torchfits 1.0.0
- harness 0.1.0.dev0, installed editable into the subject environment
- harness `test` environment also has the subject installed editable

## Checks run

| Check | Result |
|---|---|
| `pixi run preflight` (harness) | pass |
| `pytest tests/test_runner.py tests/test_run_local_sphere_bench.py -q` | 6 passed |
| `python -m torchsky_harness.cli list` | 78 evaluations: 9 WCS, 12 sphere, 48 map, 4 catalogs, 5 cross-domain |
| `python -m torchsky_harness.cli subject` | works; identity above recorded |
| torchsky test collection | 2,783 tests collected, no import errors |
| `pytest tests/test_docs_integrity.py -q` | pass after removing two migration-deleted archive paths from the expected list |
| Full torchsky suite (`pytest -n auto -q`) | **2692 passed, 104 skipped** in 274 s after the three baseline fixes |

## Known red or unverified

- Full torchsky suite is green on this machine as of the latest rerun:
  **2692 passed, 104 skipped**. The first baseline run exposed three failures,
  now fixed:
  - `test_field_gaussian_likelihood.py::test_joint_query_dtype_device_and_normal_quadratic_contract[mps-dtype3]` —
    float32/MPS footprint tolerance.
  - `test_gaussian_likelihood.py::test_available_device_likelihood_and_sampling_query[mps]` —
    `complex128` cast on MPS.
  - `test_wcs_alignment.py::test_wcs_with_affine_fit_correction_updates_linear_header` —
    missing `TSKYAF0` provenance marker.
- torchsky worktree is substantially uncommitted; the deletion set moved 57 test
  files and benchmark code to the harness, and the source set includes new
  untracked native-likelihood modules.
- No release gate has run against this subject yet; `bench_all.py` catches domain
  exceptions and returns zero, so it cannot certify a gate.
- The harness has 78 evaluations but only a small number have committed
  baselines/acceptance JSON; most are ad-hoc reports.
- `torchsky-harness` CI runs only infrastructure tests; the scientific workflows
  are manual and use a private `.subject` checkout.

## Audit blockers (verified during Phase 1; open unless stated)

### WCS

| Blocker | Evidence |
|---|---|
| Inverse solvers can return unconverged pixels silently | SIP stops after 6 iterations, TPV after 20, PCO 64, MOL 5; no residual/converged mask |
| Projection-domain policy is clamp/path dependent | CEA clamps `asin`; SIN/ZEA mask only theta; HPX generic vs 2D fast paths disagree at edges |
| TPV/TNX/ZPX behavior is under-validated | Any TAN header with PV becomes TPV; unknown TPV indices dropped; no high-order TNX/ZPX parity |
| Fused/compile caches can go stale | one-shot C++ verification; `to()` does not clear compiled graphs; `__setattr__` only invalidates cpdis/det2im |
| No WCS performance baseline | `benchmarks/baselines/` contains zero WCS baselines; `bench_wcs_suite.py` always returns 0 |
| Dead/duplicate WCS surface | four short aliases, unused `transform_spectral`, dead `solve_newton_raphson`, dead vmap paths |

### Sphere

| Blocker | Evidence |
|---|---|
| High-order MOC decode corrupts pixels | `uniq2nest` infers order with float64 `log2`; orders 24–29 can return doubled nside and pixel −1/−2 |
| No nside/order cap | `_validate_nside` accepts any power of two; `npix=12·nside²` can overflow int64 |
| Import-time `torch.compile` | `_smoke_test_torch_compile()` runs on `import torchsky.sphere` |
| SHT adjoint/device gates incomplete | public spin adjoint and CUDA/MPS numeric parity are not gated; gradcheck covers only the torch ring path |
| Three SHT/alias surfaces | `harmonics.py`, `healpy.py`, `compat.py` plus direct re-exports; docs disagree on the canonical name |
| MOC cone/union and SHT throughput not claimed | scorecards record ~0.06×/~0.0009× MOC and ~0.3–0.4× Linux spin vs healpy |
| `bench_all` cannot fail | catches sphere exceptions and returns 0 |

### Map and multi-spectral

| Blocker | Evidence |
|---|---|
| Spectral flux is not conserved | wide-band values are replicated across target bins; total flux scales with bin count |
| Spectral units/types are not normalized | WAVE/Angstrom assumed; FREQ/ENER/WAVN/micron/non-uniform axes compared raw; products force Angstrom |
| IFU cube spectral alignment is wrong for non-uniform axes | `d_wave` assumes uniform spacing; `zt` is a linear index; inferred axes hardcoded to 16 bins |
| Cube/IFU/HEALPix memory is unbounded | full `(n_band,H,W)` and `(N_wave,H,W)` allocations; no tile/chunk argument |
| HEALPix coadd is 2D-only and centre-sampled | no band axis, no area quadrature, no variance product; harmonic sparse fallback is bilinear |
| CLI/product and `coadd` paths diverge | CLI cannot emit cubes; product path skips shifts/outlier/variance handling |
| Weight/IVAR semantics are inconsistent | `weight_is_inverse_variance` does not gate all paths; IFU sets `inverse_variance=denom` for relative weights |
| No colour/uncertainty/flux gate | baselines check RMSE, SNR, total flux, time; band flux/colour metrics exist but are not checked |

### Secondary surfaces

| Blocker | Evidence |
|---|---|
| Docs advertised a missing `torchsky.surfaces` | removed from `docs/api.md` and `docs/architecture.md` in this baseline |
| Scene/field name collisions | `Observation`/`Target`/`render` differ across `scene`, `field`, `map`; no contract test |
| Field export wall and hidden shim | 99 root exports; `field/frequency_adaptive.py` shadows `map/field/frequency_adaptive.py` |
| Catalogs docs/dead code drift | per-row-radius docs contradict implementation; declination variants are test-only |
| Viz is not self-contained | Aladin Lite CDN, empty Vite scaffold, silent static mount failure |
| Coords has no unique API | all 16 exports already in `wcs`; no alias contract test |

## First gate/smoke run

`pixi run -e test -- python -m torchsky_harness.cli run bench_wcs --run-dir
.local-artifacts/runs/phase0-wcs-tan -- --cases TAN --n-points 256 --runs 1
--device cpu` completed with `execution.json` (`scientific_validity:
not_assessed`, as expected for the current runner). Recorded row:

| case | valid | astropy fw ms | torch fw ms | fw ratio | astropy inv ms | torch inv ms | inv ratio | max angle asec | p99 angle asec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TAN | 256 | 0.082 | 0.168 | 0.49 | 0.109 | 0.111 | 0.98 | 3.7e-10 | 2.8e-10 |

This is a smoke record, not a performance claim (small-N launch overhead dominates).

## Progress after the baseline (2026-10-18)

- Fixed `torchsky/tests/test_docs_integrity.py` for two archive files moved out
  of the core tree; the test now passes.
- Removed the non-existent `torchsky.surfaces` claim from `docs/api.md` and
  `docs/architecture.md`; fixed the broken getting-started scene/map examples.
- Sphere: replaced float `log2` MOC order decoding with exact integer arithmetic,
  capped HEALPix `nside` at `2**29`, and added high-order boundary tests.
  `tests/test_sphere_moc.py`: 48 passed; sphere/HEALPix suites: 298 passed,
  8 skipped.
- Sphere: removed import-time `torch.compile`; compilation is now probed lazily
  on first use (`TORCHSKY_SHT_COMPILE=0` disables it). SHT suites: 80 passed,
  2 skipped.
- Harness: `bench_all.py` now returns non-zero when a WCS or sphere domain
  raises, and a contract test forces a domain failure to verify the exit code.
- Harness: `check_map_benchmark_regressions(..., require_complete=True)` now
  fails on missing baseline rows/metrics; default behaviour is unchanged until
  release gates opt in. The three map gates and `bench_map_compat` now expose
  `--require-complete-baseline`; existing smoke tasks do not pass it yet, so
  their baselines must be regenerated to exactly match each gate before the
  option is enabled.
- Harness: `bench_wcs_suite.py` and `bench_sphere_suite.py` now return non-zero
  when any row has `status=FAILED`; `bench_all.py` also fails on `FAILED` rows,
  not only exceptions. `tests/test_map_metrics.py`: 7 passed; compatible
  harness tests: 49 passed, 2 skipped.
- Harness: added `tests/test_bench_all_contract.py`; infrastructure tests now
  7 passed.
- Fixed the three full-suite failures found in the first baseline run:
  - WCS `to_header()` now preserves the `TSKYAF*` provenance marker for
    undistorted affine corrections; `tests/test_wcs_alignment.py`: 6 passed.
  - Solver inner products use `complex64` on MPS instead of unsupported
    `complex128`; the MPS likelihood contract passes.
  - The linear-band-grid footprint bounds check uses a float32/MPS tolerance
    rather than the float64 `1e-10` threshold; the MPS field contract passes.
- **API boundary reset (breaking):** removed the torchfits alias layer from
  torchsky: root `read/write/open/get_header/get_wcs/HDUList/Header/...`, the
  `io`, `table`, `hdu`, `cache`, `datasets`, `dataloader`, `header_parser`, and
  `core` modules, and the matching compatibility tests. All internal, test, and
  example callers now use `torchfits` directly. Full suite after the reset:
  **2658 passed, 98 skipped**; torchsky preflight and harness preflight pass;
  harness catalog/runner/metrics tests pass; a WCS smoke run completes with
  parity 3.7e-10 arcsec.
- **Scene and map consolidation:** the experimental scene facade now contains
  only representations, target specs, and operator adapters; duplicated map/field
  verbs were removed. Map product classes were canonicalized to
  `ProductSpec`/`BandpassSpec`/`ProductResult`/`UncertaintyProduct`,
  `stack_map_product` became `stack_product`, and `map.__all__` is now the
  workflow set only. The target ownership model is recorded in
  [architecture-review.md](architecture-review.md).
- Full torchsky suite after the scene/map consolidation: **2658 passed,
  98 skipped**; targeted scene/map/docs tests: **357 passed, 3 skipped**.
- **Field facade and API guard:** `torchsky.field` now exposes the
  model/fit/render/sample composition surface and diagnostics; concrete model
  classes stay under `torchsky.map.field`. Added
  `scripts/api_manifest.py`, `docs/api-manifest.json`, and
  `tests/test_api_manifest.py` to freeze the public `__all__` surfaces for root,
  WCS, sphere, map, scene, field, catalogs, coords, and encoders. Preflight and
  API-manifest test pass; CI now runs the manifest check.
- **Sphere/map splits:** removed the duplicate `torchsky.sphere.harmonics` alias
  (implementation is `torchsky.sphere.spectral`) and extracted map observation,
  configuration, target, and result types from `map/image_plane.py` into
  `map/observation.py`. Full suite after these changes: **2659 passed,
  98 skipped**.
- **New spectral-cube gate:** added
  `benchmarks/bench_map_cube.py`, a synthetic truth-cube test that splits a
  known spatial-spectral cube into per-band 2D observations and recombines them
  with `coadd_cube`. It gates voxel RMSE (`<1e-5`), maximum absolute error, and
  band/total flux conservation (`<1e-3`); baseline
  `benchmarks/baselines/map_cube_smoke.json` and task `bench-map-cube-gate` are
  committed. Smoke and unit test pass (RMSE ≈4.9e-8). This covers 2D-band cube
  assembly, not the full 3D IFU/WCS evaluator, which remains open.
- **Native likelihood/evidence gate:** added
  `benchmarks/bench_native_likelihood.py` and baseline
  `benchmarks/baselines/native_likelihood_smoke.json`. It solves a diagonal
  native Gaussian likelihood with known truth, checks the recovered state
  against posterior standard deviations (max standardized error 0.226), and
  verifies that mutating bound data raises a stale-evidence error. Task
  `bench-native-likelihood-gate` and a unit test are committed.
- **Native public API frozen:** added `__all__` to `map/query.py`,
  `map/operators/continuous_response.py`, and `map/operators/epsf.py`, and
  extended `docs/api-manifest.json` to cover the native likelihood, query,
  evidence, nuisance, continuous/detector response, pixel-calibration, EPSF,
  native-response, native-physical, band-grid Gaussian, and joint-native
  modules. The manifest guard and preflight pass.
- **Native nuisance gate:** added `benchmarks/bench_native_nuisance.py` and
  baseline `native_nuisance_smoke.json`. It compares profiled and calibrated
  nuisance precision actions against independent dense references and gates
  max absolute error (<1e-10, observed ~9e-16) and profiled rank. Task
  `bench-native-nuisance-gate` and a unit test are committed.
- **Map spectral helper split:** extracted the spectral-axis validation,
  interval projection, Gaussian-filter projection, band-name, and band-metadata
  helpers (225 lines) from `map/image_plane.py` into
  `map/spectral_projection.py`. `image_plane.py` is now ~6,045 lines, down from
  6,588 at the start of the split work; map tests pass.
- **Native response gate:** added `benchmarks/bench_native_response.py` and
  baseline `benchmarks/baselines/native_response_smoke.json`. It injects a known
  pixel-integrated Gaussian star on a native stamp, fits flux plus constant
  background at the true centroid, and gates flux error (<5%) and background
  error (<0.2). Smoke recovers flux to 0.09% and background to 0.0067; task
  `bench-native-response-gate` and a unit test are committed. This is a
  linear-profile gate, not a full likelihood/joint/real-detector claim.
- **3D IFU matched-sampling gate:** added `benchmarks/bench_map_ifu.py` and
  baseline `benchmarks/baselines/map_ifu_smoke.json`. A synthetic IFU
  observation carries one spectral sample per detector pixel via a wavelength
  map; the target cube uses the same sampling. The gate checks exact valid-voxel
  recovery (max error <1e-5) and the expected sparse coverage (~1/N slices).
  Smoke: zero error, valid fraction 0.0845 vs expected 0.0833; task
  `bench-map-ifu-gate` and a unit test pass. This does **not** claim dense
  spectral interpolation; an earlier unmatched 40 Å target grid produced sparse
  nearest-mapped output and is recorded as a limitation, not an accuracy gate.
- **Runner verdict fields:** `execution.json` no longer hard-codes
  `scientific_validity: not_assessed`. `src/torchsky_harness/evidence.py`
  declares the six release-gate claims (scope, required rows, limitations) and
  the runner derives `established` / `not_established` / `not_assessed` plus an
  independent `comparison_eligibility` (`suite_owned` / `baseline_compared`)
  from the child exit code, the `--json-out` report, and the baseline flags.
  Undeclared evaluations stay `not_assessed` and every failure is listed in
  `verdict_reasons`. All six gates now record `established` +
  `baseline_compared`; record format documented in `docs/evidence-verdicts.md`
  and covered by `tests/test_evidence_verdict.py`.
- **Map module split:** extracted the spectral-cube path
  (`_has_3d_ifu_observations`, `_coadd_ifu_3d_cube`, `coadd_cube`, 387 lines)
  into `torchsky/map/cube.py` and the direct-to-HEALPix path
  (`coadd_healpix`, `resample_healpix`, 305 lines) into
  `torchsky/map/healpix_coadd.py`. `map/image_plane.py` is now 5,345 lines,
  down from 6,037 before this split and 6,588 before the split work started;
  `map.__init__` loads the moved entry points lazily from the new modules and
  the frozen API manifest is unchanged.
- **IFU spectral-axis gate and non-uniform-axis fix:** `coadd_cube` mapped
  every detector sample onto the target spectral axis with a single uniform
  `d_lambda`, so a non-uniform target axis placed samples in the wrong bins
  (for example 5100 A landed between the 5040 and 5060 A bins of an irregular
  axis). torchsky now uses a per-interval fractional index
  (`map/spectral_projection._spectral_fractional_index`), with regular axes
  unchanged, plus `tests/test_map_cube_spectral_axis.py`. The harness gate
  `bench_map_ifu_spectral_axis` (task `bench-map-ifu-spectral-axis-gate`,
  baseline `map_ifu_spectral_axis_smoke.json`) assembles a synthetic long-slit
  `IFUSliceWCS` cube and gates: exact line localization on an irregular axis for
  all kernels (zero spurious voxels), exact constant-spectrum normalization
  (7.6e-7), regular-axis smooth-spectrum interior accuracy (6e-4 vs a 2.5e-3
  tolerance), and a recorded 10% envelope for irregular-axis smooth spectra
  (3.2% observed). The irregular-axis envelope is a documented limitation: the
  kernel interpolates in index space, so strongly irregular bin widths bias
  smooth spectra; physical wavelength-space interpolation remains open.
- **HEALPix multi-band gate:** added
  `benchmarks/bench_map_healpix_multiband.py`, baseline
  `benchmarks/baselines/map_healpix_multiband_smoke.json`, task
  `bench-map-healpix-multiband-gate`, and a unit test. Three synthetic bands
  (0.04/0.08/0.16 deg native pixels, same sky) are coadded with
  `coadd_healpix` onto a common nside-2048 sparse grid and validated against
  analytic truth: constant fields recovered to 1.9e-7 relative, covered solid
  angle within 0.82% of the pixel-centre footprint area, band ratios exact to
  7.6e-8, a resolved source recovered per-pixel to 1.9e-7 of peak with a 0.076"
  absolute and 0.106" cross-band source position, a linear sky slope recovered
  to 1.2e-6 with band and flux ratios to 2.4e-7, and a band-stacked
  `resample_healpix` round trip to 1.9e-7. This covers per-band coaddition and
  band stacking; it does not claim flux-conserving reprojection (the kernel
  samples image values at HEALPix pixel centres), harmonic/SHT resampling,
  throughput against healpy, or variance propagation.
- The full harness test suite is opt-in and was stopped after one early failure
  while running long/external cases; this is not a release gate result. Targeted
  harness boundary and gate tests pass. Record the full-suite result separately
  before release.
- Docs/changelog/README now state the boundary explicitly; the removed
  compatibility-transition text is gone.
- Full suite rerun: **2692 passed, 104 skipped** earlier; after deleting three
  compat test modules the suite is **2658 passed, 98 skipped**.

The torchsky dirty-diff hash advanced because the fixes above are part of the
release work; the package-content hash is unchanged. Re-run `subject` before
each gate run and record the new hash in that run's `execution.json`.

## Newer native field-response work is in scope

The paused native-response/likelihood work is not outside the refactor; it is
part of the 0.4.0 surface and must be preserved in the migration commit and the
API decision matrix.

| Module | Lines | Role |
|---|---:|---|
| `torchsky/map/likelihood.py` | 510 | Gaussian likelihood factors, normal equations, functional uncertainty |
| `torchsky/map/query.py` | 291 | bounded forced-photometry/blend linear queries |
| `torchsky/map/evidence.py` | 127 | evidence snapshots and staleness guards |
| `torchsky/map/field/native_response.py` | 503 | native source-node response |
| `torchsky/map/field/gaussian.py` | 410 | band-grid Gaussian likelihood binding |
| `torchsky/map/field/joint_native.py` | 328 | shared native nuisance/calibration binding |
| `torchsky/map/field/native_physical.py` | 613 | physical integrated-read likelihood |
| `torchsky/map/operators/continuous_response.py` | 445 | continuous source/spectral quadrature |
| `torchsky/map/operators/detector_response.py` | 281 | detector moments and covariance |
| `torchsky/map/operators/nuisance.py` | 330 | profiled/calibrated nuisance precision |
| `torchsky/map/operators/pixel_calibration.py` | 169 | pixel radiometric/flat transfer |
| `torchsky/map/operators/epsf.py` | 302 | native PSF response references |

The corresponding docs (`docs/native-likelihood.md`,
`docs/psf-response-conventions.md`) and the field-response plan/status/amendments
are now linked from `docs/README.md`, `docs/api.md`, and the docs-integrity test.
The new tests are added to dedicated `native-likelihood` and `native-physical`
CI shards instead of remaining local-only.

Release treatment: **experimental/beta**. The package tests are strong (the latest
recorded fourteen-suite integration is 437 passed / 30 device skips; the full
local suite is green), but there is no harness benchmark gate yet, no M1
scientific acceptance, and the real ACS result is a bounded engineering fit with
`training_ready=false`. Do not claim calibrated real-sky response, SOTA, or
universal device support. Add a synthetic native injection/recovery gate and
retain the ACS campaign as evidence, not a pass.

## Next actions

Done for 0.4.0 / 0.1.0:

1. IFU/cube, HEALPix multi-band, and three native synthetic gates exist with
   committed baselines, `--require-complete-baseline`, declared claims, and a
   release evidence bundle (`release_ready: true` against the 0.4.0 wheel).
2. `execution.json` carries a real declared verdict, and
   `torchsky-harness evidence` collects the bundle.
3. Versions are `torchsky 0.4.0` / `torchsky-harness 0.1.0`; both repositories
   carry a tag-driven PyPI trusted-publishing workflow with wheel-install
   smoke checks, documented in [release-procedure.md](release-procedure.md).
4. The harness bounded suite is green after migrating benchmark modules to the
   `torchfits` I/O boundary.

Open after 0.4.0 (do not claim these yet):

1. Run the release gates on the pinned Linux runner and replace the local
   macOS wheel evidence with that bundle.
2. Physical wavelength-space spectral interpolation and flux-conserving
   redistribution for irregular IFU axes; the current kernel is index-space.
3. HEALPix variance/weight products, harmonic/SHT resampling, and comparator
   throughput rows.
4. Trim the remaining legacy `torchsky.map` exports (166 names outside
   `map.__all__`, still reachable through the lazy export table) and finish the
   giant-file splits: `wcs/core.py` 5,297 lines, `sphere/spectral.py` 4,701,
   `map/field/core.py` 5,753.
5. Real-data acceptance rows (SMACS0723 depth claims) and the experimental
   native/scene/field surfaces remain out of the 0.1.0 claim set.

- **Harness bounded suite green:** 38 harness benchmark/test/script modules
  still called the torchsky I/O aliases that the 0.4.0 boundary reset removed
  (`torchsky.read/write/get_header/get_wcs`), which failed 86 bounded-suite
  tests. They now use `torchfits.read`, `torchfits.write`, and
  `torchfits.read_header` (with `WCS(read_header(...))`), the two monkeypatch
  sites and the `get_wcs` test were updated, and the tiled-vs-full gradient
  check in `test_smacs_sampling` now uses the float32 accumulation tolerance
  (float64 agreement is ~1e-9 relative). Added the `test-bounded` pixi task:
  **942 passed, 152 skipped, 17 deselected** in 242 s against the 0.4.0 wheel. The default `test` task
  still runs the opt-in `performance`/`benchmark`/`slow`/`integration` cases.

- **Subject provenance fix:** `identify_subject()` resolved the subject's git
  identity from the directory that contains `site-packages`, so a wheel install
  inside a checkout recorded the *harness* repository revision as the subject
  revision. `source_identity()` now reports `kind=archive` with the wheel
  SHA-256 for wheel installs, `kind=source_tree` with the real source revision
  for editable installs, and `kind=registry` otherwise; the evidence bundle
  records the source kind and digest per run. Covered by
  `tests/test_subject_source.py`.
## Release candidate evidence (2026-10-18)

Subject artifact: `torchsky-0.4.0-cp313-cp313-macosx_11_0_arm64.whl`
(sha256 `f9e537dd49fdfccb17ad49774567d3e5a835af9d84132cb18cb6e560b07d6259`),
built from torchsky `4055e819` and installed into the harness test environment as
a wheel (import path `site-packages/torchsky`, not the checkout). Later
tooling/docs commits (e.g. the pyright config fix) do not change the packaged
content, so the wheel hash and subject identity below are unchanged. Subject
`package_content_sha256 = 49f816de7a1daafc60b9695e34897e13b2916148c3816e781d05ab7cec63d0d6`,
`source.kind = archive`. Every gate below is a `pixi run` task against that
install; the bundle records the harness revision.

### Synthetic and truth gates (baseline_compared)

| Gate | Evaluation | Verdict |
|---|---|---|
| `bench-map-cube-gate` | `bench_map_cube` | established / baseline_compared |
| `bench-map-ifu-gate` | `bench_map_ifu` | established / baseline_compared |
| `bench-map-ifu-spectral-axis-gate` | `bench_map_ifu_spectral_axis` | established / baseline_compared |
| `bench-map-healpix-multiband-gate` | `bench_map_healpix_multiband` | established / baseline_compared |
| `bench-native-response-gate` | `bench_native_response` | established / baseline_compared |
| `bench-native-likelihood-gate` | `bench_native_likelihood` | established / baseline_compared |
| `bench-native-nuisance-gate` | `bench_native_nuisance` | established / baseline_compared |

### Reference-comparison gates (suite_owned thresholds)

| Gate | Evaluation | Comparator | Measured |
|---|---|---|---|
| `bench-wcs-release-gate` | `bench_wcs` | astropy | 30 projections: angular parity max 4.13e-5 asec (AIR formula floor; others ≤4e-8), pole/seam-excluded sky round trip 4.13e-5 asec, pixel round trip 1.27e-6 px, p99 sky round trip 6.3e-3 asec; forward 0.54–3.70× and inverse 0.86–6.90× astropy |
| `bench-healpix-release-gate` | `bench_healpix` | healpy | 6 ops at nside 512: zero index mismatches, max angle 1.14e-13 deg, throughput 1.02–2.19× |
| `bench-healpix-advanced-release-gate` | `bench_healpix_advanced` | healpy | 6 ops: zero mismatches, throughput 1.25–2.06× |
| `bench-sphere-matrix-release-gate` | `bench_sphere_matrix` | healpy | 3 profiles at nside 512: index/angle parity gates pass; median ratios 2.00/1.54/1.27/0.98/1.43/0.98 recorded but not gated on macOS (run-to-run noise crosses the Mac-tuned threshold; pinned-runner work) |
| `bench-spin-matrix-release-gate` | `bench_spin_matrix` | healpy | spin-2 SHT: no failures, small-bucket median ratios 0.459/0.422 vs 0.15 thresholds |
| `bench-catalog-release-gate` | `bench_catalog_compat` | astropy SkyCoord | 6 rows OK, max separation difference 1.13e-10 asec |

Quad-cube seams and all-sky poles are conditioning properties of the
projections, not mapping errors: `bench_wcs` verifies them with sky-domain and
pole-excluded metrics and exempts TSC/CSC/QSC from the pixel round-trip
thresholds (astropy shows the same seam behaviour). AIR carries a ~1e-8 deg
formula-precision floor shared by astropy's implementation.

`pixi run evidence-bundle` reports `release_ready: true`,
`subject_consistent: true`, `verdict_counts = {established: 13}`,
`missing_claims: []`. The bundle (git-ignored) is regenerated for the tag;
attach it to the release notes. Comparison throughput numbers are same-machine
(macOS) ratios, not pinned-runner performance claims; correctness metrics are
hardware-independent.

## Verification iteration (0.4.0 candidate)

The candidate was re-verified with the wider harness inventory and the failures
found were fixed or recorded, not hidden:

- **Harness I/O boundary:** 38 benchmark/test/script modules still imported the
  removed `torchsky.read/write/get_header/get_wcs`; migrated to `torchfits`
  (86 bounded-suite failures to zero).
- **Subject provenance:** a wheel installed under a checkout inherited that
  checkout's git revision; `source.kind=archive` + wheel SHA-256 is now
  recorded instead (`tests/test_subject_source.py`).
- **Comparator inventory:** added reference-comparison gates and claims for WCS
  (30 projections vs astropy), HEALPix core + advanced (vs healpy), sphere
  geometry and spin-2 SHT (vs healpy), and catalog crossmatch (vs astropy).
  `bench_wcs` now reports sky-domain round trips, pole-excluded maxima, p99
  round-trip gating, and round-trip exempt cases for quad-cube seams; the
  previously misleading 0.7 px MOL "error" is identified as all-sky polar
  conditioning and the 45 asec CSC artifact as seam conditioning identical in
  astropy.
- **Summary-report evidence:** evaluators that emit a summary object
  (`bench_sphere_matrix`, `bench_spin_matrix`) can now back a claim through a
  declared failure list.
- **Map truth gates:** `bench_map_ground_truth` and
  `bench_map_multifield_truth` pass their committed smoke baselines.
  `bench_map_field_fit` was re-baselined: the stale baseline predated a
  support/metric-definition change (current RMSE is 7× lower at 0.0013 while
  the derived `signal_to_residual_db` scale moved), so the old row was not a
  comparable regression. It passes the re-baselined smoke contract; a held-out
  noninferiority baseline is still open.
- **Suites:** torchsky 1,899 + 773 passed (98 skipped); torchsky `pyright`
  0 errors; harness bounded suite **942 passed, 152 skipped, 17 deselected**.
  The default harness `test` task also runs opt-in performance cases
  (`test_inr_perf` and friends) that train models for tens of minutes; the
  release suite is `test-bounded`.

- **Cross-library map comparison:** the map-compat reproject adapters passed
  `torchfits` headers straight to `reproject`/`dfreproject`, which require an
  astropy `Header`/`WCS`, so every reproject row silently reported `SKIPPED`
  ("Input to a HighLevelWCSWrapper must be a low level WCS object"). The
  adapters now build a sanitized astropy `Header`; reproject comparisons run
  for the synthetic shifted WCS and both astropy SIP fixtures (all rows OK).
  On the synthetic two-observation shift, torchsky takes 11.2 ms vs 17.7 ms for
  `reproject_interp` and 34.9 ms for `reproject_exact`, with a 0.128 RMSE
  disagreement on the noisy synthetic field (interpolation-kernel difference to
  investigate against the existing drizzle/swarp CI baselines; not a release
  claim). This is supporting evidence, not a declared gate. The older
  `bench-map-compat-gate` also fails locally on a 1/256 (`0.00390625`) MEF
  drizzle row; it is not a declared release claim and needs a pinned-runner
  re-baseline.
## Version bump (release candidate)

torchsky is now `0.4.0` (pyproject, pixi, `__version__`) and torchsky-harness is
`0.1.0` with `dependencies = ["torchsky>=0.4.0"]`. Both repositories carry a
tag-driven PyPI trusted-publishing workflow
(`.github/workflows/publish.yml`) that verifies the tag matches `pyproject.toml`,
checks wheel contents, and runs an install smoke before publishing; the paired
procedure is [release-procedure.md](release-procedure.md). The recorded gate
runs predate the version bump, so the release bundle must be regenerated against
the `0.4.0` install before tagging.
