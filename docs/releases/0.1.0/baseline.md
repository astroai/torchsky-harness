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

1. Merge/rebase `upstream/main`; record a clean candidate commit.
2. Complete the API/evaluation decision matrix; freeze which registrations and
   public names survive.
3. Add the missing IFU/cube evaluator and an HEALPix multi-band gate; do not
   claim either scope until those gates exist.
4. Regenerate purpose-built baselines for the release gates, then enable
   `--require-complete-baseline` there.
5. Migrate harness imports to canonical names before deleting torchsky aliases.
6. Bump versions to `torchsky 0.4.0` and `torchsky-harness 0.1.0`; add the
   publish workflow, wheel install smoke, and runner verdict fields.
7. Run the four release gates on the pinned Linux runner and record the first
   real evidence bundle.
