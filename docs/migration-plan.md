# Plan: extract scientific evaluation into torchsky-harness

Status: approved; extraction implemented in a separate local repository.
Inventory date: 2026-10-11.

## Purpose and ownership

Create `torchsky-harness` as the home for torchsky's scientific benchmarks,
reference-library parity, acceptance validation, real-data experiments, and
comparison reports. Use the Python package name `torchsky_harness` and an
installed `torchsky-harness` CLI.

Torchsky remains the implementation library. The harness consumes identifiable
torchsky builds and evaluates them. Torchfits continues to own FITS/table I/O
and its I/O correctness and performance tests. No runtime dependency from
torchsky to its harness is introduced.

Keep a small, reliable test suite beside the implementation. Move exhaustive
parity and scientific evaluation into the harness. Small regressions may retain
reviewed reference vectors with provenance, without requiring a legacy library
to calculate those values on every package test run. Optional interoperability
features still need their own bounded package tests.

The current runtime dependency list is already small: torch, numpy, torchfits.
The extraction primarily reduces checkout clutter, contributor environments,
CI complexity, experiment maintenance, and ambiguity in scientific reports.

## What moves and what stays

| Surface | Torchsky ownership | Harness ownership |
| --- | --- | --- |
| Runtime | Native Python/C++/Torch kernels, public APIs, operators, scenes, field models, production compatibility adapters | Evaluation adapters that call those APIs |
| Package correctness | API contracts, invalid-input behavior, small deterministic regressions, geometry invariants, adjoints, gradients, serialization contracts, device semantics | Broad upstream parity matrices, large randomized sweeps, independent scientific validation |
| Performance | A bounded smoke check for catastrophic regressions where useful | Throughput, scaling, memory, latency, compiler/backend comparisons, hardware-specific acceptance baselines |
| External tools | Explicitly supported production interop remains opt-in | Astropy, healpy, hpgeom, healsparse, mocpy, spherical-geometry, Reproject, Drizzle, SWarp, Montage and other reference adapters |
| Data | Small fixtures needed by retained package tests; required runtime data | Upstream fixtures, acquisition/staging tools, synthetic generators, injection protocols, calibration manifests, real-data campaigns |
| Reports | Brief benchmark methodology and links to verified evidence | Baselines, scorecards, coverage matrices, experiment history, plots, claim-to-evidence index |
| Infrastructure | Build system, wheels, package release CI, developer preflight | Comparator environments, evaluation CI, CANFAR runners, job tracking and result ingestion |

Move scripts by their role, rather than their directory name. An application
helper used by a supported torchsky CLI may stay or become a proper package
component. A downloader or calibration driver used only by SMACS evaluation
moves with that campaign. Moving a benchmark helper does not justify adding
its private functions to torchsky's public API.

## Inventory and extraction hazards

The inventory identified about 85 benchmark Python files and 210 test modules.
Filename counts describe repository size, not scientific coverage. The broad
families already include WCS, sphere/HEALPix, map/field reconstruction, and
catalog/crossmatch evaluation.

Important migration units are:

- `benchmarks/bench_wcs*.py`, `bench_sphere*.py`, `bench_healpix*.py`,
  `bench_spin_matrix.py`, and their upstream replay helpers.
- `benchmarks/bench_map*.py`, `bench_multiband_depth.py`,
  `score_field_models.py`, SMACS/UPDC scorers, and `benchmarks/updc/`.
- `benchmarks/bench_catalog*.py`, `scripts/run_catalog_scale_contract.py`,
  and `scripts/run_catalog_scale_real.py`.
- `benchmarks/baselines/`, `benchmarks/scorecards/`, benchmark documentation,
  comparator bootstrap/version scripts, and CANFAR launch/fetch scripts.
- Comparator and experiment tests that import those modules: for example
  `test_map_compat_bench.py`, `test_multiband_depth_benchmark.py`,
  `test_score_field_models.py`, SMACS experiment tests, and upstream replay tests.

Several files mix package assertions with benchmark assertions. Split those
functions before moving the harness portions; do not relocate every
`test_map_*`, `test_wcs_*`, or `test_sphere_*` file wholesale.

Three current mechanisms must be replaced during extraction:

1. `benchmarks/config.py` and many tests insert the repo's `src` or benchmark
   directory into `sys.path`; workflows also use `PYTHONPATH=src:benchmarks`.
2. `scripts/run_local_sphere_bench.py` assumes it can build the repository
   containing the script. The new runner must build an explicitly selected
   torchsky source checkout, or install an explicitly selected wheel.
3. Some benchmark code falls back to loading a torchsky implementation file
   directly when normal imports fail. Harness runs must fail import/build
   validation instead of silently changing the subject being measured.

Preserve upstream licenses, source revisions, fixture provenance, and reviewed
tolerances. Existing FITS-I/O policy debt should be migrated to torchfits calls
or tracked explicitly; do not copy forbidden FITS imports into the new repo.
The current CFITSIO-related build flags are a separate ownership audit, not a
reason to move torchsky's native build into the harness.

## Proposed structure

```text
torchsky-harness/
  pyproject.toml
  pixi.toml / pixi.lock
  AGENTS.md
  src/torchsky_harness/
    cli.py
    subject.py                 # explicit wheel/source build identity
    registry.py                # suites, capabilities, required comparators
    runner.py                  # local execution and result publication
    schemas/                   # versioned run/case/metric records
    adapters/                  # torchsky and external reference backends
    metrics/                   # shared evaluation code, not library kernels
    datasets/                  # manifests, acquisition and fixture access
    suites/
      wcs/
      sphere/
      map/
      catalogs/
      cross_domain/
    reports/
    backends/                  # local and CANFAR execution
  campaigns/smacs0723/          # pinned data/calibration/model/run configurations
  contracts/                   # reviewed tolerances and selection policies
  fixtures/                    # small licensed fixtures and manifests
  baselines/                   # reviewed numerical/performance expectations
  scorecards/                  # compact published evidence
  tests/                      # tests of the harness itself
  docs/
  .github/workflows/
```

Package existing evaluation code before designing new abstraction layers.
Use ordinary Python suite functions and small TOML configurations. Share only
recurring concerns such as subject identity, result schemas, artifact access,
units, timing, and comparator execution.

Use named Pixi features for each reference stack and device. Incompatible
legacy Python environments run as separate processes exchanging arrays and
metadata; they must not force an incompatible Python version on torchsky.

## A result must identify what was actually evaluated

Every result records:

- Harness revision, torchsky revision/version, wheel or source-snapshot hash,
  native build flags, dirty-state provenance, and the actual imported package path.
- Adapter/backend identity: native torchsky, external reference, or published
  product; implementation version and executable/container identity.
- Requested and actual execution device, dtype, hardware, thread settings,
  and any host transfer or device fallback.
- Dataset/fixture IDs, input and calibration hashes, target geometry, units,
  weighting and mask semantics, PSF conventions, seeds, and train/validation/test splits.
- Exact configuration and command, resource request, execution outcome,
  metric availability, scientific eligibility, and artifact locations/hashes.
- Timing boundaries, warmup/repetition policy, synchronization for GPU timings,
  and whether transfers/build/startup are included.

Execution success, scientific validity, and comparison eligibility are separate
fields. A completed process with a skipped Reproject row is not Reproject
coverage. A successful signed-response audit with nonpositive injection totals
does not provide a valid effective PSF. Required comparators missing from an
acceptance run fail that coverage gate; optional omissions remain visible.

Represent unavailable metrics as null with a reason and audit trail. Never
replace missing values with zero, clip negative responses to manufacture a PSF,
or quietly omit failed methods from the coverage table. Historical artifacts
remain immutable; schema migration produces derived records with lineage.

Maintain a capability matrix by operation, semantics, backend, and device.
Implementing native Lanczos resampling, Drizzle accumulation, or conservative
reprojection does not by itself establish complete SWarp/Montage/Reproject
workflow compatibility. Each claimed behavior links to an explicit contract
and passing evidence.

## Evaluation suites and evidence levels

| Domain | Principal evaluation contracts |
| --- | --- |
| WCS | Projection/frame/distortion parity, inverse/round-trip accuracy, singularities, gradients, CPU/CUDA/MPS agreement, small-to-large batch scaling |
| Sphere | HEALPix indexing/interpolation, geometry and exact region behavior, sparse/MOC operations, scalar/spin transforms, upstream parity, high-resolution scaling |
| Map | Resampling/stacking semantics, support/weights, operator adjoints, flux and surface-brightness units, reconstruction, response PSFs, correlated noise, tiled/streamed execution |
| Catalogs | Candidate/index correctness, matching semantics, Arrow/tensor interchange, partitioning, sparse/dense and large-scale crossmatch |
| PSF | Normalization, encircled energy, sampling/pixel response, chromatic/spatial behavior, forward/adjoint parity, recovery on simulated and observed stars |
| Detection | Completeness/purity at a fixed false-alarm policy, injection recovery, flux/position bias, crowding/blends, uncertainty calibration, independent held-out fields |
| Segmentation | Label/instance conventions, IoU/Dice and boundary quality, deblending, masking, multiband correspondence, ambiguous-label handling |
| SSL | Spatial/exposure-disjoint data splits, leakage controls, fixed downstream probes, missing-modality and domain-shift evaluation, seed variation, compute budgets |
| Cross-domain | WCS↔sphere↔map, catalog/frame state, multiband observation operators, and later PSF→detection→segmentation pipelines |

Existing PSF operator tests stay with their current map ownership. Do not create
PSF, detection, segmentation, or SSL directories during this extraction. Add
those domains only when there are runnable suites to move; generic encoders or
pretraining experiments do not establish their evaluation capabilities.

Distinguish analytic/synthetic truth, reference parity, real held-out validation,
published-product agreement, and performance evidence in reports. A static
published UPDC image can support image agreement and proxy sharpness; it cannot
supply an injection-derived SNR or effective PSF without the generating pipeline.

For SMACS, publish one method/configuration-by-priority table for SNR/recovery,
resolution, reconstruction, artifacts, and flux, with backend/device and metric
availability visible. Keep raw six-band, calibrated two-band, synthetic-truth,
and published-static cohorts separate. Freeze the reference stack identity.
Require matching units, exposures, calibration, support, and response conventions
before comparing rows. State fixed training budgets and convergence evidence.

## Data, artifacts, and reproducibility

Keep small fixtures and reviewed contracts in Git. Keep large JWST/catalog
datasets, source snapshots, fitted products, and full run artifacts in durable
external storage. Commit manifests containing locations, hashes, licenses,
acquisition recipes, and access requirements; use torchfits at FITS boundaries.

Adopt one logical artifact layout, `runs/<run_id>/`, with immutable raw outputs
and regeneratable reports. Existing CANFAR locations remain valid through
manifest references; moving the harness does not mean moving the data.

A persisted run registry and immutable job specifications replace chat history
as experiment state. Publish reports atomically, validate completeness/schema
before consuming them, and distinguish missing, partial, failed, and completed
products. Coordinators can recover after desktop or session failure without
retraining or overwriting an earlier run.

Historical configurations remain distinct records. Derived summaries link to
their source hashes and report-generation revision. Do not promote historical
unpaired or calibration-mismatched results into current acceptance baselines.

## CI and developer workflow

Keep torchsky's lint/type/build/wheel and bounded package test gates. Keep the
pre-push gate as preflight only. Full local validation remains opt-in.

Harness tiers:

1. `smoke`: offline, small, deterministic installed-package checks and harness tests.
2. `acceptance`: required reference matrices and cross-domain contracts.
3. `scale-device`: scaling, memory, GPU/MPS parity and hardware-pinned performance.
4. `scientific`: real-data campaigns, injections, held-out validation and model sweeps.

For torchsky PRs, build the candidate artifact and run a bounded external
harness smoke at a pinned harness revision. Use reviewed harness workflows for
weekly, GPU, large-scale and scientific runs. Harness PRs test themselves against
a pinned released subject plus an explicitly selected development artifact.
Cross-repository changes need compatible harness revisions and coordinated pins;
avoid requiring simultaneous merges to recover either repository's CI.

Representative proposed interface:

```text
pixi run harness -- list --domain map
pixi run harness -- run --suite map.compat --profile smoke --subject-wheel <wheel>
pixi run harness -- run --suite map.smacs0723 --campaign <config> --backend canfar
pixi run harness -- report --run <run_id>
```

These commands are design targets, not existing runnable commands.

## Migration sequence

| Stage | Work | Completion criterion |
| --- | --- | --- |
| 0. Inventory and freeze | Classify files and mixed tests; capture current gate IDs, output contracts, baselines, fixtures, licenses, dependency and private-import graphs; record active SMACS runs | Every existing evaluation surface has an owner and migration destination; old run artifacts remain interpretable |
| 1. Establish the harness | Create package/Pixi/CI/schema/subject installation; port one small WCS and one sphere contract | Runs from an unrelated working directory against an installed wheel, with no torchsky checkout path injection |
| 2. Map and SMACS pilot | Move map comparator/depth/PSF-response/scorer helpers with their tests, dataset manifests, CANFAR control and reports; preserve frozen protocols | Replay a small frozen cohort; numerical results/gates agree or each intentional correction is documented; backend/device/availability table is generated |
| 3. WCS and sphere | Move full suites, upstream replays, comparator environments, compact baselines and historical scorecards | Required CPU acceptance retains coverage; CUDA/MPS and scale work have explicit profiles and device evidence |
| 4. Catalogs and integration | Move catalog scale/compat/real-data workloads and cross-domain suites; consolidate duplicate report/timing/provenance helpers | Pairing/indexing contracts and representative large-data results are preserved; cross-domain interfaces are covered |
| 5. Switch CI and slim torchsky | Transfer weekly/scientific workflows; pin external smoke; remove migrated files, benchmark tasks and evaluation-only extras after imports are repaired | Core builds/tests independently; no core test imports harness-private code; no dropped required gate or broken documentation link |
| 6. Extend evaluation | Add initial PSF suite and documented future detection/segmentation/SSL contracts | New implementations register cases under the same provenance, data-split and metric-availability rules |

Use bounded migration PRs, preserving file history where practical. Move a
runner/helper and its tests together. Avoid mixing extraction with numerical
algorithm changes; evaluate any scientific corrections separately with before/
after evidence. Keep temporary forwarding commands only during migration and
give them a removal milestone.

## Definition of done and immediate recommendation

The split is complete when torchsky builds and passes its retained tests without
benchmark directories or reference-tool environments; the harness evaluates a
wheel from a clean independent checkout; every migrated required gate has an
equivalent named contract; and a report can be regenerated from durable inputs
without recovering a desktop conversation.

Start with the file/contract inventory and installed-wheel runner, then the
SMACS map pilot. That gives an early usable comparison table while the WCS,
sphere, catalog and cross-domain suites migrate in controlled groups.

Existing CANFAR fits and scoring coordinators keep their frozen source and
persistent artifact paths. Their compact specifications, ledger and reports
were preserved in the harness campaign history. No comparison jobs were launched
or retired for the extraction, and the prepared Reproject comparator remains
unlaunched. No future-domain placeholder packages were created.
