# Torchsky optimization evidence plan

Status: **implementation-ready draft**  
Date: 2026-10-23  
Baseline subject: `torchsky@91a13db1` (`release/0.4.0`)  
Baseline harness: `torchsky-harness@e1d1211` (`release/0.1.0`)

This plan makes `torchsky-harness` the evidence and decision layer for five
larger optimizations identified during the torchsky cohesion audit. Production
algorithms remain in torchsky. The harness defines workloads, correctness
oracles, scaling ladders, resource measurements, promotion gates, and durable
scorecards before production changes are accepted.

## 1. Goals and non-goals

### Goals

1. Measure each suspected bottleneck independently from unrelated model or I/O
   cost.
2. Preserve scientific results, gradients, failure semantics, device behavior,
   and bounded memory while improving survey-scale execution.
3. Require an installed, identity-recorded torchsky subject for every evidence
   run.
4. Use mechanism benchmarks to guide implementation and representative
   end-to-end rows to approve promotion.
5. Leave reproducible JSON artifacts and one compact committed scorecard for the
   accepted before/after comparison.

### Non-goals

- Do not import torchsky implementation files into the harness or inject a
  checkout through `PYTHONPATH`.
- Do not restore process-global image, HDU, WCS, or target caches.
- Do not combine numerical algorithm changes with synchronization, caching, or
  batching changes.
- Do not treat a microbenchmark win as a scientific or product-level claim.
- Do not add SCAMP, TweakReg, mocpy, or healsparse as torchsky production
  backends.
- Do not submit remote jobs or run full scientific campaigns until the quick
  evidence contracts and bounded tests are green.

## 2. Common evidence contract

Every new benchmark row must record:

- subject commit, wheel/package identity, import path, harness commit, platform,
  Python, PyTorch, device, dtype, and thread configuration;
- case parameters, warmup count, repeat count, median and spread;
- peak RSS and peak device allocation when the workload is memory-sensitive;
- correctness metrics, termination/failure status, and a deterministic checksum
  or tensor summary;
- component counters relevant to the optimization, not only total wall time;
- explicit reasons for unavailable metrics or comparators.

Benchmark subprocesses must use `benchmarks/config.py` and the harness runner so
`PYTHONPATH` is removed, `PYTHONNOUSERSITE` is set, and subject identity is
captured. Quick profiles must finish locally; full CPU/CUDA profiles are opt-in.
Raw outputs belong under `benchmark_results/`, `benchmarks_results/`, or durable
`runs/<id>/outputs`, not Git.

### Timing policy

- Warm up the exact workload before measurement.
- Synchronize CUDA/MPS before and after timed regions.
- Interleave baseline and candidate execution where both can coexist safely.
- Report medians and ranges or quantiles, never a single best sample.
- Separate preparation, steady-state execution, and teardown.
- Use subprocess isolation for peak RSS or allocator measurements.

### Correctness policy

Each optimization needs:

1. an independent small reference or the current implementation as a bounded
   oracle;
2. invariant and edge-case checks in torchsky package tests;
3. harness scale/parity checks;
4. an end-to-end row showing unchanged scientific outputs;
5. fail-closed behavior for nonfinite inputs, unsupported devices, incomplete
   comparators, and missing evidence.

## 3. Phase 0: freeze evidence before production changes

Phase 0 is harness-only. It establishes baseline artifacts for all five
workstreams at the subject commit above.

### Deliverables

- Extend `benchmarks/bench_map_image_plane.py` and
  `benchmarks/bench_map_projection_mosaic_suite.py` for target-preparation
  counters and cache-capacity rows.
- Add `benchmarks/bench_solver_pcg.py`.
- Add `benchmarks/bench_wcs_coupled_calibration.py`.
- Extend `benchmarks/bench_sphere_core.py` and
  `benchmarks/bench_sphere_sparse.py` with cold/prepared sampling rows.
- Add `benchmarks/bench_field_triplane_sampling.py`.
- Add quick Pixi tasks for each focused benchmark and one aggregate
  `bench-optimization-evidence-quick` task.
- Add CLI/serialization smoke tests for every new benchmark.
- Document new commands and artifact schemas in `docs/benchmarks.md`.
- Capture baseline CPU artifacts for all workstreams and CUDA artifacts where
  the existing `cuda` environment supports the workload.

### Phase 0 exit gate

- `pixi run preflight` passes.
- Targeted harness tests pass.
- Every quick benchmark records the frozen subject identity and emits complete
  JSON.
- Repeated baseline runs are stable enough to set comparison methodology.
- No production optimization has landed.

Absolute throughput thresholds are not frozen in Phase 0. Correctness,
completeness, resource-bound, and deterministic-counter gates may be frozen
immediately; speed thresholds require representative baseline distributions.

## 4. Workstream O1: bounded prepared target-tile cache

### Hypothesis

Target-tile WCS, cap geometry, pixel area, and quadrature state are invariant
across observations. Rebuilding them per observation/tile causes avoidable
`O(E*T)` preparation and, for flux semantics, repeated `O(E*T*P_tile)` tensor
work. A call-scoped bounded cache can approach `O(T)` preparation without
restoring stale image-data caching.

### Harness implementation

Extend the existing map surfaces rather than creating another projection suite:

- `benchmarks/bench_map_image_plane.py`
- `benchmarks/bench_map_projection_mosaic_suite.py`
- F8 in `docs/proposals/map-tool-parity-2026.md`

Add rows for:

- TAN, SIP, and TPV;
- surface-brightness and flux semantics;
- footprint disabled/enabled;
- tile sizes 256, 512, 1024, and 2048;
- 1, 4, 24, and 128 exposures;
- cache capacities 0, 1, 4, and the full tile schedule;
- CPU and CUDA;
- dense and sparse exposure footprints.

Record:

- preparation calls and preparation time;
- cache hits, misses, evictions, live entries, and peak cache bytes;
- projection time, total time, Mpix/s, peak RSS/device allocation;
- data, weight, variance, context, overlap, and footprint checksums;
- maximum difference from uncached execution.

### Torchsky implementation boundary

Implement only after Phase 0:

- a call-scoped cache owned by coadd orchestration;
- explicit capacity in deterministic tile count or bytes;
- keys covering target grid identity, tile bounds, device, dtype, value
  semantics, area scaling, and quadrature requirements;
- no decoded-image entries and no process-global state;
- capacity zero as the exact uncached control path.

### Promotion gate

- Exact or declared-tolerance equality for all map products.
- Tile-size determinism remains within the existing F8 contract.
- Peak cache memory never exceeds configured capacity plus one in-flight entry.
- Preparation count scales with tiles, not exposures, when capacity retains the
  schedule.
- Representative quick and full map rows improve without regressing sparse
  overlap cases or one-exposure latency materially.

## 5. Workstream O2: PCG synchronization policy

### Hypothesis

Per-iteration tensor-to-host convergence and validity checks serialize CUDA/MPS
execution. A configurable check cadence can reduce synchronization while
preserving breakdown detection, true-residual reliability, and convergence
semantics.

### Harness implementation

Add `benchmarks/bench_solver_pcg.py` with controlled HPD systems:

- diagonal systems with known solutions;
- modest-condition dense and sparse SPD systems;
- matrix-free image operators;
- batched mixed-convergence systems;
- Foam exact normal operators.

Scale state sizes `1e4`, `1e5`, and `1e6`, iteration budgets 8/32/64/128, and
CPU/CUDA. Compare convergence-check intervals 1/4/8/final-only. Keep exact
residual refresh cadence separate from convergence-check cadence.

Record:

- wall time, iterations, operator/preconditioner applications;
- true-residual recomputations and host scalar extractions;
- final true residual and solution error;
- peak memory, termination reason, and breakdown status;
- residual-history enabled/disabled.

Use the existing SMACS linear-grid and field-fit rows only as end-to-end
promotion evidence, not as the mechanism benchmark.

### Torchsky implementation sequence

1. Add synchronization/check policy to the generic solver without changing the
   preconditioner or mathematical iteration.
2. Validate generic tensor and matrix-free paths.
3. Apply an explicit equivalent policy to Foam scalar/custom loops.
4. Apply it to batched Jacobi PCG only after mixed-convergence cases pass.
5. Validate Gauss-Newton and SMACS workflows.

### Promotion gate

- Small systems match the dense reference within current tolerances.
- Final true residual and solution agree with interval-1 control.
- Nonfinite operators, non-positive curvature, and invalid preconditioners still
  fail closed at deterministic check points.
- Mixed converged/unconverged batches remain correct.
- CUDA shows fewer host synchronizations and an end-to-end gain; CPU and
  early-convergence cases do not regress materially.

## 6. Workstream O3: calibration graph component scaling

### Hypothesis

Current component discovery combines per-row scalar extraction with
root-by-vertex scans, producing synchronization overhead and `O(C*V)`
materialization for many disconnected components. Bulk row transfer plus
one-pass root grouping should approach `O((R+V)*alpha(V))`.

### Harness implementation

Add `benchmarks/bench_wcs_coupled_calibration.py` and measure separately:

1. `AstrometricMatchGraph` construction;
2. coupled association validation;
3. photometric component discovery;
4. numerical solve;
5. diagnostics and covariance construction.

Graph families:

- one connected component;
- many equal disconnected components;
- mostly singleton components;
- anchored/unanchored mixtures;
- increasing bands per source;
- shuffled and grouped rows;
- reference rows absent/present;
- held-out exposures and sources.

Scale exposures, sources, rows, bands, and component count independently.
Record construction/component/solve time, peak RSS, rows/s, component size
distribution, gauges, recovered coefficients modulo gauge, and component-label
stability under row permutation.

### Torchsky implementation boundary

- Bulk-transfer active exposure/source/band arrays once.
- Preserve the existing union operation and numerical model initially.
- Replace root-by-all-nodes scans with one-pass root grouping.
- Preserve deterministic component ordering and gauge-exposure selection.
- Keep numerical solver changes out of this workstream.

### Promotion gate

- Exact component membership, exposure membership, anchoring flags, gauge
  choices, and diagnostic ordering on bounded fixtures.
- Solution equality modulo declared gauges.
- Row permutation does not alter scientific results.
- Many-component scaling is near-linear over the harness ladder.
- Add one bounded internal graph-scale row to `astrometry-v1` only after the
  focused benchmark stabilizes; do not attach graph throughput to external
  SCAMP/TweakReg comparator rows.

## 7. Workstream O4: prepared MOC and sparse samplers

### Hypothesis

Repeated sampling rebuilds immutable footprint state: NUNIQ conversion, cell
areas, RING-to-NEST conversion, weight sanitization, and cumulative selection
distributions. An explicit prepared sampler can amortize `O(M)` setup across
batches while retaining the one-shot API.

### Harness implementation

Extend existing sphere suites:

- MOC/BMOC rows in `benchmarks/bench_sphere_core.py`;
- sparse weighted rows in `benchmarks/bench_sphere_sparse.py`.

Separate and record:

- preparation time and storage bytes;
- first cold draw;
- repeated prepared draws;
- amortized time for 1, 10, 100, and 1000 draws;
- samples/s and device allocation.

Scale footprint cells/pixels `1e3`, `1e5`, and `1e6` where feasible, sample
counts `1e3` and `1e5`, mixed MOC orders, BMOC conversion, RING/NESTED, and
weighted/unweighted CPU/CUDA paths.

Add statistical harness checks for footprint containment, mixed-order
area-proportional frequencies, sparse weight frequencies, and all-sky
longitude/`sin(dec)` behavior.

### Torchsky implementation boundary

- Add explicit immutable prepared sampler objects; no implicit cache keyed only
  by mutable tensor identity.
- Bind prepared state to device and dtype.
- Keep `sample_randoms` as the one-shot convenience surface.
- Define mutation/invalidation behavior explicitly.
- Preserve seeded repeatability for a fixed device unless the API explicitly
  documents a sequence change.

### Promotion gate

- Every sample belongs to the source footprint.
- Statistical frequencies pass declared tests.
- Empty footprints, invalid weights, and zero effective mass fail identically.
- Prepared and one-shot device/dtype contracts agree.
- Repeated-draw workloads show amortized gains without a material one-shot
  regression.
- mocpy/healsparse remain references, not runtime backends.

## 8. Workstream O5: batched triplane sampling

### Hypothesis

Each scale launches three or six independent feature-plane samples and rebuilds
coordinate grids. Stacking equal-resolution planes and sampling once per scale
can reduce launch and allocation overhead while preserving feature order and
gradients.

### Harness implementation

Add `benchmarks/bench_field_triplane_sampling.py` covering:

- spatial dimensions 2 and 3;
- 3 and 6 planes;
- one and multiple scales;
- feature widths 4/8/16/32;
- resolutions 16/32/64/128/256;
- points `1e3`, `1e4`, `1e5`, and `1e6`;
- contiguous and broadcast coordinate inputs;
- CPU/CUDA;
- forward-only and forward+backward.

Record points/s, wall time, kernel count when profiler support is available,
peak allocation, forward maximum error, and gradients for every plane and
coordinate input.

Use `benchmarks/bench_map_field_fit.py` for representative end-to-end evidence
across planes, KPlanes, feature-field, heterogeneous, and atlas models.

### Torchsky implementation boundary

- Batch only equal-resolution/equal-feature planes within one scale.
- Preserve the exact current concatenation order.
- Use one batched grid/sample operation per scale where measured beneficial.
- Do not pad or fuse across resolutions in the first implementation.
- Retain a correct fallback for unsupported device/backend combinations.

### Promotion gate

- Forward parity at boundaries and out-of-range coordinates.
- Gradient parity for plane parameters and all coordinates.
- Local-tangent, spherical, multiscale, atlas, and scale-gating paths pass.
- Kernel/launch count falls as designed on CUDA.
- Representative field-fit time improves while product metrics and held-out
  losses remain unchanged within existing tolerances.

## 9. Atomic PR and commit sequence

Keep harness evidence and torchsky production work independently reviewable.
Recommended sequence:

| PR | Repository | Scope | Exit gate |
|---|---|---|---|
| H1 | harness | Common benchmark utilities, JSON schemas, Pixi tasks, CLI smoke tests | preflight + targeted tests |
| H2 | harness | O1 map instrumentation and frozen baseline | quick map artifact complete |
| T1 | torchsky | O1 bounded call-scoped cache | bounded tests + H2 comparison |
| H3 | harness | O2 PCG benchmark and frozen baseline | controlled systems complete |
| T2a | torchsky | Generic PCG check cadence | generic solver tests + H3 |
| T2b | torchsky | Foam/batched PCG cadence | Foam tests + SMACS evidence |
| H4 | harness | O3 graph-scale benchmark | scale ladder complete |
| T3 | torchsky | Bulk extraction and one-pass component grouping | graph equality + H4 |
| H5 | harness | O4 prepared-sampling rows/statistics | cold/repeated baseline complete |
| T4 | torchsky | Explicit prepared samplers | package tests + H5 |
| H6 | harness | O5 triplane microbenchmark/gradient oracle | forward/backward baseline complete |
| T5 | torchsky | Per-scale batched sampling | gradient tests + H6 |
| H7 | harness | Representative promotion runs and dated scorecard | all five workstreams accepted or rejected |

Do not combine T2a and T2b unless the generic policy is already proven. Do not
combine graph algorithm changes with calibration numerical changes. Each
production PR must name the harness baseline artifact and the candidate artifact
used for the decision.

## 10. Verification and promotion workflow

For each H/T pair:

1. Install the exact baseline torchsky artifact into the appropriate harness
   environment.
2. Run the focused quick benchmark and archive raw JSON outside Git.
3. Implement the bounded torchsky change and package tests.
4. Install the candidate artifact; do not evaluate a source checkout directly.
5. Run the same focused benchmark with identical workload and hardware.
6. Run representative end-to-end rows.
7. Classify the result as accepted, rejected, or inconclusive.
8. If accepted, add a stable baseline only where variance supports a meaningful
   gate.
9. Record subject/harness identities and artifact locations in the PR.
10. After all accepted workstreams, commit one dated scorecard and update
    `benchmarks/scorecards/index.md` and `docs/benchmarks.md`.

Local verification for harness changes:

```bash
pixi run preflight
pixi run -e test -- pytest <targeted harness tests>
```

Production verification follows torchsky policy: targeted package tests,
`pixi run preflight-push`, and `pixi run typecheck` when signatures/types change.
Full benchmark suites and remote CANFAR runs remain explicit opt-in promotion
steps.

## 11. Risks and controls

| Risk | Control |
|---|---|
| Cache improves time by retaining unbounded tensors | Hard capacity, peak-byte metric, eviction counters, capacity-zero control |
| PCG removes safety checks with synchronization | Deterministic check points, true-residual gate, breakdown fixtures |
| Graph optimization changes gauges or ordering | Exact component/gauge oracle and row-permutation tests |
| Prepared samplers become stale after tensor mutation | Explicit immutable construction and documented invalidation |
| Batched sampling changes feature order or gradients | Per-plane forward and gradient oracle before timing |
| Laptop microbenchmarks become release claims | Mechanism results stay descriptive; promotion requires pinned representative runs |
| Harness accidentally evaluates a checkout | Installed-subject enforcement and recorded import path |
| Workstreams become one cross-cutting patch | Atomic H/T PR sequence and independent rejection paths |

## 12. Completion definition

This optimization program is complete when every workstream is either:

- **accepted**, with mechanism evidence, bounded torchsky tests, representative
  end-to-end evidence, resource measurements, and a recorded promotion decision;
  or
- **rejected/deferred**, with the measured reason and retained baseline artifact.

The final scorecard must identify the exact subject and harness revisions,
platforms, accepted/rejected variants, correctness status, throughput/resource
changes, and remaining risks. No optimization is considered complete solely
because its microbenchmark is faster.
