# Adversarial evaluation and audit playbook

This note records why the scientific release gates did not catch the defects in
the 2026-10-20 `torchsky` modernization audit, and the evaluation/tooling
changes that close that gap. It is a working checklist, not a claim that the
gap is already closed.

## What the gates missed

The declared gates answer a narrow question: *does the subject produce the same
scientific results as a reference implementation on the sampled, declared
cases?* They do not answer:

- whether malformed structural input is rejected instead of interpreted;
- whether invalid physical domains stay invalid;
- whether tensor contracts hold for scalar, empty, noncontiguous, wrong-dtype,
  wrong-device, or gradient-tracked input;
- whether the native path actually ran, rather than silently falling back;
- whether normalization, units, variance algebra, and adjoints are
  scientifically correct rather than merely self-consistent;
- whether high-order/boundary cases are stable;
- whether concurrent calls are reentrant and cache-safe;
- whether the wheel/ABI/ISA build is portable.

Several failure mechanisms made this worse:

1. **Valid-domain masking.** Gates compare only where the reference returns
   finite values, so a subject that invents finite coordinates outside the
   projection domain is invisible.
2. **Shared-formula comparisons.** Native and Python implementations frequently
   translate the same formula. Agreement proves consistency, not correctness.
3. **Aggregate metrics.** RMS/max metrics can hide localized seam, pole, or
   invalid-domain behavior; pole/seam exclusions can hide exactly the failure
   population.
4. **Timing is not correctness.** A consistently misnormalized or biased path
   can still win a benchmark.
5. **Happy-path test corpora.** Green suites exercised representative valid
   arrays, not the contract matrix; some native tests were permanently skipped
   because their import probe referenced a removed module.
6. **Device/backend drift.** Single-machine, single-device runs skip MPS/CUDA,
   CPU-only builds, AVX2/Linux, sanitizer and race coverage.
7. **Deferred hardening.** Audit findings were documented but not immediately
   converted into executable gates; the next subject rework landed before the
   contract layer existed.

## Required layers

Every release must have evidence in all layers below. A scientific gate is not a
substitute for a contract gate, and vice versa.

### 1. Structural contract matrix

For each public/native-routed operation:

- scalar, empty, one-dimensional, N-D, and noncontiguous inputs;
- wrong dtype/device, mismatched paired shapes, aliased outputs;
- NaN/Inf, out-of-domain coordinates, malformed metadata/layouts;
- gradient-tracked inputs and double backward where applicable;
- `TORCHSKY_DISABLE_CPP=1` and forced-native/forced-Python controls;
- expected failure mode: raise, documented sentinel, or finite value.

`benchmarks/bench_native_contracts.py` is the first executable layer for the
native/dispatcher boundary. Every probe runs in a subprocess with a hard
timeout so a segfault, abort, or hang becomes a failed row.

### 2. Independent-oracle domain checks

Do not compare the subject only to a translation of itself. Use, where
applicable:

- WCSLIB/Astropy for projection domains, horizons, poles, seams, H/K variants,
  TPV singularities;
- ERFA for observed-frame directions and time;
- healpy / hpgeom / astropy-healpix / spherical-geometry for sphere geometry;
- analytic values for normalization, conserved flux, areas, variance, and
  likelihood optima;
- brute-force/reference algorithms for indexing and candidates.

Assert **domain agreement**, not only finite-value closeness: if the reference
says invalid, the subject must say invalid (or raise/sentinel as documented).

### 3. Invariants beyond adjoint tests

An operator can pass a transpose test while both directions share a wrong
scale. Add:

- conservation/unit tests (flux, areas, photometric scale);
- analytic DC/impulse/sinusoid tests for FFT paths;
- variance algebra and covariance propagation checks (predictive + measurement,
  squared interpolation coefficients);
- likelihood analytic optima, gradients, masks, reductions;
- pooled vs per-band statistics;
- empty-support NaN semantics, not fake zero error.

### 4. Boundary and scale sweeps

- SHT `lmax = mmax = 256` and high-order sparse coefficients;
- HEALPix nside near the order where `O(nside)` setup becomes pathological;
- projection poles, horizon, antipode, seams, invalid facets;
- TPV near-singular/zero/constant metadata and ill-conditioned Jacobians;
- spherical regions at tiny arcs and coincident/antipodal edges;
- distorted-pixel areas across the field, not a constant-area approximation.

### 5. Concurrency and lifetime

- concurrent mixed-parameter SHT/WCS calls in subprocesses with timeouts;
- cache/refcount invalidation under parallel calls;
- repeated construction/destruction of sessions and operators;
- determinism baselines for repeated concurrent runs.

### 6. Device, build, and packaging

- CPU, CUDA, MPS (where claimed); CPU-only torch build;
- AVX2 and baseline-ISA build/run; Linux and macOS lanes;
- ASan/UBSan/TSan builds for native code; libFuzzer for parsers/layouts;
- wheel/sdist content, linkage, path remapping, licenses, `twine check`;
- installed-wheel import and dispatcher smoke, plus `TORCHSKY_DISABLE_CPP=1`.

### 7. Test-suite meta-audit

- assert that a test claimed to exercise native actually runs native (counter,
  forced path, or disable/raise monkeypatch);
- fail on unexpected skips; list and justify optional-dependency skips;
- run the full map/field corpus, not only the default `pixi run test` lane;
- mutation-test or coverage-check every exported native/dispatcher symbol;
- ensure no test encodes current wrong behavior (normalization as inverse
  variance, singleton sample shapes, permissive sanitizers).

## How this changes release evidence

Add a contract-family claim to `torchsky_harness.evidence`, with required rows
from `bench_native_contracts.py` and, eventually, a randomized differential
domain-fuzz claim. The release criterion becomes:

1. scientific claims established;
2. contract/fuzz claims established (no crash/hang, malformed input rejected,
   domain agreement, tensor/device/grad matrix);
3. device matrix completed or explicitly unimplemented;
4. packaging/ISA checks passed;
5. full subject test corpus green, including map/field and examples.

A skipped comparator, silently skipped native test, missing device lane, or
missing contract row is a coverage failure, not a pass.

## Review process changes

- **Convert every finding into two artifacts:** a minimal regression test for
  the exact bug, and a generalized property/gate that would have caught it.
  A report without an executable gate is not closed.
- **Falsify, don't confirm.** For each directory, start from "where can this be
  wrong?" and derive counterexamples from contracts; do not stop at reachability
  and style review.
- **External-style audits are recurring.** Run an adversarial audit each release
  (or after a large native/rework change) with subprocess crash probes,
  independent oracles, fuzz, sanitizers, and packaging inspection.
- **Red-team the evidence.** Challenge each passing gate: what input population
  is excluded, which backend actually ran, what does the metric hide?
- **Do not defer hardening across a major rework.** The contract layer belongs
  before the rework, not after the next one.

## Skills to build

1. Native-boundary harness engineering (subprocess crash/hang rows, per-export
   manifest, expected-exception matrix).
2. Property-based and differential fuzzing for numerical/domain code.
3. Independent-oracle design (analytic, WCSLIB/ERFA, brute force).
4. Invariant testing for normalization, units, variance, and likelihoods.
5. Sanitizer/fuzz/TSan CI and coverage/mutation tooling.
6. Device/ISA/build portability automation.
7. Test-suite meta-auditing (forced path, skip audit, coverage of exports).
8. Evidence design: claims with scope, limitations, and negative coverage.

## Report-to-gate coverage

The 2026-10-20 audit found defects in four families. The current layer covers
the structural examples but not yet the full population:

| Defect family | Current automated coverage | Remaining gap |
|---|---|---|
| Native raw buffers, empty layouts, malformed metadata | `bench_native_contracts` empty-ring/array, malformed-TPV, noncontiguous and dispatcher probes, each in a subprocess | no sanitizer/TSan lanes, no exhaustive per-export malformed matrix |
| Concurrency and cache lifetime | concurrent spin-SHT probe; scientific spin gates | no repeated cache-eviction, mixed-parameter, or construction/destruction stress gate |
| WCS projection domains and conventions | `bench_domain_fuzz` over 25 projections plus TAN/MOL/HPX/TPV probes; fixed scientific gate | randomized near-seam/pole population and inverse TPV fuzz; CUNIT/epoch/state-preservation fuzz |
| Sphere/HEALPix/MOC semantics | index/angle/scientific gates, generalized H/K fuzz, tiny-arc probe | MOC/STMOC property tests, high-order SHT boundaries, distorted-pixel area sweeps |
| Map normalization, variance, photometric scale | existing map/synthetic gates (valid-domain parity) | analytic DC/impulse FFT normalization, IFU variance, conservative overlap caps, sanitizer and variance-invariant gates |
| Field shapes, precision, vmap, predictive variance, likelihood | existing field/native likelihood gates | scalar/empty/noncontiguous/vmap/dtype matrix and analytic likelihood invariants in harness |
| Dispatcher/autograd/Meta/FakeTensor | dispatcher registration/strided/empty probe | Meta/FakeTensor/compile/autograd rejection contracts in harness |
| Packaging, AVX2/CPU-only build, licenses | package-side scripts and CI | harness-visible wheel/sdist/ISA evidence gate |
| Full-corpus test health | targeted subset passes | bounded lane still hits unmarked slow tests; force-native and skip audits not automated |

The gaps are the next contract/fuzz gates. Until they exist, a green evidence
bundle means "the listed scientific and contract rows passed", not "the audit
families are closed".

## Current artifacts

- `benchmarks/bench_native_contracts.py` — subprocess-isolated contract probes
  for the installed subject (empty/layout, invalid domains, high-order SHT,
  concurrency, tiny arcs, noncontiguous input).
- This playbook.
- `benchmarks/bench_domain_fuzz.py` — seeded randomized differential domain
  check for WCS projections, generalized HPX, and randomized TPV polynomials.
- Next: wire the contract rows into `torchsky_harness.evidence` as a declared
  claim and add the release gate task; add randomized differential fuzz;
  add sanitizer/CI lanes; add packaging checks to harness evidence.
