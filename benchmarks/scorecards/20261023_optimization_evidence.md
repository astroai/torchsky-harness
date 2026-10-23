# Optimization evidence — 2026-10-23

## Identity and evidence

- Torchsky subject: `dcbab25ab408227d0d88a9b0376fb3f3a4e54e36`
- Harness: `1684052785306cda7443873d0ba814b77947719d`
- Subject package-content SHA-256: `b4987fbe3e71593f5748f5ff56ea1cad75359423795049df2ffdebf0bf58ceea`
- Linux CPU campaign: CANFAR `ckb1l5xf`, run `optimization-cpu-20261023-212537`
- Linux CUDA campaign: CANFAR `j8r6bttc`, run `optimization-cuda-20261023-212537`
- CPU artifacts: [gist `31b67eff`](https://gist.github.com/31b67eff05143db674f56267e77947e2)
- CUDA artifacts: [gist `5602acc0`](https://gist.github.com/5602acc02f40428a2d136e719a968c93)

Both successful campaigns evaluated clean installed subjects at the exact commit
above. Local quick evidence also passed the clean-subject guard in the `test`
and `sphere-bench` environments.

## Decisions

| Workstream | Decision | Evidence |
|---|---|---|
| O1 prepared target tiles | Accepted, opt-in bounded cache | CPU 24-exposure 256² tiles: 96 → 4 preparations and 683.7 → 597.5 ms. CUDA 24-exposure 512² tiles: 96 → 4 preparations and 571.8 → 444.1 ms. Capacity zero preserves the uncached path; insufficient capacities use bounded non-evicting admission. |
| O2 PCG synchronization | Deferred | The new controlled CPU/CUDA suites pass known-solution correctness for all 16 rows per device, but no synchronization-policy candidate was implemented. Existing behavior remains unchanged. |
| O3 calibration components | Accepted | One-pass active-node grouping passes deterministic equality, gauge, permutation, and many-component package tests. Linux full evidence retained stable component signatures and row-order behavior; 64-component photometric discovery was 2.61 ms for 6,144 fit rows. Numerical solver changes were excluded. |
| O4 prepared random samplers | Accepted | Containment and statistical checks pass on CPU/CUDA. At 1M BMOC cells, prepared repeated draws were 67,448× faster for 1k CPU samples and 59,500× faster on CUDA; for 100k samples the gains were 2,788× CPU and 56,064× CUDA. Sparse large-footprint rows also amortize preparation, while tiny one-shot cases remain available unchanged. |
| O5 batched triplanes | Rejected for production | Forward and gradient parity were exact, and the CPU candidate won 99/104 timed pairs, but stacking independently stored plane parameters duplicates the full plane storage each forward. Five CPU rows regressed (worst ratio 0.81×), and no complete CUDA triplane artifact was retained. The benchmark remains for a future storage-layout design. |

## Correctness and resource findings

- Map cache outputs matched uncached data, weight, variance, overlap, context,
  and inverse-variance products in bounded package tests.
- Prepared samplers snapshot persistent state, remain device-bound, retain the
  one-shot API, and reject empty or invalid effective distributions as before.
- Component grouping is limited to nodes participating in active fit rows;
  isolated or held-out nodes are not reintroduced.
- Target caching defaults to capacity zero. This avoids implicit multi-GiB
  retention for large flux tiles; users opt into a deterministic entry bound.
- PCG CUDA timings establish a future baseline only. They do not support a
  speedup claim without a synchronization-policy candidate.
- Triplane batching remains a harness prototype, not a torchsky implementation.
- Calibration peak RSS is explicitly unavailable until case-isolated subprocess
  measurement is added.

## Verification

- Torchsky targeted optimization integration: 222 passed, 3 skipped.
- Torchsky final cache/sampler regression set: 250 passed, 3 skipped.
- Torchsky `preflight-push`: passed.
- Torchsky typecheck: 0 errors; one existing `__all__` warning.
- Harness focused optimization tests: 28 passed.
- Harness preflight: passed.
- Exact clean local quick evidence: passed across all six tasks.
- CANFAR CPU and CUDA campaigns: exit code 0 with durable gist artifacts.

## Follow-up

1. Add case-isolated RSS measurement to the calibration benchmark.
2. Prototype PCG convergence/check cadence only after defining host-sync
   counters that are observable on CUDA.
3. Revisit triplane batching only with a parameter layout that avoids runtime
   copies or with a fused kernel; do not reintroduce `torch.stack` of complete
   plane parameters in the forward path.
4. Extend target-cache evidence to SIP/TPV flux-semantics tiled cases before
   changing the default capacity from zero.
