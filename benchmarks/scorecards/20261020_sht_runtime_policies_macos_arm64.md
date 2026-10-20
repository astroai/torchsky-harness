# SHT runtime policies on macOS arm64 (2026-10-20)

## Recommendation

Keep the current hybrid CPU FFT policy and current SpinMPool default unchanged.
Do not promote forced ATen globally, and do not replace SpinMPool with an ATen
range scheduler from this evidence. The isolated hybrid policy is faster in
geometric mean, but has four representative regressions and high dispersion.
SpinMPool scaling is mixed, and no ATen-range scheduler prototype exists in the
installed subject for a same-binary comparison.

No torchsky runtime edit is justified by this campaign. Before a scheduler or
FFT default change, add benchmark-only runtime selectors for `aten-range`,
`pocketfft-only`, and `accelerate-only`, preserve the existing `hybrid` leaf,
and repeat this matrix on a quiet macOS arm64 host and Linux x86-64.

## Subject and host

- Installed torchsky `0.4.0` wheel:
  `sha256:74a7bc304b2391a8a0d6da5e0dda45c397b3fddeb2b28e940f0506076d5660b8`.
- Installed native extension:
  `sha256:879e2fb38f405b7133799288ae4af9dccaaf0a46b06319053bba4d06137c705e`.
- Installed `sphere/spectral.py`:
  `sha256:2b67babbf538e84b4e976be25ad3f662a77ab14e0ab6de0fa2472e0ff515e18e`.
- Apple M2, 8 physical/logical CPUs, macOS 14.8.2 arm64.
- Python 3.13.14, PyTorch 2.13.0, PyTorch CPU capability `DEFAULT`.
- Selected torchsky ISA leaf: compile-time `aarch64-neon`; runtime ISA
  dispatch is not available.
- Thread matrix: `torch.set_num_threads`, `TORCHSKY_SPIN_THREADS`, and
  `OMP_NUM_THREADS` set to `1`, `4`, or `8`; inter-op threads fixed at `1`;
  `VECLIB_MAXIMUM_THREADS=1` in all rows.

## Method

Six cases cover NSIDE 16/64/128, lmax 12/48/128, spin 1/2, and half/full mmax.
Each policy/thread worker runs cases from small to large. Policy order alternates
by thread setting. Small/medium/large timing samples average 20/3/1 calls to
reduce timer noise.

The FFT campaign uses 3 warmups and 21 measured samples. It compares the current
C++ hybrid policy against the existing forced Python ATen path. The SpinMPool
campaign uses 2 warmups and 9 measured samples. RSS is process RSS plus macOS
`ru_maxrss`. Correctness is relative L2 against forced ATen or healpy, plus the
complex ring-FFT and packed-alm adjoint identities.

No installed selector can force all rings through pocketfft or Accelerate.
Accelerate remains an internal radix-2 leaf of hybrid. No installed ATen-range
scheduler prototype can replace SpinMPool.

## FFT policy evidence

`speedup = forced ATen median / hybrid median`; values above one favor hybrid.

| Group | Analysis geomean | Synthesis geomean |
|---|---:|---:|
| All 18 case/thread rows | 3.895x | 2.201x |
| Small | 6.559x | 2.635x |
| Medium | 3.712x | 2.324x |
| Large | 2.428x | 1.742x |
| 1 thread | 5.693x | 2.343x |
| 4 threads | 1.749x | 1.802x |
| 8 threads | 5.935x | 2.526x |

Hybrid regressed by more than 5% in 1/18 analysis rows and 3/18 synthesis
rows: large `(nside=128,lmax=128,mmax=64,spin=1)` at four threads for both
directions, large full-mmax spin-2 synthesis at four threads, and small
full-mmax spin-2 synthesis at eight threads. This violates the no-regression
condition for a new unconditional default.

Peak-RSS hybrid/ATen ratios ranged from 0.937 to 1.536 with median 1.003.
Correctness maxima were `3.96e-17` analysis relative L2, `1.56e-16` synthesis
relative L2, and `5.62e-14` FFT adjoint relative error.

Dispersion remained high despite batching. Median/max coefficient of variation
was 0.924/1.511 for hybrid analysis, 0.858/1.473 for hybrid synthesis,
0.622/1.455 for ATen analysis, and 0.671/1.370 for ATen synthesis. This is
adequate to reject a global forced-ATen default, but not to introduce crossover
thresholds.

## SpinMPool evidence

Times are medians in milliseconds for 1/4/8 SpinMPool workers.

| Case | map2alm 1/4/8 | alm2map 1/4/8 |
|---|---:|---:|
| small n16 l12 m6 spin1 | 4.564 / 20.563 / 22.059 | 11.348 / 12.589 / 33.029 |
| small n16 l12 m12 spin2 | 9.144 / 7.904 / 14.661 | 27.717 / 21.193 / 27.834 |
| medium n64 l48 m24 spin1 | 28.234 / 55.407 / 29.136 | 38.664 / 103.702 / 133.315 |
| medium n64 l48 m48 spin2 | 75.242 / 80.561 / 32.211 | 171.815 / 64.472 / 123.133 |
| large n128 l128 m64 spin1 | 88.094 / 47.344 / 169.678 | 173.784 / 122.878 / 240.093 |
| large n128 l128 m128 spin2 | 157.694 / 112.530 / 85.798 | 367.992 / 100.132 / 293.454 |

Relative to one worker, four workers have geometric-mean speedups of 0.826x
for map2alm and 1.351x for alm2map. Eight workers have 0.808x and 0.708x.
Scaling changes sign by case, so neither a one-worker cap nor the current
machine-maximum policy is uniformly supported.

Spin correctness maxima were `9.46e-14` map2alm relative L2 vs healpy,
`1.60e-14` alm2map relative L2 vs healpy, and `1.72e-13` adjoint relative
error. Peak RSS ranged from 254.7 to 345.3 MiB; the maximum timing-phase RSS
delta was 15.5 MiB. Median/max coefficient of variation was 0.653/1.261 for
map2alm and 0.546/0.746 for alm2map.

## Artifacts

- FFT raw JSON: `benchmark_results/sht_fft_policies_macos_arm64_20261020_v3.json`,
  sha256 `e633d3c3f1109538a267d73c824e67876162b992c2e8858baf92a0cb635a73a8`.
- SpinMPool raw JSON:
  `benchmark_results/sht_spinmpool_macos_arm64_20261020_v3.json`, sha256
  `d9584189600e4816543ed7c735f56ea42feb01ab5e6259dc6360a06360513597`.
- The hashes above are updated in the companion JSON scorecard when raw runs
  are regenerated.
