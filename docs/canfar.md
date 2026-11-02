# Running the harness on CANFAR

`scripts/canfar/run_campaign.sh` runs a full CPU or GPU campaign in one headless
session: it builds the **explicit subject** (a torchsky source tarball) into a wheel,
installs the comparator packages, installs the harness, runs the selected suites and
writes a summary. Nothing is read from a torchsky checkout; the subject is imported
from site-packages and its identity (sdist and wheel sha256, optional commit, import
path) is recorded.

## Inputs

| variable | meaning |
|---|---|
| `TSKY_SRC` | torchsky source tarball on `/arc` (required) |
| `HARNESS_SRC` | torchsky-harness source tarball on `/arc` (required) |
| `RUN` | output directory name (default `harness_<UTC timestamp>`) |
| `MODE` | `cpu` or `gpu` (default `cpu`); `gpu` installs the `cu130` torch wheel (CUDA 13, the supported lane) and adds CUDA passes |
| `SUITES` | comma list of `correctness`, `parity`, `performance`, `examples` (default all) |
| `TORCHSKY_ARC_ROOT` | durable root (default `/arc/projects/ots`); output goes to `$TORCHSKY_ARC_ROOT/harness-campaigns/$RUN` unless `OUT` is set |
| `TSKY_COMMIT` | subject commit to record (a tarball carries none) |
| `TORCH_VERSION`, `TORCHFITS_SPEC` | defaults `2.14.0`, `torchfits==1.3.1` |
| `GATE_TIMEOUT` | per-gate timeout, seconds (default 3600) |
| `REQUIRE_MAP_COMPAT` | `1` makes the SWarp/Montage gates required (default `0`: optional; `1` under `improc`) |
| `REQUIRE_TOOLS` | `1` makes the external-binary gates required: SWarp, SCAMP, SExtractor, PSFEx, solve-field, Montage, WCSTools (default `0`; `1` under `improc`) |
| `IMAGE_PROFILE` | `improc` for the `astroai/improc` image; auto-selected when `swarp`, `mProject` and `solve-field` are on `PATH` |
| `ASTROMETRY_INDEX_DIR` | astrometry.net index files (default `/arc/projects/uvickbos/astrometry.net/data`) |
| `TORCHSKY_HARNESS_DE_KERNEL` | DE440s SPK for the ephemeris gate (default: downloaded from NAIF into `$SCRATCH`) |

## Launch

Make the tarballs from the two checkouts and copy them (and this script) to `/arc`:

```sh
git -C ../torchsky archive --format=tar.gz --prefix=torchsky/ -o /tmp/torchsky-src.tar.gz HEAD
git archive --format=tar.gz --prefix=torchsky-harness/ -o /tmp/torchsky-harness-src.tar.gz HEAD
```

CPU campaign:

```sh
canfar create headless images.canfar.net/astroai/base:latest --cores 8 --memory 32 \
  -e TSKY_SRC=/arc/projects/ots/harness-inputs/torchsky-src.tar.gz \
  -e HARNESS_SRC=/arc/projects/ots/harness-inputs/torchsky-harness-src.tar.gz \
  -e TSKY_COMMIT=<sha> -e RUN=campaign-cpu-001 -e MODE=cpu \
  -e SUITES=correctness,parity,performance,examples \
  -- bash /arc/projects/ots/harness-inputs/run_campaign.sh
```

`improc` profile (SWarp 2.41, SCAMP, SExtractor, PSFEx, solve-field 0.97, Montage and WCSTools
are on the image, so their gates are required and fail rather than skip; CPU only):

```sh
canfar create headless images.canfar.net/astroai/improc:latest --cpu 8 --memory 32 \
  -e TSKY_SRC=/arc/projects/ots/harness-inputs/torchsky-src.tar.gz \
  -e HARNESS_SRC=/arc/projects/ots/harness-inputs/torchsky-harness-src.tar.gz \
  -e TSKY_COMMIT=<sha> -e RUN=campaign-improc-001 -e MODE=cpu -e IMAGE_PROFILE=improc \
  -- bash /arc/projects/ots/harness-inputs/run_campaign.sh
```

Python comparators still go into the campaign venv (the system Python's packages are not used).
Binary gates (`test_external_tools_required.py`, `test_wcstools_parity.py`,
`test_montage_projection_parity.py`, `test_map_swarp_optional_parity.py`) were written without
the binaries available locally; the first improc run is their first execution.

GPU campaign:

```sh
canfar create headless images.canfar.net/astroai/base:latest --gpu 1 \
  -e TSKY_SRC=... -e HARNESS_SRC=... -e TSKY_COMMIT=<sha> -e RUN=campaign-gpu-001 -e MODE=gpu \
  -- bash /arc/projects/ots/harness-inputs/run_campaign.sh
```

## Suites

| suite | content | gate |
|---|---|---|
| `correctness` | harness non-slow suite (`not performance/benchmark/slow/integration`); GPU adds `tests/test_cuda_smoke.py` | exit status |
| `parity` | gated comparator suites listed in `scripts/canfar/run_gates.py` (`PARITY_TESTS`): WCS, HEALPix, sphere/MOC/polygons/sparse, catalogs, map resamplers, cosmology, extinction, spectrophotometry, redshift, transforms. `TORCHSKY_HARNESS_REQUIRE_GATES=1` turns a missing comparator into a failure | exit status |
| `performance` | the bounded benchmark gates (`PERFORMANCE_TASKS`) taken from `pixi.toml` (same commands, `pixi run -e X python` rewritten to the venv), JSON under `benchmark_results/`; GPU repeats CPU-device gates with `--device cuda` | exit status |
| `examples` | every `examples/*.py` of the subject, run from a scratch directory with timing | exit status |

## Outputs

`$OUT/summary.md` and `summary.json` (pass/fail per gate, subject identity, comparator
versions, timings, GPU info), `gates.jsonl`, `gates/<gate>.log` (raw logs),
`benchmark_results/` (JSON), `pip-freeze.txt`, `comparator-install.log`, `session.log`.
A comparator that failed to install is reported in the summary as `NOT INSTALLED` and its
gated tests fail rather than skip.

The script exits non-zero if any gate failed. Comparator binaries that are not on PyPI
(SWarp, SCAMP, solve-field, Montage, WCSTools) come from the `improc` image and are optional
elsewhere; MontagePy is installed from PyPI. Comparator versions added by the claims audit are
pinned in `run_campaign.sh` (gwcs 1.0.3, sgp4 2.27, jplephem 2.24, mhealpy 0.3.7, cdshealpix
0.9.0, specutils 2.4.0, torch-harmonics 0.8.0, s2fft 1.5.0 with jax<0.8).
