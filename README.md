# torchsky-harness

Scientific validation, upstream parity, performance benchmarks and real-data
comparisons for independently installed torchsky builds. Existing WCS, sphere,
map and catalog evaluators are preserved; future domains have no placeholder
packages. Production code and bounded regressions remain in torchsky.

## Run

Install the subject explicitly, then the harness (both non-editable for the
subject, editable for the pure-Python harness). The subject can be a checkout, an
sdist, or a wheel; it is built with `--no-build-isolation --no-deps` from the
environment's own toolchain, and its identity (version, git revision, dirty flag,
archive sha256, import path) is recorded in every report:

```sh
pixi run -e test install-subject ../torchsky
pixi run -e test install-subject dist/torchsky-0.5.0rc1.tar.gz   # or a .whl
```

The harness never builds its own checkout as if it were torchsky and never puts a
torchsky checkout on `sys.path`. CANFAR campaigns: see `docs/canfar.md`.

```sh
pixi run harness list --domain map
pixi run harness subject
pixi run harness run bench_map_compat --run-dir runs/example -- \
  --size 32 --n-obs 2 --runs 1 --cases synthetic_shifted_wcs \
  --torchsky-kernels bilinear --json-out /tmp/map-compat.json
```

Each run directory contains immutable execution identity and process logs.
Default benchmark outputs are directed into that run's `outputs/` directory;
explicit output arguments retain the selected evaluator's original behavior.
Numerical metrics, backend/device evidence, required comparator gates and
eligibility are provided by the selected evaluator. A zero process exit code
does not certify scientific validity by itself: the record declares
`scientific_validity` only for the release gates whose claims are declared in
`torchsky_harness.evidence`, and every other evaluation stays `not_assessed`.
See [docs/evidence-verdicts.md](docs/evidence-verdicts.md). Collect a release
bundle and mandatory paired-performance verdict with `pixi run release-readiness`.
Do not reuse run IDs.

The `benchmarks/` and `scripts/` trees install under the `torchsky_harness`
namespace, preserving existing file names and colocated baselines. Use
`python -m torchsky_harness.benchmarks.<name>` for direct evaluation execution;
use the CLI when durable execution provenance is needed. No `PYTHONPATH`
setup or torchsky checkout path injection is required.

Large datasets, fitted products and active CANFAR artifacts stay at their
existing durable locations. Historical reports retain their original cohort,
software and calibration conventions; extraction does not rerun experiments.

See `docs/` for migration ownership, verification and existing evaluation
methodology. The 0.1.0 release procedure, gate bundle, and publish workflow are
documented in [docs/releases/0.1.0/release-procedure.md](docs/releases/0.1.0/release-procedure.md). Reusable map scoring and benchmark-row helpers live in
[`torchsky_harness.metrics`](docs/map-metrics.md); they are harness APIs and do
not load through `torchsky.map`. No CANFAR jobs are launched by installation or
verification.

Calibration evidence commands are `torchsky-harness calibration-manifest` and
`torchsky-harness calibration-extract`; see [the evidence guide](docs/calibration-evidence.md).
Experiment notebooks live in [`notebooks/`](notebooks/). The
[second-pass extraction record](docs/second-pass-extraction.md) documents the
remaining code moves, verification, and preserved local artifacts.
