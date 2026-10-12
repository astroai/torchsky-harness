# torchsky-harness

Scientific validation, upstream parity, performance benchmarks and real-data
comparisons for independently installed torchsky builds. Existing WCS, sphere,
map and catalog evaluators are preserved; future domains have no placeholder
packages. Production code and bounded regressions remain in torchsky.

## Run

Install the chosen torchsky wheel into the desired Pixi environment, then
`pixi run dev` to install this pure-Python harness. The harness never builds its
own checkout as if it were torchsky. A source build must be selected explicitly.

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
does not certify scientific validity. Do not reuse run IDs.

The `benchmarks/` and `scripts/` trees install under the `torchsky_harness`
namespace, preserving existing file names and colocated baselines. Use
`python -m torchsky_harness.benchmarks.<name>` for direct evaluation execution;
use the CLI when durable execution provenance is needed. No `PYTHONPATH`
setup or torchsky checkout path injection is required.

Large datasets, fitted products and active CANFAR artifacts stay at their
existing durable locations. Historical reports retain their original cohort,
software and calibration conventions; extraction does not rerun experiments.

See `docs/` for migration ownership, verification and existing evaluation
methodology. No CANFAR jobs are launched by installation or verification.
