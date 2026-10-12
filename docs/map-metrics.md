# Map evaluation metrics

The `torchsky_harness.metrics` module owns scientific scoring and benchmark
report helpers used by map evaluations. It is available when the harness is
installed and a compatible `torchsky` subject is installed in the same
environment. It does not add a runtime dependency from torchsky back to the
harness.

The module exports these helpers:

- `evaluate_map_product` scores reconstruction, support, and flux metrics for
  predictions against truth.
- `evaluate_map_frontier_regions` reports error by support, overlap, and
  extrapolation regions.
- `evaluate_map_uncertainty` checks how an inverse-variance estimate relates to
  residual magnitudes and supported pixels.
- `field_quality_acceptance_report` applies the scientific acceptance gates
  to field-quality evaluation rows.
- `build_map_benchmark_markdown`, `map_benchmark_row_key`,
  `check_map_benchmark_regressions`, and `write_map_benchmark_artifacts`
  summarize benchmark rows and apply explicit baseline tolerances.

Benchmark scripts import these functions directly from
`torchsky_harness.metrics`. They consume installed torchsky product and WCS APIs;
core runtime code does not import this module. Bounded runtime contracts, such
as uncertainty construction, device behavior, gradients, and operator adjoints,
remain tested in torchsky.

Use the map benchmark commands and baselines documented in
[`benchmarks.md`](benchmarks.md) to run scientific evaluations. Metric
implementations were relocated without changing their calculations or default
tolerances.

These scoring and acceptance-report helpers are harness APIs. Torchsky no
longer exports the evaluator facade alias or `field_quality_acceptance_report`;
the runtime `field_quality_scorecard` diagnostic remains in the core package.
