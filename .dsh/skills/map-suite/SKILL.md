---
name: map-suite
description: Run torchsky-harness map scientific evaluations, acceptance gates, and SMACS0723 comparisons for an independently installed subject
metadata:
  project: torchsky-harness
---

# Map suite (tests + benches)

Run these evaluation commands from the torchsky-harness checkout with an
explicitly installed torchsky subject. Record both revisions. Runtime map
contracts remain in torchsky; scientific map evidence is owned by this repo.

## Map tests are part of the default suite

`pixi run test` runs the full `pytest tests -q` tree in the test env, so map tests
are included and the default suite is expensive. For focused loops run the map
files explicitly:

- `pixi run -e test -- pytest tests/test_map_*.py`
- or via the acceptance gates below.

## Map gates

- `pixi run bench-map-acceptance-smoke` — ground-truth, multifield, field-fit,
  compat smoke gates vs `benchmarks/baselines/map_*_smoke.json`
- `pixi run bench-map-acceptance` — rep gates vs
  `benchmarks/baselines/map_*_rep.json`
- `pixi run bench-map-compat-external` — optical/IR stacker comparators
  (reproject, drizzle, Montage, SWarp, casutools). Bootstrap external tools
  first: `pixi run bootstrap-map-compat-tools` (or `--tool swarp|montage|casutools`).
- Foam / operator / spectral / training: `pixi run bench-map-foam-*`; field
  fit: `pixi run bench-map-field-fit`; projection/mosaic and TAN-SIP suites:
  `pixi run bench-map-projection-mosaic-suite` and `pixi run bench-map-tan-sip-suite`.

## Real-data repro (SMACS0723)

`benchmarks/bench_map_smacs0723_*.py` reproduce field-fit / stacker / ladder /
joint results on real JWST SMACS0723 data. Setup scripts in `scripts/`:
`download_jwst_smacs0723_mast.py`, `download_smacs0723_multiband_public.py`,
`build_smacs0723_stack_baseline.py`, `smacs0723_full_mosaic_pipeline.py`,
`visualize_smacs0723_products.py`, `download_updec_webb.py`. Heavy artifacts go
to `data/benchmark_results/` (not committed). CFHT Megacam MEF mosaics:
`scripts/bench_cfht_mef_mosaic.py`.
