# WCS and sphere migration verification

This note records the geometry side of the torchsky-harness extraction. The
function-level move/retain manifest is `docs/migration-tests.json`; it records
the source snapshot hashes and exact function names. The staged files keep the
existing test module names so upstream provenance stays easy to trace.

## Boundary decisions

Whole evaluator modules staged under `tests/` include the upstream Astropy WCS
fixtures and stress checks, all-sky and cylindrical projection parity, ZPN/AIT
intermediate parity, the upstream HEALPix/HEALSparse/SHT replays, the spin
parity gate, and the exact-region performance gate. Their function IDs are in
the manifest.

Mixed modules were split at test-function boundaries. The manifest lists each
moved evaluator function and each unchanged core test. Key retained contracts
include native HEALPix device behavior and `ud_grade` bad-value handling;
torchsky-native compatibility API checks; native exact-region area and
containment; polygon topology, set operations, and CPU/device agreement; WCS
shape, error, gradient, and device behavior; and deterministic TPV/Jacobian
roundtrips. Small supported-interop checks remain in core, including the
limited spin compatibility smoke, the CSC fallback comparison, and the
frame/ERFA checks.

Three mixed evaluator functions retain only their native assertions in
torchsky tests. Their full oracle comparisons stay in the staged evaluator:

- `test_temporal_ephemeris_corrections_native` checks topocenter identity,
  BARYCENTER applicability, and finite correction output.
- `test_paper4_distortion_lookup_table_roundtrip_native` checks forward/inverse
  roundtrip and post-construction distortion assignment.
- `test_wcs_geometry_helpers` checks pixel scales, area, position angle,
  footprint shape, and pixel-area-map units.

The cylindrical module also contains a small invalid-input API contract. The
Astropy forward/inverse parity tests moved to the harness; core retains
`test_cylindrical_not_implemented` without an Astropy import. Similarly, the
retained native HEALPix regressions no longer carry a module-level skip tied to
healpy, so missing comparator packages cannot suppress core tests. Stale
Astropy imports and comparator-only fixture helpers were removed from the
retained native WCS/sphere modules.

The geometry test and replay files contain no `astropy.io.fits` or `fitsio`
imports. Torchsky-owned FITS access uses torchfits or the supported torchsky
facade; an upstream HEALSparse fixture is written by the reference library to
exercise its interoperability boundary. The staged geometry benchmarks use
package imports through `torchsky_harness.benchmarks`; the WCS/sphere and
HEALPix benchmark modules no longer load torchsky implementation files directly
when normal imports fail. Stale source-root calculations and path-injection
comments were removed from the geometry runners and replays. No geometry runner
mutates `sys.path` or calls `ensure_src_on_path`; the shared benchmark config
still defines that helper without any callers. The local sphere runner is
handled separately by the root task.

## Focused verification

Commands ran from the torchsky checkout, using its already materialized Pixi
test environment with the staged harness installed editable:

```text
pixi run --frozen --no-install -e test -- pytest -q tests/test_wcs_hpx.py tests/test_wcs_sip.py tests/test_wcs_tpv.py tests/test_sphere_compat.py tests/test_sphere_polygons.py tests/test_sphere_exact_regions.py
47 passed, 7 skipped

pixi run --frozen --no-install -e test -- pytest -q /private/tmp/torchsky-harness-20261011/tests/test_wcs_hpx.py /private/tmp/torchsky-harness-20261011/tests/test_wcs_sip.py /private/tmp/torchsky-harness-20261011/tests/test_wcs_tpv.py /private/tmp/torchsky-harness-20261011/tests/test_sphere_compat.py /private/tmp/torchsky-harness-20261011/tests/test_sphere_polygons.py /private/tmp/torchsky-harness-20261011/tests/test_sphere_exact_regions.py
6 passed, 18 skipped

pixi run --frozen --no-install -e test -- pytest -q tests/test_wcs_cylindrical.py::test_cylindrical_not_implemented
1 passed

pixi run --frozen --no-install -e test -- pytest -q /private/tmp/torchsky-harness-20261011/tests/test_wcs_cylindrical.py
8 passed
```

The harness subset's 18 skips are optional sphere comparator cases guarded by
`pytest.importorskip`; the focused run used the core test environment, not the
comparator-specific sphere-bench feature. The checks above verify the retained
core contracts and collection/execution of the staged WCS/sphere subset without
running full suites. A focused `ruff check --select F401,F821` over the edited
WCS/sphere/HEALPix benchmark and replay entry points passed.
