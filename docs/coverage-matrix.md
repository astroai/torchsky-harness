# Comparator coverage matrix

Which torchsky functionality is evaluated against which comparator, and whether it is a
release gate. "Gate" means `TORCHSKY_HARNESS_REQUIRE_GATES=1` makes a missing comparator a
failure (`tests/conftest.py`, `RELEASE_GATE_FILES`); the CANFAR campaign sets it. Exhaustive
domain gates are listed in [releases/0.1.0/gate-matrix.md](releases/0.1.0/gate-matrix.md).

## torchsky 0.5 comparator map

Every torchsky primitive area against the full comparator map ("own API, compared in the harness").
Parity = bounded deterministic gate listed in `scripts/canfar/run_gates.py` (`PARITY_TESTS`);
Performance = throughput and speedup recorded (never gated) by `benchmarks/bench_comparator_perf.py`
(`bench-comparator-perf-gate`, JSON `benchmark_results/comparator_perf*.json`, summary table
"Comparator performance") or an older gate named below. "Oracle" = out-of-process comparator
(`tests/_oracle.py`, NPZ exchange, own interpreter).

| Area | Comparator | Parity gate | Performance | Status |
|---|---|---|---|---|
| wcs | astropy.wcs / WCSLIB | `test_wcs_parity.py`, `test_wcs_upstream_astropy.py`, `test_wcs_sip.py`, `test_wcs_tpv.py` (+ `bench-wcs-release-gate`, `bench-wcs-real-headers-gate`) | `bench-wcs-release-gate`; comparator_perf `wcs` (TAN, TAN-SIP; 1e5, 1e6) | release gate |
| wcs | PyAST (starlink-pyast) | `test_wcs_pyast_parity.py` (TAN, TAN pole, SIN, ARC, ZEA, STG, CAR, CAR wrap, AIT, MOL, HPX, CEA, MER, SFL, ZPN, AZP, TAN-SIP, TPV; 1e-4 mas, SIP/TPV 1e-3 mas; 1e-6 px) | comparator_perf `wcs` (PyAST timed inside its oracle) | release gate, Oracle (own venv) |
| wcs | Kapteyn (WCSLIB 4.24) | `test_wcs_kapteyn_parity.py` (same cases minus SIP/TPV) | comparator_perf `wcs` | Opt-in oracle: runs only when `TORCHSKY_HARNESS_KAPTEYN_SOURCE` (pip-installable source) is provided; otherwise skipped and not a required gate (no buildable source, see below) |
| wcs | WCSTools (extra C implementation) | `test_wcstools_parity.py` | - | required on `improc` |
| coords | astropy.coordinates | `test_wcs_frames.py`, `bench_coords_astropy` (`bench-coords-astropy-gate`) | `bench-coords-astropy-gate` | release gate |
| coords | ERFA (pyerfa) | `test_erfa_observed_place_parity.py` | `bench-observed-place` (local task, not a campaign gate) | release gate |
| sphere | healpy | `test_healpix_parity.py`, `test_sphere_compat.py`, `test_synalm_parity.py` | `bench-healpix-release-gate`, `bench-healpix-advanced-release-gate`; comparator_perf `healpix` (ang2pix, pix2ang, neighbours) | release gate |
| sphere | hpgeom | `test_healpix_parity.py` | `bench-healpix-release-gate`; comparator_perf `healpix` | release gate |
| sphere | astropy-healpix | `test_healpix_parity.py` | comparator_perf `healpix` | release gate |
| sphere | healsparse | `test_healsparse_upstream.py` | comparator_perf `sparse` (get_values, ud_grade) | release gate |
| sphere | mocpy | `test_sphere_moc.py` | comparator_perf `moc` (from_circle, union, intersection, contains) | release gate |
| sphere | spherical_geometry | `test_sphere_polygons.py`, `test_sphere_exact_regions.py` | `sphere-bench-polygons` (local task, not a campaign gate) | release gate |
| sphere | s2fft (and healpy SHT) | `test_sphere_s2fft.py`, `test_synalm_parity.py`; healpy SHT in `bench-spin-matrix-release-gate` | comparator_perf `sht` (map2alm, alm2map vs healpy and s2fft), `bench-spin-matrix-release-gate` | release gate |
| map | reproject | `test_map_conservative_external_parity.py`, `test_map_reproject_fft_parity.py` | `bench-map-compat-gate`; comparator_perf `map` (2048, 4096 bilinear and exact) | release gate |
| map | drizzle (STScI package) | `test_map_drizzle_external_parity.py` | `bench-map-drizzle-required` (local) | release gate |
| map | AstroDrizzle (drizzlepac `do_driz`/`cdriz.tdriz`) | `test_map_astrodrizzle_parity.py` (square/point, pixfrac 0.7/1.0, shift/rotate/rescale; 1e-4 abs). The `drizzle` package gate covers the same algorithm re-implemented by STScI; this gate ties torchsky to the original C kernel. AstroDrizzle scales output by the pixel-area ratio (flux conserving), the test divides it out | `bench_map_compat` lists a `drizzlepac` label (local) | release gate, Oracle (own venv) |
| map | dfreproject | `test_map_dfreproject_parity.py` | comparator_perf `map` | release gate |
| map | SWarp | `test_map_swarp_optional_parity.py`, `test_map_external_tools_smoke.py` | comparator_perf `map` (binary wall time), `bench-map-swarp-required` (local) | required on `improc` |
| map | Montage (`mProject`) | `test_montage_projection_parity.py` | comparator_perf `map`, `bench-map-montage-required` (local) | required on `improc` |
| catalogs | astropy SkyCoord matching | `test_catalog_compat.py` | `bench-catalog-release-gate`; comparator_perf `catalogs` (1e5, 1e6) | release gate |
| catalogs | STILTS (`tmatch2`, `tskymatch2`, `tmatch1`) | `test_stilts_parity.py`: synthetic (1e5, RA wrap, pole cap) and **real Gaia DR3 x 2MASS** (`scripts/fetch_real_catalogs.py`); exact pair-set equality | `test_stilts_parity.py` records per-mode timings; comparator_perf `catalogs` (tmatch2, tskymatch2; 1e5, 1e6) | required on `improc` (`REQUIRE_TOOLS=1`) |

STILTS modes and what torchsky 0.5 provides (`tests/test_stilts_parity.py`):

| STILTS | torchsky | Status |
|---|---|---|
| `tmatch2 matcher=sky find=best1` / `best2` | `crossmatch_sky(find="best")` (left to right; swap arguments for `best2`) | pair-set equal |
| `tmatch2 find=all` | `crossmatch_sky(find="all")` | pair-set equal |
| `tmatch2 join=1not2 / 2not1` (and `1and2`) | rows absent from / present in the pair set | counts equal |
| `tskymatch2` | same as `matcher=sky` | pair-set equal |
| `tmatch2 matcher=skyerr` | per-left-row radius `err1 + err2_const` (STILTS criterion: separation <= err1 + err2) | pair-set equal (constant right error only) |
| `tmatch1 matcher=sky action=identify` | pair graph of the self-match; groups are connected components built in the test | same partition; grouping itself is not a torchsky primitive |
| `tmatch2 find=best` (symmetric one-to-one) | not provided by torchsky 0.5 | no equivalent |
| `tmatchn` (multi-catalogue, pairs/group) | not provided by torchsky 0.5 | no equivalent |
| `matcher=sky3d`, `exact`, `skyellipse`, per-right-row errors | not provided by torchsky 0.5 | no equivalent |
| `join=all1/all2/1or2` outer tables | index sets only (no joined table output) | n/a |

Kapteyn and AstroDrizzle installability (checked with CPython 3.13, NumPy 2.5, macOS arm64, 2026-11):
`starlink-pyast 4.2.0` and `drizzlepac 3.11.0` install from wheels (and drizzlepac's `cdriz` runs on 3.13).
`kapteyn 2.3.1` (PyPI sdist) does **not** build on 3.13 / NumPy 2: its `setup.py` replaces the package version
by the compiler banner on macOS, the shipped Cython output targets Python < 3.7, regenerating needs Cython < 3
(NumPy 2 `.pxd` headers need Cython >= 3, Cython 3 breaks the `/` semantics) and the C sources use `NPY_OWNDATA`,
`->flags` and `->base`, removed in NumPy 2. conda-forge ships `kapteyn 2.3` only for Python <= 3.6
(linux-64/osx-64) and the official 3.4 tarball does not build on CPython 3.10-3.13. The campaign therefore provisions Kapteyn only when `TORCHSKY_HARNESS_KAPTEYN_SOURCE` points at a buildable source (the oracle script `scripts/oracle/wcs_oracle.py` is kept for that); the summary reports "not provisioned (no source)" otherwise.

Deferred to 0.6/0.7 (global solvers): SCAMP-like global astrometric/photometric fits and solve-field pipelines.
Only the presence gates (`test_external_tools_required.py`) remain.

## Earlier coverage rows

| Domain | Comparator | Evidence | Status |
|---|---|---|---|
| WCS projections, SIP/TPV, real headers | astropy.wcs | `tests/test_wcs_*.py`, `bench_wcs*` | release gate |
| Coordinate frames | astropy.coordinates | `bench_coords_astropy` | release gate |
| HEALPix geometry, advanced ops | healpy, hpgeom, astropy-healpix | `tests/test_healpix_parity.py`, `bench_healpix*` | release gate |
| Spherical transforms (scalar, spin) | healpy, s2fft | `bench_sphere_spectral`, `bench_spin_matrix`, `tests/test_sphere_s2fft.py` | release gate (s2fft pinned `1.5.0` with `jax<0.8`; s2fft HEALPix needs `L >= 2*nside`, the previous test silently skipped) |
| MOC | mocpy | `tests/test_sphere_moc.py` | release gate |
| Polygons / exact regions | spherical_geometry | `tests/test_sphere_polygons.py`, `test_sphere_exact_regions.py` | release gate |
| Sparse maps | healsparse | `tests/test_healsparse_upstream.py` | release gate |
| Catalog crossmatch | astropy.coordinates | `tests/test_catalog_compat.py`, `bench_catalog_compat` | release gate |
| Resampling / coadd | reproject, drizzle, dfreproject | `tests/test_map_*_external_parity.py`, `bench_map_compat` | release gate |
| SWarp, Montage | SWarp binary, Montage `mProject` | `tests/test_map_swarp_optional_parity.py`, `test_montage_projection_parity.py`, `test_map_external_tools_smoke.py` | **required on the `improc` image** (`IMAGE_PROFILE=improc` sets `REQUIRE_MAP_COMPAT=1`, `REQUIRE_TOOLS=1`); optional on `astroai/base` (binaries absent there). Not verified locally (no binaries on the dev machine) |
| External binaries present and runnable | SWarp, SCAMP, SExtractor, PSFEx, solve-field, Montage, WCSTools; `ASTROMETRY_INDEX_DIR` | `tests/test_external_tools_required.py` | required on `improc` (`REQUIRE_TOOLS=1`) |
| WCS vs an independent C implementation | WCSTools `xy2sky`/`sky2xy` | `tests/test_wcstools_parity.py` | required on `improc`; TAN/SIN/ARC/ZEA/STG/CAR, 1 mas / 2e-3 px |
| ERFA / observed place | pyerfa | `tests/test_erfa_observed_place_parity.py` (`atco13`, `c2i06a`, `pnm06a`, `starpm`), `bench_observed_place` | release gate. Documents the Keplerian Earth-velocity approximation (max 0.56 arcsec vs ERFA, claim "about 1 arcsec") |
| GWCS freeze | gwcs | `tests/test_gwcs_live_parity.py` | release gate (`freeze_live_gwcs` vs live gwcs, sub-mas) |
| TLE / ephemeris / SkyCoord interop | sgp4, jplephem (+astropy, ERFA `epv00`), astropy.coordinates | `tests/test_wcs_optional_extras_parity.py` | release gate; DE440s kernel from `TORCHSKY_HARNESS_DE_KERNEL` or NAIF download. Small DE440s has planet-centre segments only for Mercury/Venus/Earth/Moon |
| Multi-order HEALPix interchange | mhealpy, cdshealpix | `tests/test_multiorder_interchange_parity.py` | release gate. One strict xfail: `from_mhealpy` always reports `values_are_densities=True` (torchsky bug) |
| Spectral rebin / Doppler | specutils | `tests/test_specutils_parity.py` | release gate (`FluxConservingResampler`, `shift_spectrum_to`) |
| CAR spherical harmonics | torch-harmonics | `tests/test_torch_harmonics_parity.py` | release gate (native backend vs the torch-harmonics adapter) |
| Cosmology (distances, H(z), dV/dz) | astropy.cosmology | `tests/test_cosmology_astropy_parity.py` | release gate. Ages are not in torchsky. One strict xfail records python-float precision loss |
| Extinction (CCM89, O94, F99, Calzetti) | dust_extinction, published Calzetti polynomial | `tests/test_extinction_parity.py` | release gate. G23 is not provided by torchsky |
| Synthetic photometry, filter profiles | speclite, synphot | `tests/test_spectrophotometry_parity.py` | release gate |
| Redshift, log-lambda | astropy.units, astropy.constants | `tests/test_redshift_units_parity.py` | release gate |
| Transforms | astropy.visualization, NumPy | `tests/test_transforms_correctness.py` | release gate; two strict xfails record subject deviations (ZScale upper limit, even-count median) |
| Field models / INRs | analytic truth, drizzle/coadd baselines | `bench_map_field_fit`, `tests/test_inr_*` | bounded gates; the SMACS campaigns are evidence, not gates |
| Encoders | none | `torchsky.encoders` | **experimental, not gated**: learned components with no established external reference; covered by bounded subject unit tests only |
| Scene model | none | `torchsky.scene` | **experimental, not gated**: no comparator implements the same contract; exercised through the map/field gates |
| Quality / mask contract | none | `torchsky.quality` | **experimental, not gated**: a torchsky-defined convention, not a comparator-defined one; subject unit tests only |
| Spectral cube readers | specutils | not covered | **not gated**: only rebin/Doppler are compared (above) |
| Astrometric calibration vs peers | SCAMP, solve-field (astrometry.net) | binaries only (`test_external_tools_required.py`) | **deferred to 0.6/0.7 (global solvers)**: presence gates only; `campaigns/phase0` manifests remain manifests |
| Deblending peers, TweakReg | Tractor, scarlet, tweakwcs/jwst | `campaigns/phase0` manifests | manifests only; torchsky makes no parity claim for these |
| SExtractor / PSFEx | source-extractor, psfex | binary presence only | torchsky makes no photometry/PSF parity claim; no comparison suite |
| astroquery SVO filter profiles | astroquery.svo_fps | `tests/test_svo_passband_bootstrap.py` (offline bootstrap) | live SVO query not gated (network) |
