# Comparator coverage matrix

Which torchsky functionality is evaluated against which comparator, and whether it is a
release gate. "Gate" means `TORCHSKY_HARNESS_REQUIRE_GATES=1` makes a missing comparator a
failure (`tests/conftest.py`, `RELEASE_GATE_FILES`); the CANFAR campaign sets it. Exhaustive
domain gates are listed in [releases/0.1.0/gate-matrix.md](releases/0.1.0/gate-matrix.md).

| Domain | Comparator | Evidence | Status |
|---|---|---|---|
| WCS projections, SIP/TPV, real headers | astropy.wcs | `tests/test_wcs_*.py`, `bench_wcs*` | release gate |
| Coordinate frames | astropy.coordinates | `bench_coords_astropy` | release gate |
| HEALPix geometry, advanced ops | healpy, hpgeom, astropy-healpix | `tests/test_healpix_parity.py`, `bench_healpix*` | release gate |
| Spherical transforms (scalar, spin) | healpy (s2fft optional) | `bench_sphere_spectral`, `bench_spin_matrix` | release gate (s2fft: optional) |
| MOC | mocpy | `tests/test_sphere_moc.py` | release gate |
| Polygons / exact regions | spherical_geometry | `tests/test_sphere_polygons.py`, `test_sphere_exact_regions.py` | release gate |
| Sparse maps | healsparse | `tests/test_healsparse_upstream.py` | release gate |
| Catalog crossmatch | astropy.coordinates | `tests/test_catalog_compat.py`, `bench_catalog_compat` | release gate |
| Resampling / coadd | reproject, drizzle, dfreproject | `tests/test_map_*_external_parity.py`, `bench_map_compat` | release gate |
| SWarp, Montage | SWarp binary, MontagePy | `tests/test_map_swarp_optional_parity.py`, `test_map_external_tools_smoke.py` | required only in the map-compat environment (`REQUIRE_MAP_COMPAT=1`); the binaries are absent on the default CANFAR image |
| ERFA / observed place | pyerfa | `bench_observed_place` | benchmark only, not gated |
| Cosmology (distances, H(z), dV/dz) | astropy.cosmology | `tests/test_cosmology_astropy_parity.py` | release gate. Ages are not in torchsky. One strict xfail records python-float precision loss |
| Extinction (CCM89, O94, F99, Calzetti) | dust_extinction, published Calzetti polynomial | `tests/test_extinction_parity.py` | release gate. G23 is not provided by torchsky |
| Synthetic photometry, filter profiles | speclite, synphot | `tests/test_spectrophotometry_parity.py` | release gate |
| Redshift, log-lambda | astropy.units, astropy.constants | `tests/test_redshift_units_parity.py` | release gate |
| Transforms | astropy.visualization, NumPy | `tests/test_transforms_correctness.py` | release gate; two strict xfails record subject deviations (ZScale upper limit, even-count median) |
| Field models / INRs | analytic truth, drizzle/coadd baselines | `bench_map_field_fit`, `tests/test_inr_*` | bounded gates; the SMACS campaigns are evidence, not gates |
| Encoders | none | `torchsky.encoders` | **experimental, not gated**: learned components with no established external reference; covered by bounded subject unit tests only |
| Scene model | none | `torchsky.scene` | **experimental, not gated**: no comparator implements the same contract; exercised through the map/field gates |
| Quality / mask contract | none | `torchsky.quality` | **experimental, not gated**: a torchsky-defined convention, not a comparator-defined one; subject unit tests only |
| Spectral cubes / readers | specutils | not covered | **not gated**: specutils parity is not implemented yet |
| Astrometry / deblending peers | SCAMP, TweakReg, astrometry.net; Tractor, scarlet | `campaigns/phase0` manifests | manifests only; peers unavailable, fail-closed |
