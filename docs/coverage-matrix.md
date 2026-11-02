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
| Astrometric calibration vs peers | SCAMP, solve-field (astrometry.net) | binaries only (`test_external_tools_required.py`) | **solution comparison not implemented**: needs real-sky star data (the packaged MegaPipe fixture is synthetic); `campaigns/phase0` manifests remain manifests |
| Deblending peers, TweakReg | Tractor, scarlet, tweakwcs/jwst | `campaigns/phase0` manifests | manifests only; torchsky makes no parity claim for these |
| SExtractor / PSFEx | source-extractor, psfex | binary presence only | torchsky makes no photometry/PSF parity claim; no comparison suite |
| astroquery SVO filter profiles | astroquery.svo_fps | `tests/test_svo_passband_bootstrap.py` (offline bootstrap) | live SVO query not gated (network) |
