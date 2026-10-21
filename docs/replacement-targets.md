# Replacement and parity targets

This is the working inventory of external packages whose user-facing
functionality torchsky (or its sibling `torchfits`) is intended to replace,
complement, or interoperate with. It separates production replacement targets
from optional adapters and from packages that are only references; a package
listed here is not automatically a release claim.

Status vocabulary:

- **gated** — a declared release claim compares this package on the listed
  behaviour.
- **covered** — a benchmark or acceptance task exists, but it is not in the
  release claim set.
- **unit-only** — torchsky has bounded package tests, no cross-package
  benchmark.
- **gap** — no meaningful cross-package comparison yet.
- **interop** — coexistence/adapter, not a replacement claim.
- **owner: torchfits** — FITS/table implementation lives in the sibling repo.

## 1. FITS and tables

| User-facing package | Current torchsky/torchfits surface | Status | Evidence |
|---|---|---|---|
| `astropy.io.fits` | `torchfits` HDU/image/table I/O | owner: torchfits | torchfits owns FITS/HDU/table parity and throughput |
| `fitsio` (python-fitsio) | `torchfits` | owner: torchfits | do not import in torchsky |
| CFITSIO | `torchfits` backend | owner: torchfits | package-level |
| `astropy.table` (FITS tables) | `torchfits.table`, `torchsky.catalogs` batches | partial | torchfits owns table codec; torchsky owns catalog semantics |
| pandas / polars / PyArrow / DuckDB | Arrow/tensor batches at boundaries | interop | catalog tests and `bench_catalog_crossmatch_arrow`; not replacements |

## 2. WCS, astrometry, coordinates, time

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `astropy.wcs` / WCSLIB | `torchsky.wcs.WCS` | **gated** | 38-projection release gate, real instrument headers, domain fuzz |
| PyAST (`starlink.Ast`) | `torchsky.wcs.WCS`, `convert_celestial`, `plan` | covered (legacy env) | `bench-wcs-legacy`/`bench-wcs-ecosystem`; not a release claim; env currently lacks harness |
| Kapteyn | `torchsky.wcs.WCS` | covered (legacy env) | legacy comparator; not a release claim; optional install |
| `gwcs` | `torchsky.wcs.sample_gwcs*`, `gwcs_live` | unit-only | build-time sampling/live adapter tests; no harness parity |
| `astropy.coordinates` / `astropy.time` / `astropy.units` | `torchsky.coords`, `torchsky.transforms`, `torchsky.wcs.frames` | **gated** (geometric frames) | `bench-coords-astropy-gate`: ICRS/Galactic/FK5/ecliptic <1e-6 asec; time/units remain unit-only |
| `pyerfa` / ERFA | `torchsky.wcs.iau2006`, observed frames | covered | `bench_observed_place` compares CIRS/TETE/AltAz against ERFA; not a release claim |
| SOFA | same IAU 2006/2000A path | unit-only | no direct SOFA binding comparison |
| `jplephem` | `torchsky.wcs.ephemeris` adapter | unit-only/interop | optional extra; no parity gate |
| `sgp4` | `torchsky.wcs.tle` adapter | unit-only/interop | optional extra; no parity gate |
| `skyfield` | none | gap | not a target; possible future coordinate/ephemeris comparison |
| `astrometry.net` (`solve-field`), SCAMP, `drizzlepac.tweakreg` | `torchsky.wcs.solve_astrometry`, `fit_sky_affine` (partial) | gap | scripts exist for calibration, no parity campaign |

## 3. Sphere, HEALPix, MOC, harmonics

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `healpy` | `torchsky.sphere.healpix`, `spectral`, `compat` | **gated** | index/angle/advanced/spin gates, real-sky map gate |
| `hpgeom` | `torchsky.sphere.healpix` queries | covered | `bench_sphere_geometry*`, ecosystem matrix; not release-gated |
| `astropy-healpix` | HEALPix core and WCS HPX | covered | ecosystem benchmark/upstream sync; not release-gated |
| `healpix` (C++/Python) | HEALPix core | covered | ecosystem benchmark; package missing in current sphere env |
| `cdshealpix` (Python/Rust) | `sphere.healpix`, MOC/STMOC | gap | mentioned in archived matrix; no harness comparator |
| `healsparse` | `torchsky.sphere.sparse` | covered | upstream fixtures/sync; no release gate |
| `mocpy` | `torchsky.sphere.MOC`, `BMOC`, `STMOC` | covered | `bench_sphere_core --moc-only`; not release-gated |
| `mhealpy` | multi-order HEALPix maps | gap | ecosystem benchmark entry; package missing |
| `spherical_geometry` | `torchsky.sphere.geom` | covered | polygon/area comparisons; not release-gated |
| `s2fft` | `torchsky.sphere.spectral` peer | covered (optional) | `sht-bench` extra; not currently installed/release-gated |
| `ducc0` / `ducc0.sht` | `torchsky.sphere.spectral` | gap | no comparator wired |
| `pixell`/`enmap` | WCS map + SHT | gap | no CAR-native SHT comparison |
| `NaMaster`/`pymaster` | pseudo-C_l, mode coupling | gap | not implemented as a target yet |
| `cuhpx` | CUDA SHT spike | covered (optional) | CANFAR `cuhpx` mode rc=0; not a release claim |

## 4. Mapmaking, reprojection, coaddition

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `reproject` | `torchsky.map.resample/coadd` | **gated** | real CFHT cutout gate |
| `drizzle` (LSST) | `ResampleConfig(kernel="drizzle*")` | covered | `bench-map-drizzle-required`; not release-gated |
| `drizzlepac` / AstroDrizzle | drizzle/blot/outlier/context paths | gap | no comparator gate; package present in some envs |
| `dfreproject` | DEIMOS-style reprojection | covered | `bench_map_compat`/bench env; not release-gated |
| Montage (`MontagePy`, `mProject`, `mFitplane`) | exact spherical overlap + sky matching | covered | `bench-map-montage-required`; `MontagePy` missing locally |
| SWarp | resampling/coadd/background/`.head` | covered | `bench-map-swarp-required`, clipped variant; not release-gated |
| casutools/imcore | aperture/resample compatible operations | gap | bootstrap script only |
| `stcal` | variance/outlier/cube build | gap | JWST pipeline not compared |
| `jwst.cube_build` | IFU cubes | gap | useful real-data comparator |
| T-PHOT / SCARLET / Tractor | forward/deblended photometry | gap | planned surfaces |
| MaxiMask / WeightWatcher | defect/weight masks | gap | `torchsky.quality` target, no comparator |

## 5. Detection, photometry, PSF, calibration

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `photutils` | `torchsky.catalogs`, `torchsky.map.psf` (partial) | gap | used in SMACS scripts, no parity gate |
| SExtractor / SEP | source detection, catalogs | gap | scripts only; no numerical comparison |
| PSFEx | PSF modelling | gap | no comparator |
| `stpsf` / WebbPSF | JWST PSF | gap | SMACS feature; no parity gate |
| `poppy` | optical PSF | gap | no comparator |
| SCAMP / `tweakreg` | astrometric bundle | gap | calibration scripts, no parity campaign |

## 6. Spectral, photometric, cosmology, extinction

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `astropy.cosmology` | `torchsky.cosmology`, `redshift` | unit-only | package tests; no harness comparator |
| `astropy.units` spectral equivalencies | `torchsky.spectral`, `spectrophotometry` | partial | unit tests; no astropy equivalency parity |
| `synphot` / `pysynphot` | filter integration, bandpasses | unit-only | no comparator |
| `speclite` | filter response integration | unit-only | no comparator |
| `specutils` | spectral axes/line analysis | unit-only | no comparator |
| `astroquery` SVO FPS | filter profile lookup | unit-only/interop | fake/offline tests; no live parity |
| `dust_extinction` | `torchsky.extinction` | unit-only | package tests; no harness parity |
| `dustmaps` | extinction maps | gap | no wrapper yet |
| `sncosmo` / `sedpy` / `prospect` | SED/spectral modelling | gap | not targets yet |

## 7. Catalogs and cross-matching

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `astropy.table`/`SkyCoord` matching | `torchsky.catalogs.crossmatch` | **gated** | catalog compatibility gate vs SkyCoord |
| `scipy.spatial.cKDTree` | spatial index | reference | used in tests/benchmarks |
| `xmatch` (CDS) | crossmatch engine | gap | scorecard only; package not installed |
| `duckdb`/`polars`/PyArrow/pandas | table ingestion | interop | catalog I/O tests; not replacements |

## 8. Field models, rendering, visualization

| User-facing package | Current torchsky surface | Status | Evidence / gap |
|---|---|---|---|
| `tiny-cuda-nn` / `tinycudann` | hash-grid encoders | gap (research) | no direct comparator; field models validated on truth |
| `nerfstudio`, `torch-ngp`, `gsplat` | neural/radiance fields | gap (research) | not applicable to survey maps yet |
| `scipy.interpolate` / `scipy.ndimage` | continuous field/interp baselines | unit-only | no harness comparator |
| `matplotlib` | visualization | interop | examples/site only |
| `astropy.visualization`/WCSAxes | WCS plotting | interop | torchsky.viz is optional |
| Aladin Lite / JS9 / skyview | browser FITS viewer | interop | torchsky.viz serves Aladin Lite |

## 9. Ownership note for FITS

Torchsky's compatibility policy forbids `astropy.io.fits`, `fitsio`, and
`astropy.Table.write(..., format="fits")` in production paths. The replacement
for those packages is **torchfits**, evaluated in the torchfits repository; it
is listed here only so the boundary is explicit.

## Current release evidence (2026-10-21)

Release-gated comparisons: astropy WCS/WCSLIB, healpy (core/advanced/spin/real
map), astropy SkyCoord catalog matching, reproject real cutout, real instrument
WCS headers. Contract/fuzz gates cover structural/domain behaviour.

Non-release coverage exists for hpgeom, astropy-healpix, healsparse, mocpy,
spherical_geometry, drizzle, dfreproject, SWarp, Montage, s2fft, cuhpx, PyAST,
Kapteyn, ERFA, and pandas/polars/Arrow. Everything marked **gap/unit-only** is
the next campaign list.
