# Release gate matrix — torchsky 0.4.0 / harness 0.1.0

Status: **active**. Thirteen declared gates pass with
`scientific_validity: established` in the release bundle (seven
synthetic/truth gates with `comparison_eligibility: baseline_compared`, six
reference-comparison gates with `suite_owned` thresholds). Rows below that are
still marked open are not claimed. Tolerances are frozen after the first
calibrated run and are not adjusted to make a failing gate pass. Release steps:
[release-procedure.md](release-procedure.md).

## Gate rules

1. A gate is a non-zero-exit command run through
   `torchsky-harness run ... --run-dir <unique>`.
2. Required rows and required comparators are declared. A missing row, a
   `SKIPPED` required comparator, or a missing required metric is a failure.
3. Correctness baselines are hardware-independent and committed. Performance
   baselines are hardware-fingerprinted and only gate on the pinned reference
   machine.
4. Every run records the installed subject identity, harness revision,
   comparator versions, device, dtype, seeds, and the run-directory hash.
   A declared gate also records `scientific_validity` and
   `comparison_eligibility`; see [evidence-verdicts.md](../../evidence-verdicts.md).
   A run may only claim `established` for a claim declared in
   `src/torchsky_harness/evidence.py` with all required rows present and passing.
5. A passing gate certifies only its declared cases and contract. It does not
   certify unlisted projections, devices, data regimes, or models.

## P0 gates

### WCS

| Item | Contract |
|---|---|
| Evaluations | `bench_wcs`, `bench_wcs_spectral_temporal`, `bench_wcs_tabular`, `bench_wcs_tpv_inverse`, `bench_observed_place` |
| Required cases | TAN, SIN, ARC, CEA, AIT, MOL, HPX, ZPN, TPV, TAN-SIP; forward and inverse; interior, boundary, outside-domain grids; 3-axis cube slices |
| Comparators | astropy (required); PyAST/Kapteyn optional legacy rows, never a pass substitute |
| Correctness | analytic projections ≤1e-9 deg; distorted ≤1e-5 deg; round-trip ≤1e-6 px; invalid domain raises or returns an explicit invalid mask, never silent NaN; spectral/time/tabular axes match their declared reference fixtures |
| Performance (pinned) | large-N common projections first or within 10% of astropy; small-N overhead recorded, not gated; CSC limitations documented |
| Delivered comparison gate | `bench-wcs-release-gate` (bench-all env): 30 projections vs astropy, 20k samples each. Angular parity max **4.13e-5 asec** (AIR formula floor; all other cases ≤4e-8 asec). Sky-domain round trip excluding poles/seams max **4.13e-5 asec**, pixel round trip max **1.27e-6 px**, p99 sky round trip max **6.3e-3 asec**. Throughput: forward 0.54–3.70× astropy (25/27 non-seam cases faster; AIR 0.54×, XPH 0.78× slower), inverse 0.86–6.90× (PAR 0.86×). Quad-cube seam conditioning is identical in astropy (CSC max 45.15 asec in both) and is exempt from round-trip thresholds, not from angular parity. Task thresholds are committed in `pixi.toml` |
| Open blockers | silent inverse non-convergence; clamp/path-dependent domain policy; TPV auto-detection; stale fused/compile caches; AIR/XPH forward throughput on macOS; no pinned-runner perf baseline |

### Sphere

| Item | Contract |
|---|---|
| Evaluations | `bench_sphere_matrix`, `bench_healpix`, `bench_healpix_advanced`, `bench_sphere_sparse`, `bench_sphere_spectral`, `bench_spin_matrix`, `bench_sphere_geom_exact_contains`, `bench_sphere_polygons` |
| Required cases | ring/nest `ang2pix`/`pix2ang`, `ring2nest`/`nest2ring`, neighbors/interp, nside 1..2**29, poles, high nside; MOC build/contains/union/degrade; sparse get/update/ud_grade; scalar and spin SHT to lmax 128; polygon containment incl. non-convex and >hemisphere cases |
| Comparators | healpy, hpgeom, astropy-healpix, healsparse, spherical-geometry, mocpy, s2fft (required where the row is claimed) |
| Correctness | index parity exact; angle parity ≤1e-10 deg; MOC set algebra matches mocpy on synthetic high-order cells; sparse values match healsparse; SHT rel-L2 ≤1e-6 scalar and ≤1e-5 spin against healpy; adjoint identity ≤1e-10 relative; device branches agree within declared tolerances |
| Performance (pinned) | core geometry first within declared ratios; SHT either ≥0.8× healpy or marked **beta**; MOC cone/union fixed or marked beta; `get_values`/`ud_grade` claimed only where measured |
| Delivered comparison gates | `bench-healpix-release-gate`, `bench-healpix-advanced-release-gate`, `bench-sphere-matrix-release-gate`, `bench-spin-matrix-release-gate` (sphere-bench env). HEALPix: 6 core ops and 6 advanced ops at nside 512 vs healpy with **zero index mismatches**, max angle 1.14e-13 deg, throughput 1.02–2.19× and 1.25–2.06×. Sphere geometry: 3 profiles at nside 512, index/angle parity gates pass; median healpy ratios 2.00/1.54/1.27/0.98/1.43/0.98 are recorded but not gated on macOS because run-to-run noise crosses the Mac-tuned throughput threshold; hardware-fingerprinted perf gating stays on the pinned runner. Spin-2 SHT: no failures, small-bucket median ratios 0.459/0.422 vs 0.15 thresholds |
| Open blockers | high-order MOC decode (fixed, pending gate); nside cap (fixed); import-time compile (fixed); SHT adjoint/device gates; alias surface consolidation; MOC cone/union throughput; no pinned-runner perf baseline |

### Map and multi-spectral

| Scope | Cases | Comparators / truth | Correctness gate | Performance gate |
|---|---|---|---|---|
| Multi-band imaging | 3–6 band TAN/SIP/TPV synthetic exposures with distinct bandpasses, PSFs, masks, rotations; real SIP MEF fixture | analytic truth; `reproject`/SWarp per band | per-band RMSE ≤1.10× reference coadd; band flux error ≤1%; colour-ratio error ≤0.02 mag; variance coverage 0.95±0.05 | `coadd_multiband` within 10% of per-band coadd |
| IFU / cubes | 3-axis WCS synthetic cube with lines, non-uniform/log spectral axis, masks; `plan_cube`/`coadd_cube`; `write_cube_fits` round-trip. Delivered: matched-sampling cube gate, and an IFU spectral-axis gate over a synthetic long-slit `IFUSliceWCS` | analytic truth; astropy WCS | matched sampling: exact valid-voxel indexing. Spectral axis: exact line localization and exact constant-spectrum normalization on an irregular axis (all kernels); regular-axis smooth-spectrum interior ≤2.5e-3 (observed 6e-4); irregular-axis smooth spectrum gated as a recorded ≤10% envelope (observed 3.2%) because the kernel interpolates in index space. Still open: physical wavelength-space interpolation, flux-conserving redistribution, line-flux/centroid metrics, `write_cube_fits` round-trip gate | cube coadd within 10% of slice-by-slice; peak RSS bounded by output, not exposure count |
| Joint fields | `bench_map_field_fit` with held-out observations and sky quadrants; models: `coadd_bilinear`, `sine_implicit`, one recommended model. Gate `bench-map-field-fit-gate` passes a re-baselined smoke contract (`sine_implicit` RMSE 0.0013, `coadd_bilinear` 0.0148) | held-out truth; coadd baseline | noninferiority to coadd on held-out support; seeded determinism; no NaN. Still open: held-out noninferiority baseline and preregistered budget | preregistered fit budget and peak memory |
| HEALPix multi-band | 3 synthetic bands (0.04/0.08/0.16 deg pixels) at nside 2048; `coadd_healpix` per band plus `resample_healpix` band stacking | analytic constants, Gaussian source, and linear sky slope | constant recovery 1.9e-7 rel; covered solid angle within 0.82% of the pixel-centre footprint; band ratio 7.6e-8; per-pixel Gaussian residual 1.9e-7; absolute source position 0.076", cross-band 0.106"; plane residual 1.2e-6; band-stacked round trip 1.9e-7. Gate: `bench-map-healpix-multiband-gate` | not gated: no healpy throughput row, no harmonic/SHT path, no variance product |

Shared map contract: `flux` vs `surface_brightness` area scaling, variance/weight
propagation, outlier rejection, spectral unit/type normalization, and CLI/Python
parity are all part of the gate.

**Current evidence is bounded and synthetic/fixture-based; no row is a
real-data claim.** WCS, sphere, and catalog comparator rows are delivered by the
reference-comparison gates above. Multi-band imaging exists only as smoke rows in
`bench_map_image_plane` and `bench_map_multifield_truth` (no per-band/colour
gate); a synthetic 2D-band cube assembly gate exists (`bench_map_cube`, voxel
RMSE and flux conservation); a matched-sampling 3D IFU gate checks exact
valid-voxel indexing and sparse coverage, and an IFU spectral-axis gate covers
irregular-axis localization, normalization, and a recorded interpolation
envelope, but physical wavelength-space interpolation and flux-conserving
redistribution are still missing; a dedicated
HEALPix multi-band gate now covers per-band `coadd_healpix` plus band stacking
(`bench_map_healpix_multiband`), replacing the opt-in
`--include-spherical-inverse` path as the HEALPix evidence; the joint field
held-out contract exists but has no committed noninferiority baseline.

Open blockers: spectral flux/unit semantics, IFU non-uniform axes, unbounded
cube/HEALPix memory, `coadd_healpix` is 2D-only with centre sampling (multi-band
means one coadd per band), no HEALPix variance/weight product, no healpy
throughput row, CLI/product path divergence, weight/IVAR semantics, no
colour/uncertainty gate, and missing IFU non-uniform evaluator.

## P1/P2 gates

| Domain | Release bar |
|---|---|
| `catalogs` | `bench-catalog-release-gate` delivers `bench_catalog_compat` vs astropy `SkyCoord`: 6 rows OK (synthetic clustered, astropy VOTable catalogs, VOTable→FITS round trip), max separation difference **1.13e-10 asec**. `bench_catalog_crossmatch` remains unclaimed; completeness/false-positive claims need real survey catalogs |
| `field` | experimental; included in the map field gate above; model zoo remains non-default |
| **native response/likelihood** | **P1 experimental gates, partially added:** `bench_native_response` gates recovered flux/background for a known pixel-integrated Gaussian star; `bench_native_likelihood` solves a known Gaussian likelihood and gates standardized error plus stale-evidence rejection; `bench_native_nuisance` checks profiled/calibrated nuisance precision against dense references. Still needed: joint/shared-nuisance and poorer-data cases. The real ACS campaign is supporting evidence, not a pass (`training_ready=false`). No SOTA or universal-device claim. |
| `scene` | experimental facade; contract/`__all__` tests only |
| `viz` | optional extra; import/server/static security smoke |
| `encoders` | experimental; bounded shape/device/dtype tests |
| `coords` | compatibility alias; identity test against `wcs` |

## Evidence bundle

`docs/releases/0.1.0/` must contain, per gate (record format:
[evidence-verdicts.md](../../evidence-verdicts.md)):

- exact command, run directory, `execution.json` hash (collect the whole set
  with `pixi run evidence-bundle`, which emits
  `benchmark_results/evidence_bundle.{json,md}`);
- subject version, import path, package-content SHA-256, dirty flag;
- harness revision;
- comparator versions;
- hardware and software fingerprint;
- correctness and performance tables;
- claim/non-claim verdict and known limitations.
