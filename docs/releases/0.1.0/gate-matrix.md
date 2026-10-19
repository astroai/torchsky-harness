# Release gate matrix — torchsky 0.4.0 / harness 0.1.0

Status: **draft**. No gate in this matrix has passed yet. The matrix is the
contract the release must satisfy; tolerances are frozen after the first
calibrated run and are not adjusted to make a failing gate pass.

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
| Open blockers | silent inverse non-convergence; clamp/path-dependent domain policy; TPV auto-detection; stale fused/compile caches; no WCS perf baseline |

### Sphere

| Item | Contract |
|---|---|
| Evaluations | `bench_sphere_matrix`, `bench_healpix`, `bench_healpix_advanced`, `bench_sphere_sparse`, `bench_sphere_spectral`, `bench_spin_matrix`, `bench_sphere_geom_exact_contains`, `bench_sphere_polygons` |
| Required cases | ring/nest `ang2pix`/`pix2ang`, `ring2nest`/`nest2ring`, neighbors/interp, nside 1..2**29, poles, high nside; MOC build/contains/union/degrade; sparse get/update/ud_grade; scalar and spin SHT to lmax 128; polygon containment incl. non-convex and >hemisphere cases |
| Comparators | healpy, hpgeom, astropy-healpix, healsparse, spherical-geometry, mocpy, s2fft (required where the row is claimed) |
| Correctness | index parity exact; angle parity ≤1e-10 deg; MOC set algebra matches mocpy on synthetic high-order cells; sparse values match healsparse; SHT rel-L2 ≤1e-6 scalar and ≤1e-5 spin against healpy; adjoint identity ≤1e-10 relative; device branches agree within declared tolerances |
| Performance (pinned) | core geometry first within declared ratios; SHT either ≥0.8× healpy or marked **beta**; MOC cone/union fixed or marked beta; `get_values`/`ud_grade` claimed only where measured |
| Open blockers | high-order MOC decode (fixed, pending gate); nside cap (fixed); import-time compile (fixed); SHT adjoint/device gates; alias surface consolidation; MOC cone/union throughput |

### Map and multi-spectral

| Scope | Cases | Comparators / truth | Correctness gate | Performance gate |
|---|---|---|---|---|
| Multi-band imaging | 3–6 band TAN/SIP/TPV synthetic exposures with distinct bandpasses, PSFs, masks, rotations; real SIP MEF fixture | analytic truth; `reproject`/SWarp per band | per-band RMSE ≤1.10× reference coadd; band flux error ≤1%; colour-ratio error ≤0.02 mag; variance coverage 0.95±0.05 | `coadd_multiband` within 10% of per-band coadd |
| IFU / cubes | 3-axis WCS synthetic cube with lines, non-uniform/log spectral axis, masks; `plan_cube`/`coadd_cube`; `write_cube_fits` round-trip | analytic truth; astropy WCS | voxel RMSE ≤1.10× reference; line flux ≤1%; centroid ≤0.1 spectral pixel; wavelength calibration ≤0.1 pixel; flux conservation | cube coadd within 10% of slice-by-slice; peak RSS bounded by output, not exposure count |
| Joint fields | `bench_map_field_fit` with held-out observations and sky quadrants; models: `coadd_bilinear`, `sine_implicit`, one recommended model | held-out truth; coadd baseline | noninferiority to coadd on held-out support; seeded determinism; no NaN | preregistered fit budget and peak memory |
| HEALPix multi-band | nside 64/256 synthetic multi-band maps; `coadd_healpix`, `sample_multiband_healpix`, SHT round-trip | analytic truth; healpy | per-band RMSE and colour ratio; partial-pixel area handling; SHT rel-L2 | sampling/coadd throughput vs healpy |

Shared map contract: `flux` vs `surface_brightness` area scaling, variance/weight
propagation, outlier rejection, spectral unit/type normalization, and CLI/Python
parity are all part of the gate.

**Current evidence is not sufficient for any row in the table above.** Multi-band
imaging exists only as smoke rows in `bench_map_image_plane` and
`bench_map_multifield_truth` (no per-band/colour gate); a synthetic 2D-band cube
assembly gate now exists (`bench_map_cube`, voxel RMSE and flux conservation),
a matched-sampling 3D IFU gate now checks exact valid-voxel indexing and sparse
coverage, but a dense spectral-interpolation/non-uniform-axis evaluator is still
missing; HEALPix multi-band is only the opt-in
`--include-spherical-inverse` path in `bench_map_multifield_truth`; the joint
field held-out contract exists but has no committed noninferiority baseline.

Open blockers: spectral flux/unit semantics, IFU non-uniform axes, unbounded
cube/HEALPix memory, 2D-only HEALPix coadd, CLI/product path divergence,
weight/IVAR semantics, no colour/uncertainty gate, and missing IFU/multiband
evaluators.

## P1/P2 gates

| Domain | Release bar |
|---|---|
| `catalogs` | `bench_catalog_compat` and `bench_catalog_crossmatch` with astropy `SkyCoord` reference; crossmatch separations ≤1e-9 deg and completeness 1.0; performance claim only if measured |
| `field` | experimental; included in the map field gate above; model zoo remains non-default |
| **native response/likelihood** | **P1 experimental gate, partially added:** `bench_native_response` injects a known pixel-integrated Gaussian star and gates recovered flux/background at a fixed centroid. Still needed: full likelihood/joint/evidence-staleness case and a poorer-data/shared-nuisance case. The real ACS campaign is supporting evidence, not a pass (`training_ready=false`). No SOTA or universal-device claim. |
| `scene` | experimental facade; contract/`__all__` tests only |
| `viz` | optional extra; import/server/static security smoke |
| `encoders` | experimental; bounded shape/device/dtype tests |
| `coords` | compatibility alias; identity test against `wcs` |

## Evidence bundle

`docs/releases/0.1.0/` must contain, per gate:

- exact command, run directory, `execution.json` hash;
- subject version, import path, package-content SHA-256, dirty flag;
- harness revision;
- comparator versions;
- hardware and software fingerprint;
- correctness and performance tables;
- claim/non-claim verdict and known limitations.
