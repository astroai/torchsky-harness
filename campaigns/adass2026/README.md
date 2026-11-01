# ADASS XXXVI (2026) campaign: forward-model sky fits, UNIONS coadd, coadd speed

Evidence behind the talk "From FITS archives to differentiable multispectral sky models"
(S. Fabbro, ADASS XXXVI, Perth, 4 November 2026). Job sources here are the versions
used for the reported numbers; each run also froze its own script, launcher, `launch.env`,
sha256 sums, `pip freeze` and versions in `<run>/provenance/`.

Storage (not in Git): `arc:projects/ots/adass2026/` and the mirror
`vos:sfabbro/adass2026/` (runs under `smacs_four/<RUN>`, `unions_coadd_v5/`,
`unions_gaia_astrom_v5*/`, `bench_coadd*/`, `unions_diag*/`, `unions_coadd_v8*/`).
Subject: torchsky `main`, installed from source tarballs `scripts/torchsky-src-<commit>.tar.gz`
(science runs 9749c80, 14c9a6d and 3bd7b79; speed runs 82522bf and df37916), torch 2.14,
torchfits 1.3.1, weightmask 0.3.0, on CANFAR `astroai/base:latest` (SWarp benchmark:
`astroai/improc:latest`).

## 1. SMACS J0723, NIRCam F090W–F444W (`jobs/smacs_four_v14.py`, `jobs/smacs_run.sh`)

Forward model `d_i(p) = g_i ∫dλ R_i(λ) ∫d²x K_i(T_i(p) − x; p, λ) s(x, λ) + B_i(p) + n_i(p)`,
fitted to the detector pixels: chromatic PSF field (diffraction-unit MLP, smooth in λ and
detector position), per-detector cubic distortion + per-exposure affine, explicit background
and 1/f stripes, per-exposure gain, Huber loss (δ = 3); sky as a pixel grid, an implicit
neural field, SIREN, or a continuous `s(α, δ, λ) = Σ_k a_k φ_k(log λ)` integrated over SVO
transmission curves (photon-weighted nodes).

Protocol: `jobs/holdout_main.json` names one test exposure per band, never used for fitting
or stopping; one more exposure per band is validation (early stopping, best checkpoint).
Every method, coadds included, gets the same per-exposure nuisance refit on the test
exposures (affine registration, gain + quadratic background by weighted least squares, 1/f
stripes). Metrics:

- `⟨r²/σ²⟩` over all pixels and over pixels with S/N > 5 in the SWarp-like coadd;
  robust variant: pseudo-Huber at 3σ.
- flux error: NMAD of (data − prediction)/σ in 0.2″ apertures on isolated detections
  (scored when S/N > 20), plus median Δm and centroid NMAD.

Main results (`results/smacs_metrics.json`, 0.04″ grid, test exposures):

| method | ⟨r²/σ²⟩ all | S/N>5 | flux error |
|---|---|---|---|
| SWarp-like (torchsky Lanczos-3) | 6.11 | 3.65 | 2.8σ |
| Drizzle-like (torchsky square, pixfrac 0.8) | 5.88 | 2.77 | 2.7σ |
| explicit grid, one image per band | 5.60 | 1.84 | 2.1σ |
| implicit neural field | 5.66 | 2.03 | 2.2σ |
| explicit s(α, δ, λ), K = 8, relative spectral prior | 5.60 | 2.18 | 2.6σ |

Key runs: `v7_e04_huber`, `v7_i04_huber`, `v7_e04_gauss`, `v7_s04_siren_huber`,
`v11_e3d_04_relsmooth`, `v11_n3d_04_fixedbasis_relsmooth`, `v12_e3d_04_k12_rel`,
re-scored with the photometry metric as `rescore14(b)_<run>` (`--load-dir`, same CLI rebuilt
by `jobs/rescore_cli.py`). UPdec-Webb comparison (in-sample, all 123 exposures):
`refit_e04_huber_all_v12`, `rescore14_refit_e04_huber_all_v12`, `results/updec_metrics.json`.
Super-resolution trials at a 0.04″ target resolution (`v13b_sr_*`, `v13d_sr_convdecoder_cont`):
not better than Drizzle-like on S/N > 5 pixels; kept out of the main results.

## 2. UNIONS tile CFIS.000.229 (`jobs/unions_coadd_v5.*`, `jobs/unions_gaia*`)

Individual MegaCam exposures → weightmask (master flats from `vos:sgwyn/flats`, no streak
detection) → torchsky Lanczos-3 weighted coadd on the official TAN grid. Against the official
tile: 2″ aperture photometry within 2.1 mmag (r) in every magnitude bin; centroids within
1 mas. Against Gaia DR3 propagated to the exposure epoch (`results/unions_gaia_astrom_r.json`):
torchsky (−0.2, +2.9) mas, official (+0.2, +1.6) mas (east, north).

The MegaCam CCD headers keep `RADECSYS = 'FK5'` although the astrometry is Gaia-based; read
literally, torchsky applied the FK5 → ICRS frame bias (−29, −9 mas). v5 declares the frame
ICRS when `ASTCAT` is Gaia; torchsky now emits `CelestialFrameBiasWarning`.

## 3. Coadd speed against the legacy codes (`jobs/bench_coadd.*`, `results/bench_speed.json`)

135 NIRCam exposures (566 Mpix), six bands, 0.04″ TAN grids, wall clock including reads,
8 CPU threads / H100 MIG 1g.12gb. SWarp gets the SIP distortion converted to TPV (sip_tpv).
Subject torchsky main 82522bf (run `bench_coadd_82522bf`); in brackets torchsky 14c9a6d
(run before the large-field work).

| output grid | SWarp 2.41 | torchsky Lanczos CPU / GPU | STScI drizzle 3.0 | torchsky drizzle CPU / GPU |
|---|---|---|---|---|
| 512² | 30.1 s | 14.3 s / 5.2 s (24.2 / 5.7) | 171 s | 71 s / 27 s (124 / 42) |
| 5025² | 144 s | 221 s / 65 s (784 / OOM) | 245 s | 149 s / 148 s (1849 / 584) |

Outputs: torchsky drizzle = STScI drizzle to < 2e-6σ; torchsky Lanczos-3 vs SWarp median
< 0.1σ, scatter 0.1–0.9σ; torchsky CPU vs GPU: drizzle bitwise identical, Lanczos-3 1e-7σ.
Changes behind the speed-up (torchsky main): footprint cropping per observation, verified
coarse-lattice target→source mapping for SIP/TPV, fused chunked separable sampler with
shared-tap variance, memory-budget automatic tiling with host accumulators, exact
axis-aligned drizzle overlap with a native CPU kernel, deterministic CUDA accumulation.
torchsky before the SIP fix (9749c80): 213 / 217 s and 250 / 48 s on the small grid.

UNIONS tile, CPU (`jobs/unions_diag.*`: the same 80 prepared MegaCam CCDs onto the 10k² tile,
Lanczos-3 flux, 16 threads, cProfile):

| torchsky | 2500² tiles | automatic tiles |
|---|---|---|
| 9749c80 (v5 baseline) | 886 s | — |
| 82522bf | 816 s | 1878 s |
| df37916 (source area on the tap window) | 431 s | 483 s |
| 3a32386 (+ windowed quadrature variance) | 263 s | 294 s |

The 82522bf profile showed the exact per-pixel source solid angle (flux semantics,
spherical pixel polygons on TPV) recomputed over the whole CCD for every (tile, CCD) pair,
1432 s of 1878 s, then the four-point quadrature variance (Python scan over sorted taps).
Whole tile (280 CCDs, `jobs/unions_coadd_v5.*` with `--auto-tile`): 2292 s (v5, 2500² tiles)
→ 6102 s (`unions_coadd_v6_cpu`, 364269d) → 1084 s (`v8`, df37916) → 711 s (`v9`, 3a32386);
on an H100 MIG 1g.12gb slice 660 s (`unions_coadd_v10_gpu`, 95b5f2b, auto-tiled, no OOM).
Pixel, photometry and astrometry statistics against the official tile are identical in all
runs (`results/unions_compare_v9_cpu.json`, `results/unions_compare_v10_gpu.json`).

## Known deviations from harness policy

These are the job sources as executed. Some use `astropy.io.fits` for reading JWST products,
writing cubes and preparing the legacy-code inputs, and they install the subject from a source
tarball in a venv rather than through Pixi. Migrate to torchfits and a Pixi environment before
turning any of this into a release gate.
