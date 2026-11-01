# ADASS XXXVI (2026) campaign: forward-model sky fits, UNIONS coadd, coadd speed

Evidence behind the talk "From FITS archives to differentiable multispectral sky models"
(S. Fabbro, ADASS XXXVI, Perth, 4 November 2026). Job sources here are the versions
used for the reported numbers; each run also froze its own script, launcher, `launch.env`,
sha256 sums, `pip freeze` and versions in `<run>/provenance/`.

Storage (not in Git): `arc:projects/ots/adass2026/` and the mirror
`vos:sfabbro/adass2026/` (runs under `smacs_four/<RUN>`, `unions_coadd_v5/`,
`unions_gaia_astrom_v5*/`, `bench_coadd*/`). Subject: torchsky `main` 466ef1c
(source tarball `scripts/torchsky-src-14c9a6d.tar.gz` and earlier `9749c80`, `3bd7b79`),
torch 2.14, torchfits 1.3.1, weightmask 0.3.0, on CANFAR `astroai/base:latest`
(SWarp benchmark: `astroai/improc:latest`).

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

| output grid | SWarp 2.41 | torchsky Lanczos CPU / GPU | STScI drizzle 3.0 | torchsky drizzle CPU / GPU |
|---|---|---|---|---|
| 512² | 26.8 s | 24.2 s / 5.7 s | 171 s | 124 s / 42 s |
| 5025² | 141 s | 784 s / OOM | 242 s | 1849 s / 584 s |

torchsky before the SIP fix: 213 / 217 s and 250 / 48 s on the small grid. Outputs:
torchsky drizzle = STScI drizzle to < 0.001σ; torchsky Lanczos-3 vs SWarp median 0, scatter
0.1–0.6σ. Large output grids remain slower than SWarp (per-pixel variance propagation, SIP
inverse per target pixel); the full-grid Lanczos run exceeds a 12 GB GPU slice.

## Known deviations from harness policy

These are the job sources as executed. Some use `astropy.io.fits` for reading JWST products,
writing cubes and preparing the legacy-code inputs, and they install the subject from a source
tarball in a venv rather than through Pixi. Migrate to torchfits and a Pixi environment before
turning any of this into a release gate.
