# Published UPDC comparison inventory

## Existing comparison surface

- `benchmarks/bench_updec_webb_real.py` compares the public Module B UPDC-v2 iter0 Drizzle mosaic with iter10. It reads SCI through `torchfits`, aligns the central 768-pixel crop with an analytic linear-TAN transform and bilinear sampling, and reports image RMSE/MAE, summed-flux ratio, gradient and Laplacian ratios, WCS/header/HDU details, and read time. The full-grid WCS keys differ; this is a static image-pair diagnostic, not a detector-level or source-completeness comparison.
- The archived six-band JSON is `benchmarks_results/smacs-priority-evaluation-20261010/evidence/20260729-updec-iter0-vs-iter10-e87cb11/updec-pair.json`. It covers F090W–F444W. The review notes summarize iter10/iter0 summed-flux ratios of 1.0008–1.0243, gradient ratios of 1.48–2.11, and Laplacian ratios of 1.29–4.22. The durable CANFAR result directory is `/arc/projects/ots/torchsky-runs/20260729-updec-iter0-vs-iter10-e87cb11/` (the run log records JSON and Markdown there). See `docs/projects/adass_updec_depth_week.md` and `benchmarks_results/smacs-priority-evaluation-20261010/EVALUATION.md` for interpretation.
- `benchmarks/updc/compare_catalogs_vs_reference.py` compares a candidate FITS/ECSV catalog with a reference catalog using `torchsky.catalogs.crossmatch_sky`, then deterministic maximum-cardinality one-to-one matching. It reports matched/candidate-only/reference-only counts, separation median/p95, flux-ratio median/q16/q84, and optional reference-SNR bins. `--detect-images` can make a basic photutils catalog from SCI images. The repository does not identify a published iter10 source-catalog filename or its column schema; this CLI needs those inputs supplied.
- `benchmarks/updc/prepare_inputs.py` is the preprocessing path for corrected native CAL exposures. Its upstream survey-specific 1/f correction is an explicit prerequisite; this script handles outlier estimation/completion and writes a `UPDCREJ` extension. The complete chain and example commands are in `benchmarks/updc/README.md`.
- `scripts/run_canfar_adass_depth.sh mosaic` uses the published iter0 grids as target WCS for full-field Torchsky coadds. It is not an iter10 science comparison. The current policy treats published UPDC products as secondary appearance/photometry context, never injected-depth truth; the same injected detector exposures would have to be run through UPDC for a primary depth comparison.

## Exact Module B reference filenames and input locations

`scripts/download_updec_webb.py` pins the filenames and NADC download IDs used by the existing benchmark. Local default: `data/updec_webb/`. CANFAR mount used by the runner: `/arc/projects/ots/torchsky-data/updec_webb/` (overridable with `UPDEC_REFERENCE_ROOT`). These are the six iter0/iter10 pairs the benchmark selects:

| Band | Iter 0 filename (NADC ID) | Iter 10 filename (NADC ID) |
|---|---|---|
| F090W | `F090Wb_UPDC-v2_iter0_i2d.fits.fz` (47945) | `F090Wb_UPDC-v2_iter10_i2d.fits.fz` (47947) |
| F150W | `F150Wb_UPDC-v2_iter0_i2d.fits.fz` (47958) | `F150Wb_UPDC-v2_iter10_i2d.fits.fz` (47959) |
| F200W | `F200Wb_UPDC-v2_iter0_i2d.fits.fz` (47969) | `F200Wb_UPDC-v2_iter10_i2d.fits.fz` (47970) |
| F277W | `F277Wb_UPDC-v2_iter0_i2d.fits.fz` (47980) | `F277Wb_UPDC-v2_iter10_i2d.fits.fz` (47981) |
| F356W | `F356Wb_UPDC-v2_iter0_i2d.fits.fz` (47991) | `F356Wb_UPDC-v2_iter10_i2d.fits.fz` (47992) |
| F444W | `F444Wb_UPDC-v2_iter0_i2d.fits.fz` (48002) | `F444Wb_UPDC-v2_iter10_i2d.fits.fz` (48003) |

The runner constructs download URLs as `https://nadc.china-vo.org/res/file_upload/download?id=<NADC ID>`. No network access or download was used for this inventory. Raw SMACS detector inputs are staged under `/arc/projects/ots/torchsky-data/jwst_smacs0723/{cal,crf,aligned}`; the matched 135-exposure input pattern is `aligned/*_crf_rshift_tweakreg.fits`, with the source alignment directory documented as `results/wcs_aware_multiband_inr/official_no_psf_whitened_l2_batch100_400_tweakreg_rshift_8plus3_run2/alignment/tweakreg_rshift`.

## Reusable commands

Download the public image pairs and reproduce the existing static comparison:

```bash
pixi run -e test -- python scripts/download_updec_webb.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --iterations 0,10 --output-dir data/updec_webb
pixi run -e test -- python benchmarks/bench_updec_webb_real.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --input-dir data/updec_webb \
  --json-out data/benchmark_results/updec_webb_real_smoke.json \
  --markdown-out data/benchmark_results/updec_webb_real_smoke.md
```

Compare catalogs once an iter10-derived reference table and compatible candidate catalog are available:

```bash
pixi run -e test -- python benchmarks/updc/compare_catalogs_vs_reference.py \
  --candidate <candidate-catalog.fits> --reference <iter10-reference-catalog.fits> \
  --ra-col ra --dec-col dec --flux-col flux --snr-col snr \
  --radius-arcsec 0.25 --out benchmark_results/updc_catalog_qa.json
```

Prepare upstream-corrected native exposures before the field-fit / mosaic chain:

```bash
pixi run python benchmarks/updc/prepare_inputs.py \
  --input-dir <externally-1f-corrected-CAL-dir> \
  --output-dir <prepared-dir> --target-reference <UPDC-iter0-grid.fits> \
  --bands F277W,F356W,F444W
```

The `catalog-qa` path is gated on candidate/reference table schemas and filenames that are not specified in this repository. It should not be treated as runnable with the mosaic filenames above: it expects catalogs with RA, Dec, and flux columns.

## Missing scientific measurements

The archived image diagnostic establishes sharper pixel-scale structure and near-conserved central-crop summed flux, but it does not establish a calibrated science advantage. To support that claim, the comparison still needs:

- Effective PSFs or response transfer functions for both products, including encircled energy and resolution matching on the common sky grid. Gradient and Laplacian ratios are uncalibrated sharpness proxies.
- Completeness versus magnitude at fixed false-positive surface density, with paired uncertainties and source classes. This requires matched injections into the same native detector exposures and rerunning both stacks through detection; static published images cannot supply it.
- Per-source flux bias and scatter, color bias, astrometric residuals, and morphology/size recovery against injected truth or a separately specified source catalog. Aggregate crop-sum ratios are not per-source photometry.
- Noise and uncertainty calibration, including correlated noise/covariance after resampling/deconvolution, plus held-out detector-pixel predictive scores if the comparison is intended to rank reconstruction fidelity.
- A pinned reference-catalog source, explicit column/unit mapping, and matching/detection configuration before the catalog QA command can produce a reproducible report.

No jobs, downloads, or test suites were run for this inventory.
