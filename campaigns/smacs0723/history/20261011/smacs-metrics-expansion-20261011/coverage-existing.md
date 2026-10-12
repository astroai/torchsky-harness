# Existing SMACS / UPdec metrics coverage

Generated 2026-10-11 from local JSON artifacts. This is a read-only inventory: no remote jobs were queried and no metric was inferred from an image.

Found **40 shared-foam paths** grouped into **38 byte-distinct reports**, **8 model kinds** including the separate early smoke model, and three distinct common-world injection batches.

## Model kinds and distinct saved configurations

| Model | Role | Distinct config signatures | Recorded settings |
| --- | --- | ---: | --- |
| `coadd_bilinear` | Non-fitted bilinear coadd comparator. | 8 | F150W,F277W; crop=128@(112,0); steps=[0]; lr=unrecorded; `{"foam_basis_count": 4, "foam_extent": 0.55, "foam_grid_side": 32, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F150W,F277W; crop=128@(None,None); steps=[0]; lr=unrecorded; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F150W,F277W; crop=128@(None,None); steps=[0]; lr=unrecorded; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F150W,F277W; crop=512@(112,0); steps=[0]; lr=unrecorded; `{"foam_basis_count": 4, "foam_extent": 0.55, "foam_grid_side": 32, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>… 4 more signatures in JSON |
| `explicit_spectral_grid_field` | Explicit spatial/spectral grid field. | 20 | F090W,F150W,F200W,F277W,F356W,F444W; crop=32@(None,None); steps=[20]; lr=[0.001]; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F150W,F277W; crop=128@(None,None); steps=[150]; lr=[0.001]; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F150W,F277W; crop=128@(None,None); steps=[150]; lr=[0.001]; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F150W,F277W; crop=128@(None,None); steps=[150]; lr=[0.001]; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>… 16 more signatures in JSON |
| `gaussian_basis` | Gaussian basis / splat field. | 1 | F150W,F277W; crop=128@(None,None); steps=[20]; lr=[0.003]; `{"num_basis": 1024, "query_chunk_size": 1024, "spectral_resolution": 4, "splat": true}` |
| `linear_band_grid_field` | Direct linear band-grid comparator; saved effective kwargs identify CG where available. | 12 | F150W,F277W; crop=128@(112,0); steps=[34, 43]; lr=[0.003]; `{"spatial_resolution": 256}`<br>F150W,F277W; crop=128@(None,None); steps=[34, 48]; lr=[0.003]; `{"spatial_resolution": 256}`<br>F150W,F277W; crop=128@(None,None); steps=[26, 36]; lr=[0.003]; `{"spatial_resolution": 256}`<br>F150W,F277W; crop=128@(None,None); steps=[35, 45]; lr=[0.003]; `{"spatial_resolution": 256}`<br>… 8 more signatures in JSON |
| `sasnet` | SASNet implicit field model. | 1 | F090W,F150W,F200W,F277W,F356W,F444W; crop=32@(None,None); steps=[20]; lr=[0.001]; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}` |
| `sine_implicit` | SIREN-style implicit field model. | 9 | F090W,F150W,F200W,F277W,F356W,F444W; crop=32@(None,None); steps=[20]; lr=[0.001]; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F150W,F277W; crop=128@(112,0); steps=[120]; lr=[0.003]; `{"hidden_dim": 64, "hidden_layers": 3, "omega0": 30.0}`<br>F150W,F277W; crop=128@(112,0); steps=[30]; lr=[0.003]; `{"hidden_dim": 64, "hidden_layers": 3, "omega0": 30.0}`<br>F150W,F277W; crop=128@(112,0); steps=[30]; lr=[0.003]; `{"hidden_dim": 64, "hidden_layers": 3, "omega0": 30.0}`<br>… 5 more signatures in JSON |
| `sky_lambda_feature_field` | Sky/lambda feature field from the early two-band multi-model smoke. | 1 | F090W,F150W; crop=32@(None,None); steps=[4]; lr=[0.001]; `{"foam_basis_count": 4, "foam_extent": 0.55, "foam_grid_side": 32, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}` |
| `spectral_power_foam` | Spectral-power Foam field. | 3 | F090W,F150W,F200W,F277W,F356W,F444W; crop=128@(None,None); steps=[200]; lr=[0.02]; `{"foam_basis_count": 6, "foam_extent": 0.55, "foam_grid_side": 64, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F090W,F150W; crop=32@(None,None); steps=[2]; lr=[0.02]; `{"foam_basis_count": 3, "foam_extent": 0.55, "foam_grid_side": 4, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}`<br>F090W,F150W; crop=32@(None,None); steps=[4]; lr=[0.02]; `{"foam_basis_count": 4, "foam_extent": 0.55, "foam_grid_side": 32, "foam_lr": 0.02, "inr_hidden_dim": 64, "inr_hidden_layers": 3, "inr_lr": 0.001}` |

Model-specific kwargs are included when saved. Older reports often preserve only experiment-level Foam/INR knobs, steps, and learning rates; missing optimizer algorithm names are marked as unrecorded. `linear_band_grid_field` records a CG solver in effective model kwargs.

## Matched two-band comparison

The recovered October 10 field-fit report compares `coadd_bilinear`, `linear_band_grid_field`, and `sine_implicit` on F150W/F277W, crop 128×128 at `(x0=112, y0=0)`. INR job specs distinguish Adam initialization from optional neural-PCG refinement. The recovered 120-step Adam report exists; a corresponding recovered 120-step PCG shared-foam report is absent from the local inventory. See the JSON for exact specs and artifact paths.

## Published-stack and map-kernel measurements

| Evidence | Metric families | Scope |
| --- | --- | --- |
| `benchmark_results/map_swarp_required.json` | timing/throughput, coverage/weight and image/weight deltas; swarp:bilinear, swarp:lanczos3, torchsky:bilinear, torchsky:lanczos3 | synthetic map compatibility evidence; not a real-SMACS product comparison |
| `benchmark_results/map_swarp_clipped_required.json` | timing/throughput, coverage/weight and image/weight deltas; swarp_clipped:bilinear, torchsky:bilinear | synthetic map compatibility evidence; not a real-SMACS product comparison |
| `benchmark_results/map_drizzle_required.json` | timing/throughput, coverage/weight and image/weight deltas; drizzle:point, drizzle:square, torchsky:drizzle_point, torchsky:drizzle_square | synthetic map compatibility evidence; not a real-SMACS product comparison |
| `benchmark_results/map_montage_required.json` | timing/throughput, coverage/weight and image/weight deltas; montage:mProjectPP, torchsky:conservative_polygon | synthetic map compatibility evidence; not a real-SMACS product comparison |
| `benchmark_results/map_compat_gate.json` | timing/throughput, coverage/weight and image/weight deltas; casutools:None, dfreproject:bilinear, drizzle:point, drizzlepac:None, montage:mProject, reproject_exact:exact, reproject_interp:bilinear (+2) | synthetic map compatibility evidence; not a real-SMACS product comparison |

Real-SMACS evidence also includes archived stack validation and detection-score reports, effective-PSF and aperture-flux reports, and an SWarp SIP validation. Those reports cover selected SWarp, Drizzle, and Torchsky variants; no local real-SMACS Montage response or detection report was found.

## Published UPdec products

Published UPDC iter0/iter10 products are now an accepted **secondary static-map comparison** across all five priorities, with the limits below. `updec-pair.json` compares the six bands on 768×768 crops after `analytic_linear_tan_bilinear` alignment. It contains common/crop RMSE, MAE and flux ratio, gradient-mean and Laplacian-standard-deviation ratios, product shape/HDU/BUNIT/WCS metadata, and Torchfits read times. The report records differing WCS keys.

The newer `smacs-published-updc-map-metrics-20261011-r2` CANFAR run completed all six bands (exit code 0; `session.log` records each band). It reads iter0 as the comparison reference and iter10 as the candidate, resamples both through `torchsky.map.coadd_image_plane` with bilinear surface-brightness semantics onto the frozen 512×512 `common-support-r9-f0p99.fits` grid, and evaluates their common support. Its report calls `field_image_quality_metrics` and `evaluate_map_product`; these provide weighted image residuals (RMSE/MAE, bias/std and multiscale RMSE), target/prediction RMS, support/coverage/weight, absolute-error summaries, per-band and total flux errors/ratios, and finite-difference sharpness energy. It also records the common-support pixel-sum ratio against iter0 and input provenance. The output report is at the run's artifact root `/arc/projects/ots/torchsky-runs/smacs-inr-autonomous-20261010/smacs-published-updc-map-metrics-20261011-r2/published-metrics.json`; this local inventory has the synced exit record/log, not the six-row JSON.

The fields named `residual_snr`, `signal_to_residual_ratio`, and their dB variants are **image-agreement ratios**: target-image RMS divided by residual RMSE. They are not noise-calibrated source-detection SNR. `prediction_sharpness` / `sharpness_ratio`, and the earlier gradient/Laplacian ratios, are static finite-difference proxies, not injected effective PSFs or calibrated resolution. These comparisons provide useful common-grid image, support, and aggregate-flux context, but they are not truth, detector-pixel likelihood, injection response, completeness, false-alarm, per-source flux-recovery, or ePSF measurements. `r1` failed before input reads with `ModuleNotFoundError: No module named 'torchsky.map'`; it produced no metrics.

## Injection manifests and crops

Three independent fixed source manifests are:

- `benchmarks_results/smacs-grid-validation-20260924-r3/manifests/validation-000/manifest.json` — validation-000, 28 sources, seed 501, 512×512 at (112,0), `smooth_continuum_point`, six NIRCam bands.
- `benchmarks_results/smacs-grid-validation-20260924-r3/manifests/validation-001/manifest.json` — validation-001, 28 sources, seed 502, 512×512 at (112,0), `smooth_continuum_point`, six NIRCam bands.
- `benchmarks_results/smacs-grid-validation-20260924-r3/manifests/validation-002/manifest.json` — validation-002, 28 sources, seed 503, 512×512 at (112,0), `smooth_continuum_point`, six NIRCam bands.

The `scoring-v2` to `scoring-v5` brightest-injection manifests are derived subsets of validation-000. The separate local UPdec smoke uses F150W/F277W 32×32 cutouts, one exposure per band, with exact/perturbed PSF flux, centroid, valid-fraction, and background-plane metrics; it has no depth claim.

## Metric families

- Field fit and residual: losses, fit time, parameter/memory counts, source/support RMSE, contrast slope and flux ratios; later reports add background and high-frequency diagnostics.
- Held-out detector: per-band chi-square and Gaussian NLL per pixel with exposure/pixel counts in selected reports.
- Injection validation: pooled/per-batch completeness, purity, matches, selected thresholds, and completeness by AB magnitude for selected stack configurations.
- Effective PSF: Gaussian-equivalent width, negative-lobe fraction, centroids, peak/sum, aperture flux bias, NMAD, and maximum absolute flux error.
- Map compatibility: runtime, MPix/s, coverage, weights, image/weight MAE/RMSE/max-difference, edge/interior differences, support overlap, and speed ratios in synthetic reports.
- Published UPDC pair/common-grid map comparison: common/crop or weighted common-support image deltas, image-agreement RMS ratios, aggregate flux ratios, finite-difference sharpness proxies, and provenance/read-time fields. None is an injection-derived SNR or effective PSF.

## Coverage limits

The root `benchmark_results/map_*_required.json` files and the 2026-10-11 map-ground-truth run are synthetic compatibility cases, not real-SMACS detections. Existing real-SMACS validation has injection-response/effective-PSF metrics for selected SWarp, Drizzle, and Torchsky variants; no completed matched neural response report is recorded in this inventory. The published iter0/10 map comparison is useful secondary static context on a common grid, but cannot fill missing injection-derived SNR/completeness, detector likelihood, per-source flux recovery, artifact rate, or effective-resolution metrics. No metric is assigned to a model/method from quicklook appearance or apparent smoothness.

The deduplicated source paths, per-file metric fields, configuration signatures, and exact injection metadata are in [`coverage-existing.json`](coverage-existing.json).
