# Completed SMACS metrics results

Values below come only from locally captured result JSONs and session exits. They are descriptive measurements from distinct cohorts; no depth or best-stacker claim is made. Frozen result JSONs remain unchanged.

## Published UPDC iteration 0 versus iteration 10

Both published static products were bilinearly resampled to the same 512 × 512 target grid. Iteration 0 is the comparison reference, not truth. The reported signal/residual metric is map agreement to iteration 0, not detection S/N. This profile cannot measure injection response, held-out detector likelihood, or effective PSF; `depth_claim` is false.

| Band | Common pixels | RMSE vs iter 0 | Signal/residual dB vs iter 0 | Iter10/iter0 common-support image-sum ratio |
| --- | ---: | ---: | ---: | ---: |
| F090W | 262,144 | 0.6111 | -1.8959 | 0.8641 |
| F150W | 262,045 | 1.8499 | -5.9298 | 0.9637 |
| F200W | 262,144 | 1.3841 | -3.8787 | 0.9709 |
| F277W | 262,144 | 0.5221 | -2.1051 | 1.0140 |
| F356W | 262,144 | 0.4466 | -3.0000 | 1.0132 |
| F444W | 262,144 | 0.3329 | -1.8006 | 0.9527 |

The earlier `total_flux_relative_error` column is omitted: its integrated-flux denominator is clamped at `1e-8`, so it is not a valid relative error for these low-flux totals. The image-sum ratio is a direct common-support map diagnostic, not a calibrated conservation or flux-loss measurement.

## Published UPDC versus archived SWarp comparison

The completed static-integrity run (`smacs-published-static-integrity-20261011-r1`, exit 0) compared both published iterations with the archived SWarp Lanczos3 validation leader on the shared 512 × 512 grid. The archived map is a reference, not truth. The residual mean bias below is in MJy/sr; the sum ratio is the signed common-support sum ratio for the iter-10 image relative to SWarp.

| Band | Iter-10 residual bias (MJy/sr) | RMSE vs SWarp (MJy/sr) | Iter-10 common-support image-sum ratio |
| --- | ---: | ---: | ---: |
| F090W | -0.1858 | 0.9088 | 0.0610 |
| F150W | -0.1914 | 2.0460 | 0.1209 |
| F200W | -0.1756 | 12.8592 | 0.1536 |
| F277W | -0.1088 | 3.4363 | 0.1489 |
| F356W | -0.1104 | 0.9759 | 0.1398 |
| F444W | -0.2402 | 0.7142 | 0.0623 |

The large offsets and unlike calibration/exposure histories prevent interpreting the small map-sum ratios as flux loss. Unit-aware Jy sums were computed, but this archival comparison does not establish flux conservation. Neither this analysis nor the UPDC iteration comparison measures detection depth or effective PSF.

## Synthetic known-truth CPU run

The CPU retry completed 18 records: 12 fitted model rows and two coadd-baseline rows, plus four non-measurement recommendation records excluded here. Both synthetic cases used 120 optimizer steps and held out one detector observation. However, `heldout_sky_pixel_count` is zero in the output rows, so these results do not establish spatially independent held-out-sky performance. RMSE and flux errors below describe the generated synthetic truth cases only.

| Synthetic case | Model | Reconstruction RMSE | Signal/residual dB | Total-flux relative error | Held-out observation RMSE | Fit seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tan_sip_field_fit_multicolor | `sine_implicit` | 0.0361 | 30.6351 | 0.0002 | 0.0342 | 4.14 |
| tan_sip_field_fit_multicolor | `sasnet` | 0.1123 | 20.7703 | 0.0439 | 0.2160 | 14.91 |
| tan_sip_field_fit_multicolor | `spectral_power_foam` | 1.0621 | 1.2568 | 0.2716 | 1.5643 | 5.90 |
| tan_sip_field_fit_multicolor | `gaussian_basis` | 0.1458 | 18.5028 | 0.0566 | 0.2044 | 2.89 |
| tan_sip_field_fit_multicolor | `explicit_spectral_grid_field` | 0.6387 | 5.6738 | 0.3303 | 1.0107 | 1.87 |
| tan_sip_field_fit_multicolor | `sky_lambda_feature_field` | 0.0631 | 25.7767 | 0.0256 | 0.1210 | 35.75 |
| tan_field_fit_psf_mismatch | `sine_implicit` | 0.0007 | 36.4472 | 0.0094 | 0.0183 | 1.73 |
| tan_field_fit_psf_mismatch | `sasnet` | 0.0348 | 2.6341 | 0.4912 | 0.0923 | 7.81 |
| tan_field_fit_psf_mismatch | `spectral_power_foam` | 0.1273 | -8.6393 | 0.0628 | 0.3135 | 4.43 |
| tan_field_fit_psf_mismatch | `gaussian_basis` | 0.3136 | -16.4712 | 5.4506 | 0.5218 | 2.02 |
| tan_field_fit_psf_mismatch | `explicit_spectral_grid_field` | 0.0340 | 2.8264 | 0.5805 | 0.0485 | 1.26 |
| tan_field_fit_psf_mismatch | `sky_lambda_feature_field` | 0.0011 | 32.6788 | 0.0173 | 0.0184 | 24.07 |

**Weighted cube coadd baseline values:**
- `tan_sip_field_fit_multicolor`: RMSE 0.0377; signal/residual 30.2570 dB; total-flux relative error 0.0060; coverage 0.6356; fit time 0.02 s.
- `tan_field_fit_psf_mismatch`: RMSE 0.0149; signal/residual 9.9662 dB; total-flux relative error 0.0058; coverage 0.4036; fit time 0.00 s.

## Historical raw-aligned classical blank batch

This is the six-band historical raw-aligned, five-pattern, 512-pixel crop cohort; it is not the calibrated 128-pixel response-model cohort. Each run wrote a blank-batch stackers JSON and then exited 1 because `--fail-on-method-error` was enabled. No injection batches or effective-PSF command ran.

| Band | Successful stackers | Recorded result |
| --- | ---: | --- |
| F090W | 0/7 | All 7 rows FAIL: six methods report no overlap with the reference metric grid; Montage fails (`mProject`). |
| F150W | 0/7 | All 7 rows FAIL: six methods report no overlap with the reference metric grid; Montage fails (`mProject`). |
| F200W | 0/7 | All 7 rows FAIL: six methods report no overlap with the reference metric grid; Montage fails (`mProject`). |
| F277W | 0/7 | All 7 rows FAIL: six methods report no overlap with the reference metric grid; Montage fails (`mProject`). |
| F356W | 6/7 | 6/7 OK, but reference support is only 3,465/262,144 pixels (1.32%); support RMSE 0.10819–0.10822, support flux ratio -138.31–-138.28, runtime 4.81–19.02 s. Montage fails (`mProject`). |
| F444W | 6/7 | 6/7 OK, but reference support is only 3,465/262,144 pixels (1.32%); support RMSE 0.23783–0.23786, support flux ratio -182.57–-182.54, runtime 6.04–17.53 s. Montage fails (`mProject`). |

For F356W/F444W, the tiny reference-support fraction and large negative support-flux ratios make the nominally successful rows unsuitable for a stacker ranking; they expose a comparison-grid/support problem. In all six bands, the real-SMACS source-built Montage path failed (`Montage command failed (mProject):`). A separate test of the same projection cases confirmed the CLI binaries crash with signal 11 and empty diagnostics, while the installed MontagePy 2.3.0 projection calls succeeded on those inputs. This verifies the Python binding on test inputs; it does not retroactively make the source-built CLI rows valid.

## Historical raw-aligned classical cohort, retry r3

All six bands now have complete r3 runs: F090W, F150W, F200W, F277W, F356W, and F444W. Each completed its blank plus three validation batches with six of six non-Montage methods successful, then measured the effective PSF and flux response for the brightest injected source. This remains the raw-aligned 512-pixel historical cohort, distinct from the calibrated 128-pixel response-model cohort. Across the six methods, the measured ranges were:

| Band | Gaussian-equivalent FWHM (pixels) | Median flux bias | Flux NMAD |
| --- | ---: | ---: | ---: |
| F090W | 0.936–1.453 | -45.8% to -1.7% | 4.92%–22.10% |
| F150W | 1.321–1.549 | -9.17% to +0.33% | 0.72%–13.25% |
| F200W | 1.490–1.680 | +0.04% to +3.12% | 0.62%–3.84% |
| F277W | 1.883–2.114 | -0.56% to +0.26% | 0.088%–2.12% |
| F356W | 2.189–2.411 | -0.072% to +0.786% | 0.023%–0.398% |
| F444W | 2.603–2.785 | -1.283% to -0.034% | 0.049%–0.739% |

F090W has a large method-dependent brightest-source flux bias and scatter; these response measurements should not be reduced to a stacker ranking. The blank-stack map metrics and injection responses do not establish a detection-depth ranking. Six separate raw-aligned per-band MontagePy scorer runs also completed; their common-support validation screens are reported below.

## Matched calibrated classical response128 retry r3

The runner wrote all 100 method/batch map-product records for F150W and F277W and a 20-row null-aperture table before exiting 1 in its later response-scoring command. The F150W dropout has zero injected truth flux; its rows correctly mark flux recovery and effective PSF as unevaluated. Across the ten methods its median absolute null aperture flux ranged from `4.37e-14` to `6.85e-14` Jy, and its maximum absolute null aperture flux ranged from `1.03e-13` to `2.41e-13` Jy. This is a null-control measurement, not a depth estimate. The scorer then stopped on a support-mask dtype mismatch (`uint8` support versus floating-point science image), so this run supplies no completed positive-source response or PSF measurement.

Result artifacts: `benchmarks_results/smacs-inr-20261010/runs/smacs-published-updc-map-metrics-20261011-r2/published-metrics.json`, `.../smacs-published-static-integrity-20261011-r1/static-integrity.json`, `.../smacs-model-known-truth-20261011-r2/truth.json`, `.../smacs-matched-classical-response128-20261011-r3/null-response.json`, and `.../smacs-classical-<band>-inj512-20261011-r3/effective_psfs.json` for F277W/F356W/F444W.

## Available matched-response128 validation scores

The completed MontagePy scoring run and the first completed model row from recovery r2 report validation-only `m50` summaries on F150W/F277W. `m50` here is an interpolated threshold-purity measurement over three 28-source injection batches; the threshold is selected using the same injected truth. It is descriptive and does not establish independent depth. The SWarp reference passes the PSF resolution gate; neither MontagePy nor the linear model does.

| Scoring output | Method | m50 (AB mag; 95% bootstrap interval) | Pooled completeness / purity | Effective FWHM F150W / F277W (px) | Resolution gate |
| --- | --- | ---: | ---: | ---: | --- |
| MontagePy response128 | MontagePy mProject | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.514 / 2.084 | fail |
| MontagePy response128 | SWarp Lanczos3 reference | 29.58 (29.25–29.75) | 0.679 / 1.000 | 1.430 / 1.880 | pass |
| Recovery r2, linear model only | coadd bilinear baseline | 29.50 (29.00–29.75) | 0.643 / 1.000 | 1.404 / 2.086 | fail |
| Recovery r2, linear model only | linear band-grid model | 29.71 (29.67–30.00) | 0.679 / 0.950 | 3.638 / 3.907 | fail |

The linear band-grid score is not an eligible resolution-preserving result: its measured effective FWHM fails the stated gate in both bands. The replacement recovery r3 coordinator (wkgscklq) was Running in the 22:17 UTC filtered process snapshot. It wrote model score JSONs for the linear and explicit spectral-grid models; their per-method values, common-support hashes, and effective-PSF numeric fields match the earlier r2 outputs. Foam has a separate signed-response record, but positive-PSF scoring is marked unavailable because an injection response has nonpositive total flux. Four model fits still await complete products and joint scoring remains pending. The prior coordinator r2 (wmclm1ba) was also Running in the 22:17 snapshot; it remains recorded separately. Provenance for r3 verifies 240 control and 30 model science products against matching source/staged hashes and the frozen manifests. These outputs remain validation-only, with `depth_claim: false`.

Scorer artifacts: `benchmarks_results/smacs-inr-20261010/runs/smacs-matched-montagepy-response128-20261011-r1/scoring/model_selection.json`, `benchmarks_results/smacs-inr-20261010/runs/smacs-matched-metrics-recovery-20261011-r2/model_scoring/linear_band_grid_field/model_selection.json`, and the corresponding r3 per-model JSONs under `benchmarks_results/smacs-inr-20261010/runs/smacs-matched-metrics-recovery-20261011-r3/model_scoring/`.

## Raw-aligned six-band common-support scorer

The combined scorer r4 used the staged seven-method products for the historical raw-aligned 512-pixel cohort and one common support of 234,535 pixels. The strict-JSON recovery writes unavailable values as null and records their original paths in an audit; numerical results and selection gates were unchanged. Each method reached the truth-tuned target-purity threshold, but the three-batch m50 bootstrap had zero finite draws for every method. No stacker was selected and no m50 depth comparison is available.

| Method | Pooled completeness | Pooled purity | m50 | FWHM range across six bands (px) | Resolution gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Torchsky bilinear | 0.917 | 1.000 | unavailable | 1.231–2.785 | fail |
| Torchsky Lanczos3 | 0.952 | 1.000 | unavailable | 0.936–2.606 | fail |
| Torchsky square drizzle | 0.917 | 1.000 | unavailable | 1.453–2.786 | fail |
| Torchsky conservative polygon | 0.917 | 1.000 | unavailable | 1.453–2.786 | fail |
| Drizzle square | 0.917 | 1.000 | unavailable | 1.453–2.785 | fail |
| SWarp Lanczos3 | 0.940 | 1.000 | unavailable | 1.350–2.603 | pass |
| MontagePy mProject | 0.917 | 1.000 | unavailable | 1.451–2.786 | fail |

Completeness uses 84 injected sources per method over three validation batches; purity thresholds use those same injected truths. The zero-finite-draw m50 result means this screen cannot rank detection depth even though the point completeness values are finite. depth_claim is false. The initial combined scorer attempts r1 and r2 reached scoring after staging all bands but failed strict JSON serialization on NaN; r3 stopped before scoring on a missing import. The corrected output is benchmarks_results/smacs-inr-20261010/runs/smacs-classical-sixband-metrics-20261011-r4/scoring/model_selection.json.

## Raw-aligned per-band MontagePy validation screens

Each row compares official MontagePy mProject with SWarp Lanczos3 in its band-specific run. These are three-batch validation summaries with thresholds tuned using the injected truth, not independent depth measurements. Pooled purity was 1.000 in every row; MontagePy failed the resolution gate in all bands and SWarp passed.

| Band | MontagePy m50 (95% interval) | MontagePy completeness | MontagePy FWHM (px) | SWarp FWHM (px) | SWarp m50 (95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| F090W | 29.375 (29.333–29.500) | 0.750 | 1.451 | 1.350 | 29.500 (29.333–29.667) |
| F150W | 29.375 (29.333–29.500) | 0.762 | 1.547 | 1.477 | 29.429 (29.333–29.500) |
| F200W | 29.333 (29.250–29.500) | 0.750 | 1.681 | 1.612 | 29.375 (29.333–29.500) |
| F277W | 29.278 (29.167–29.333) | 0.714 | 2.118 | 1.883 | 29.375 (29.333–29.500) |
| F356W | 29.214 (29.000–29.250) | 0.667 | 2.408 | 2.189 | 29.250 (29.250–29.250) |
| F444W | 28.750 (28.500–28.833) | 0.560 | 2.786 | 2.603 | 28.833 (28.750–29.000) |

The common-support completeness and m50 columns are descriptive validation screens. They do not establish a production depth ranking. Each scorer artifact is under benchmarks_results/smacs-inr-20261010/runs/smacs-montagepy-<band>-inj512-20261011-r1/scoring/model_selection.json.

## Completed matched response128 classical common-support score

The matched classical scorer r3 evaluated twelve methods on the calibrated 128-pixel F150W/F277W cohort, using 12,095 common-support pixels and three 28-source validation batches. All methods reached the truth-tuned purity threshold (pooled purity 1.000). SWarp Lanczos3 was selected: it has the largest reported m50 among the resolution-gate-passing methods. Torchsky Lanczos3 also passes the resolution gate, with lower m50. This is a validation-only comparison, not an independent depth claim.

| Method | m50 (AB mag; 95% interval) | Completeness / purity | FWHM F150W / F277W (px) | Resolution gate |
| --- | ---: | ---: | ---: | --- |
| Coadd bilinear | 29.50 (29.00–29.75) | 0.643 / 1.000 | 1.404 / 2.086 | fail |
| Torchsky bilinear | 29.33 (29.25–29.67) | 0.643 / 1.000 | 1.404 / 2.086 | fail |
| Torchsky Lanczos3 | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.319 / 1.879 | pass |
| Torchsky conservative polygon | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.507 / 2.081 | fail |
| SWarp Lanczos3 | 29.58 (29.25–29.75) | 0.679 / 1.000 | 1.430 / 1.880 | pass |
| Torchsky drizzle, pixfrac 0.600 | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.485 / 2.009 | fail |
| Torchsky drizzle, pixfrac 0.800 | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.495 / 2.040 | fail |
| Torchsky drizzle, pixfrac 1.000 | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.507 / 2.081 | fail |
| Official Drizzle, pixfrac 0.600 | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.485 / 2.009 | fail |
| Official Drizzle, pixfrac 0.800 | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.495 / 2.040 | fail |
| Official Drizzle, pixfrac 1.000 | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.507 / 2.081 | fail |
| MontagePy mProject | 29.25 (29.00–29.67) | 0.631 / 1.000 | 1.514 / 2.084 | fail |

The m50 threshold was selected using the same injected truth, so the estimate and bootstrap interval describe this validation set. The exact scorer output is benchmarks_results/smacs-inr-20261010/runs/smacs-matched-classical-metrics-20261011-r3/scoring/model_selection.json. The paired model detector-residual comparison is summarized separately in [detector-validation.md](detector-validation.md); it covers blank-batch likelihood only and contains no SWarp detector-forward score.

## Explicit spectral-grid fixed-budget configuration

A separate recovery score measured the specific response128 explicit-spectral-grid run configuration. It reached pooled completeness 0.25 at truth-tuned pooled purity 1.000, with m50 = 27.58 AB mag (95% bootstrap interval 27.50–27.75). Its effective FWHM was 3.015 / 3.985 pixels in F150W/F277W, and its brightest-source median flux biases were -99.67% / -49.33%. It failed the stated resolution gate. These results apply to this fixed-budget configuration only; they do not establish a limit for the model family.

The original response128 run exited 1 during null-response assembly because the expected blank coadd-bilinear F150W prediction was missing. The recovery score used the staged fit products and a shared SWarp comparison. Result artifacts: `benchmarks_results/smacs-inr-20261010/runs/smacs-matched-metrics-recovery-20261011-r2/model_scoring/explicit_spectral_grid_field/model_selection.json` and the matching r3 score JSON. The r3 provenance check found matching source/staged SHA-256 values for the model science products.

## Foam signed-response audit

The Foam audit exited 0 and retained the signed response without forcing positive normalization. The registered positive-response scoring status is unavailable because the F150W validation response also produced a nonpositive effective-PSF stamp; this was not limited to the zero-flux dropout. The audit reported brightest-source median flux biases of -101.064% in F150W and -94.216% in F277W. The F150W dropout absolute aperture response had median 1.109e-9 Jy and maximum 3.058e-8 Jy. These are border-median-subtracted 19 × 19 registered-source stamp measurements, not diffuse-flux conservation or an independent false-source-rate estimate. See [foam-response-audit.md](foam-response-audit.md) for the signed-flux details and limitations.

The matched recovery r3 Foam record independently classifies `spectral_power_foam` as `unavailable_nonpositive_injection_response`, with its signed response retained. Four other model families await fit products; the joint model score is pending. This is a configuration-specific scientific unavailability, not a completed family comparison. See `benchmarks_results/smacs-inr-20261010/runs/smacs-matched-metrics-recovery-20261011-r3/signed-response/spectral_power_foam.json` and `.../model_scoring/spectral_power_foam/scoring-status.json`.

The six-band combined scorecard across the two classical cohorts is summarized in [classical-priority-scorecard.md](classical-priority-scorecard.md). It preserves per-band/method support and score provenance; neither validation table above is a general production-depth claim.

## Provisional ready-model ordinary-response score

The run `smacs-ready-model-responses-20261011-r1` (`z5d4ls9p`) exited 0. It covered blank plus three validation batches for sine implicit and Gaussian basis, with the ordinary positive-source response measured on the four brightest AB27 sources in validation-000. Both model scores are scientifically unavailable: at least one registered effective-PSF stamp per model has nonpositive total flux, so the unchanged scorer rejects positive normalization. The signed flux responses were retained:

| Model | F150W median flux bias | F277W median flux bias | Ordinary effective-PSF score |
| --- | ---: | ---: | --- |
| Sine implicit | -107.1481% | +52.0246% | Unavailable: nonpositive stamp total |
| Gaussian basis | -100.0158% | -100.0045% | Unavailable: nonpositive stamp total |

Biases are the signed-response artifact's median fractional flux bias converted to percent. They use border-median-subtracted 19 × 19 registered stamps and do not measure diffuse-flux conservation. The joint selection's actual `selected` field is `fitted_gaussian__swarp_lanczos3`. It scored 12 methods—coadd plus eleven classical controls—on 12,095 common-support pixels; neither model was eligible for that joint score. SWarp Lanczos3 had pooled completeness 0.679, truth-tuned purity 1.000, m50 29.58 (95% bootstrap interval 29.25–29.75), and effective FWHM 1.430 / 1.880 pixels in F150W / F277W, passing the resolution gate. This is a validation-only, provisional control selection, not an independent depth claim.

SASNet, Sky Lambda, Foam, linear-band-grid, and explicit-spectral-grid were excluded from this run's subset. The all-seven coordinator remains separate and still awaits SASNet/Sky ordinary products plus Sine/Gaussian dropout products. Dropout response was not measured here; no null or dropout estimate is implied. `depth_claim` remains false.

