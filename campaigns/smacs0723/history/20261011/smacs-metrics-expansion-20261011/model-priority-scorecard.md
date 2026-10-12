# SMACS model-priority scorecard

This is a validation-only snapshot across seven model families. The blank detector metrics below are not injection results. Full matched injection scores are available for the linear-band and explicit-spectral grid fields; Foam has a signed-flux audit but no usable positive effective PSF; four families remain pending. No cross-family winner is assigned. Full-precision values and source hashes are in [model-priority-scorecard.json](model-priority-scorecard.json).

## Blank detector validation

Chi-squared per pixel is reported for the blank batch on the same two held-out exposures per band (127,972 F150W pixels; 31,368 F277W pixels). The source report hash for each family is recorded in the JSON.

| Model family | F150W chi²/pixel | F277W chi²/pixel | Injection evidence |
| --- | ---: | ---: | --- |
| `linear_band_grid_field` | 1.7175 | 2.4826 | Complete validation-only score |
| `sine_implicit` | 3.1429 | 14.2297 | Pending full injection metrics |
| `sasnet` | 3.4027 | 17.4560 | Pending full injection metrics |
| `sky_lambda_feature_field` | 4.3800 | 31.3759 | Pending full injection metrics |
| `gaussian_basis` | 14.3187 | 51.9974 | Pending full injection metrics |
| `explicit_spectral_grid_field` | 32.4054 | 26.1807 | Complete validation-only score |
| `spectral_power_foam` | 79.0977 | 227.2280 | Partial signed-flux audit; positive-PSF score unavailable |

Blank detector fit is one evidence channel only; it does not establish source completeness, purity, resolution, or recovered source flux.

## Matched injection scores

The verified r3 outputs use three validation batches of 28 sources each, common support of 12,095 pixels, target purity 0.95, and 10,000 finite bootstrap draws for the reported `m50` intervals. Both model-specific runs selected `fitted_gaussian__swarp_lanczos3` as their comparator under the stated gates; this is not an overall model-family selection.

| Model family | Completeness | Purity | `m50` AB mag (95% bootstrap CI) | Effective FWHM px (F150W / F277W) | Negative-lobe fraction (F150W / F277W) | Bright-source median fractional flux bias (F150W / F277W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_band_grid_field` | 0.6786 | 0.9500 | 29.714 (29.667–30.000) | 3.638 / 3.907 | 0.709 / 0.577 | −0.032 / −0.022 |
| `explicit_spectral_grid_field` | 0.2500 | 1.0000 | 27.583 (27.500–27.750) | 3.015 / 3.985 | 0.172 / 0.251 | −0.997 / −0.493 |

The effective-PSF resolution gate is false for both rows. Bright-source response statistics use the four brightest AB27 injected sources per band, separate from the frozen operator calibration. The full score JSON also records flux NMAD, maximum absolute flux error, per-batch completeness/purity, validation manifest hashes, effective-PSF hashes, and the selected comparator.

## Foam signed-flux audit

Foam's r3 signed-response audit preserves negative response and reports registered 19×19 stamp flux. It does not produce a positive finite total in every stamp, so completeness, purity, effective FWHM, and `m50` remain unavailable. The table shows the bright validation-source median fractional flux bias and the dropout-batch audit; zero-truth F150W dropout entries have no fractional bias.

| Band | Validation-000: sources; median flux bias | Validation stamps all positive finite? | Dropout-000: sources; median flux bias | Dropout stamps all positive finite? |
| --- | ---: | :---: | ---: | :---: |
| F150W | 4; −1.0106 | No | 28; n/a (zero truth) | No |
| F277W | 4; −0.9422 | Yes | 28; −0.9103 | No |

Four families still need full matched injection metrics: `sine_implicit`, `sasnet`, `gaussian_basis`, and `sky_lambda_feature_field`. Their blank-batch detector measurements are listed above but are not used to infer injection performance.

## Scope and limits

This is a fixed-budget validation comparison, not a convergence or depth claim. Models used 120 Adam steps at learning rate 0.003 with no PCG refinement; the linear-band grid used at most 64 CG iterations. Detector scoring uses diagonal Gaussian likelihood with supplied ERR and ignores pixel correlations. There is no independent test evaluation, and the joint all-family evaluation remains pending. Foam's signed 19×19 stamp flux, with border-median subtraction, is not a measure of diffuse-flux conservation.

The complete detector source is [detector-validation.json](detector-validation.json). The matched linear and explicit model-selection artifacts are in [r3 linear scoring](../smacs-inr-20261010/runs/smacs-matched-metrics-recovery-20261011-r3/model_scoring/linear_band_grid_field/model_selection.json) and [r3 explicit scoring](../smacs-inr-20261010/runs/smacs-matched-metrics-recovery-20261011-r3/model_scoring/explicit_spectral_grid_field/model_selection.json); Foam's [r3 signed-response audit](../smacs-inr-20261010/runs/smacs-matched-metrics-recovery-20261011-r3/signed-response/spectral_power_foam.json) explains the unavailable PSF score.
