# Paired detector reconstruction validation

All seven fitted families have a completed blank-batch report. The same frozen calibration, validation exposure IDs, and pixel counts were verified across reports. Lower values indicate better agreement on this validation set; they do not rank flux conservation, artifacts, or resolution.

| Model | F150W chi²/pixel | F277W chi²/pixel | F150W NLL/pixel | F277W NLL/pixel |
| --- | ---: | ---: | ---: | ---: |
| `sine_implicit` | 3.1429 | 14.2297 | -1.4989 | 2.9922 |
| `sasnet` | 3.4027 | 17.4560 | -1.3690 | 4.6054 |
| `spectral_power_foam` | 79.0977 | 227.2280 | 36.4785 | 109.4913 |
| `gaussian_basis` | 14.3187 | 51.9974 | 4.0890 | 21.8760 |
| `explicit_spectral_grid_field` | 32.4054 | 26.1807 | 13.1323 | 8.9677 |
| `linear_band_grid_field` | 1.7175 | 2.4826 | -2.2116 | -2.8813 |
| `sky_lambda_feature_field` | 4.3800 | 31.3759 | -0.8804 | 11.5653 |

The linear band grid has the lowest detector residual metrics among these fixed-budget fits, while its measured injection PSF is broader and has much stronger negative lobes than SWarp. Detector agreement alone is insufficient for selecting a reconstruction.

The comparison uses 127,972 F150W pixels and 31,368 F277W pixels, from two validation exposures per band. Test exposures were not evaluated. Likelihoods use a diagonal Gaussian with pipeline ERR and ignore pixel correlations; NLL is restored to the original MJy/sr scale. These reports contain no detector-forward score for SWarp, Drizzle, or Montage. Injection and effective-PSF scoring remains separate.

Source paths and SHA-256 identities are preserved in [detector-validation.json](detector-validation.json).
