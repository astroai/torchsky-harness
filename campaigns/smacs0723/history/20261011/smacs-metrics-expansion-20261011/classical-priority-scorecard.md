# Complete classical comparison

SWarp Lanczos3 is the explicit resolution reference. These are validation-only, truth-tuned purity screens; the two cohorts have different inputs and cannot be pooled. Flux values use brightest-source injected-minus-blank, registered 19×19 stamps after border-median subtraction. They measure point-source response, not diffuse flux conservation.

## Backend and device provenance

The completed matched and raw-six-band classical scoring cohorts ran on CPU. The rows below identify the implementation behind each comparator and separate measured methods from attempted synthetic comparators that produced no measurements.

| Method / comparator | Backend in this comparison | Device | Evidence status |
| --- | --- | --- | --- |
| Torchsky bilinear, Lanczos3, conservative polygon, and Drizzle pixfrac variants | Native Torchsky map/coadd implementations | CPU | Scored in matched response128 and/or raw-six-band cohorts. |
| SWarp bilinear/Lanczos3 and nearest | Original SWarp executable through the benchmark adapter | CPU | Scored in matched response128, raw-six-band, and synthetic truth cohorts. |
| Official Drizzle pixfrac 0.6/0.8/1.0 | Official Drizzle implementation | CPU | Scored in matched response128; the synthetic truth run also evaluated point and square variants. |
| MontagePy `mProject` | Official MontagePy 2.3.0 Python binding | CPU | Scored in matched response128 and all six raw-aligned bands. |
| `reproject_interp` nearest/bilinear/bicubic | Optional `reproject` comparator | CPU attempt | All six synthetic truth rows (three kernels × two cases) were `SKIPPED`: optional dependency unavailable. No current real-source or injection score. |
| `dfreproject` nearest/bilinear/bicubic | Optional `dfreproject` comparator | CPU attempt | All six synthetic truth rows were `SKIPPED`: optional dependency unavailable. No current real-source or injection score. |
| CASUtools `imstack` | Optional CASUtools comparator | CPU attempt | Both synthetic truth rows were `SKIPPED`: `imstack` unavailable. |
| Synthetic Montage `mProjectPP` | Adapter's Montage command path | CPU | The aligned-case output is a direct copy and does not execute projection; the multiresolution row was `SKIPPED` after command failure. Separate scored MontagePy rows above use the official Python binding. |

The synthetic truth output has 36 records, of which 17 contain evaluated numerical outputs, four are informational recommendation rows, and 15 are skipped. It is not accurate to describe all 36 output records as successful comparator evaluations. No GPU stacker performance claim is made.

## calibrated-response128

Bands: F150W, F277W; shared support: 12,095 pixels. Selected by the existing validation rule: fitted_gaussian__swarp_lanczos3.

| Method | Matched / truth | Purity | m50 screen (AB) | Resolution gate |
| --- | ---: | ---: | ---: | --- |
| `coadd_bilinear` | 54/84 | 1.000 | 29.500 | fail |
| `torchsky_bilinear` | 54/84 | 1.000 | 29.333 | fail |
| `torchsky_lanczos3` | 53/84 | 1.000 | 29.250 | pass |
| `torchsky_conservative_polygon` | 53/84 | 1.000 | 29.250 | fail |
| `swarp_lanczos3` | 57/84 | 1.000 | 29.583 | pass |
| `torchsky_drizzle_pf0p600` | 53/84 | 1.000 | 29.250 | fail |
| `torchsky_drizzle_pf0p800` | 53/84 | 1.000 | 29.250 | fail |
| `torchsky_drizzle_pf1p000` | 53/84 | 1.000 | 29.250 | fail |
| `official_drizzle_pf0p600` | 53/84 | 1.000 | 29.250 | fail |
| `official_drizzle_pf0p800` | 53/84 | 1.000 | 29.250 | fail |
| `official_drizzle_pf1p000` | 53/84 | 1.000 | 29.250 | fail |
| `montagepy_mproject` | 53/84 | 1.000 | 29.250 | fail |

| Method | Band | FWHM (px) | Median point-source flux bias | Flux NMAD | Negative-lobe fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `coadd_bilinear` | F150W | 1.4043 | +3.762% | 9.688% | 0.000000 |
| `coadd_bilinear` | F277W | 2.0864 | +0.631% | 0.355% | 0.000000 |
| `torchsky_bilinear` | F150W | 1.4042 | +5.537% | 11.215% | 0.000002 |
| `torchsky_bilinear` | F277W | 2.0864 | -0.053% | 2.261% | 0.000001 |
| `torchsky_lanczos3` | F150W | 1.3190 | +3.121% | 18.976% | 0.000420 |
| `torchsky_lanczos3` | F277W | 1.8794 | +0.368% | 1.096% | 0.008214 |
| `torchsky_conservative_polygon` | F150W | 1.5074 | -1.574% | 0.079% | 0.000003 |
| `torchsky_conservative_polygon` | F277W | 2.0813 | -0.088% | 0.642% | 0.000001 |
| `swarp_lanczos3` | F150W | 1.4303 | -1.595% | 0.289% | 0.000252 |
| `swarp_lanczos3` | F277W | 1.8803 | +0.388% | 1.138% | 0.008276 |
| `torchsky_drizzle_pf0p600` | F150W | 1.4854 | -2.603% | 0.599% | 0.000003 |
| `torchsky_drizzle_pf0p600` | F277W | 2.0087 | +0.478% | 0.129% | 0.000001 |
| `torchsky_drizzle_pf0p800` | F150W | 1.4950 | -2.152% | 0.012% | 0.000004 |
| `torchsky_drizzle_pf0p800` | F277W | 2.0395 | +0.076% | 0.386% | 0.000001 |
| `torchsky_drizzle_pf1p000` | F150W | 1.5074 | -1.574% | 0.079% | 0.000003 |
| `torchsky_drizzle_pf1p000` | F277W | 2.0813 | -0.088% | 0.642% | 0.000001 |
| `official_drizzle_pf0p600` | F150W | 1.4854 | -2.603% | 0.599% | 0.000003 |
| `official_drizzle_pf0p600` | F277W | 2.0087 | +0.478% | 0.129% | 0.000001 |
| `official_drizzle_pf0p800` | F150W | 1.4950 | -2.152% | 0.011% | 0.000004 |
| `official_drizzle_pf0p800` | F277W | 2.0395 | +0.076% | 0.386% | 0.000001 |
| `official_drizzle_pf1p000` | F150W | 1.5074 | -1.574% | 0.079% | 0.000003 |
| `official_drizzle_pf1p000` | F277W | 2.0813 | -0.088% | 0.642% | 0.000001 |
| `montagepy_mproject` | F150W | 1.5136 | -1.451% | 0.108% | 0.000002 |
| `montagepy_mproject` | F277W | 2.0844 | -0.141% | 0.914% | 0.000001 |

Original result: [smacs-matched-classical-metrics-20261011-r3](/Users/fabbros/src/astroai/torchsky/benchmarks_results/smacs-inr-20261010/runs/smacs-matched-classical-metrics-20261011-r3/scoring/model_selection.json).

## raw-sixband512

Bands: F090W, F150W, F200W, F277W, F356W, F444W; shared support: 234,535 pixels. Selected by the existing validation rule: none.

| Method | Matched / truth | Purity | m50 screen (AB) | Resolution gate |
| --- | ---: | ---: | ---: | --- |
| `torchsky_bilinear` | 77/84 | 1.000 | unbracketed | fail |
| `torchsky_lanczos3` | 80/84 | 1.000 | unbracketed | fail |
| `torchsky_drizzle_square` | 77/84 | 1.000 | unbracketed | fail |
| `torchsky_conservative_polygon` | 77/84 | 1.000 | unbracketed | fail |
| `drizzle_square` | 77/84 | 1.000 | unbracketed | fail |
| `swarp_lanczos3` | 79/84 | 1.000 | unbracketed | pass |
| `montagepy_mproject` | 77/84 | 1.000 | unbracketed | fail |

The completeness curves do not cross 50% within the tested AB 27–30 range. The scorer produced no finite m50 bootstrap draws; those values are explicitly unavailable. No extrapolated depth or winning depth comparison is inferred.

| Method | Band | FWHM (px) | Median point-source flux bias | Flux NMAD | Negative-lobe fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `torchsky_bilinear` | F090W | 1.2308 | -16.135% | 5.693% | 0.000000 |
| `torchsky_bilinear` | F150W | 1.4289 | -6.299% | 3.407% | 0.000000 |
| `torchsky_bilinear` | F200W | 1.5771 | +3.125% | 0.735% | 0.000000 |
| `torchsky_bilinear` | F277W | 2.1067 | -0.559% | 2.125% | 0.000000 |
| `torchsky_bilinear` | F356W | 2.4105 | +0.711% | 0.201% | 0.000000 |
| `torchsky_bilinear` | F444W | 2.7853 | -1.283% | 0.379% | 0.000000 |
| `torchsky_lanczos3` | F090W | 0.9361 | -45.788% | 22.102% | 0.014588 |
| `torchsky_lanczos3` | F150W | 1.3215 | -9.175% | 13.250% | 0.002427 |
| `torchsky_lanczos3` | F200W | 1.4896 | +0.044% | 3.845% | 0.000002 |
| `torchsky_lanczos3` | F277W | 1.8845 | -0.025% | 0.486% | 0.005559 |
| `torchsky_lanczos3` | F356W | 2.1969 | +0.216% | 0.239% | 0.000028 |
| `torchsky_lanczos3` | F444W | 2.6056 | -0.557% | 0.049% | 0.000000 |
| `torchsky_drizzle_square` | F090W | 1.4527 | -1.712% | 4.920% | 0.000000 |
| `torchsky_drizzle_square` | F150W | 1.5486 | +0.327% | 1.154% | 0.000000 |
| `torchsky_drizzle_square` | F200W | 1.6797 | +1.582% | 0.623% | 0.000000 |
| `torchsky_drizzle_square` | F277W | 2.1139 | -0.527% | 0.906% | 0.000000 |
| `torchsky_drizzle_square` | F356W | 2.4069 | +0.786% | 0.398% | 0.000000 |
| `torchsky_drizzle_square` | F444W | 2.7853 | -1.079% | 0.739% | 0.000000 |
| `torchsky_conservative_polygon` | F090W | 1.4527 | -1.712% | 4.920% | 0.000000 |
| `torchsky_conservative_polygon` | F150W | 1.5486 | +0.327% | 1.154% | 0.000000 |
| `torchsky_conservative_polygon` | F200W | 1.6797 | +1.582% | 0.623% | 0.000000 |
| `torchsky_conservative_polygon` | F277W | 2.1139 | -0.527% | 0.906% | 0.000000 |
| `torchsky_conservative_polygon` | F356W | 2.4069 | +0.786% | 0.398% | 0.000000 |
| `torchsky_conservative_polygon` | F444W | 2.7853 | -1.079% | 0.739% | 0.000000 |
| `drizzle_square` | F090W | 1.4527 | -1.712% | 4.920% | 0.000000 |
| `drizzle_square` | F150W | 1.5486 | +0.327% | 1.154% | 0.000000 |
| `drizzle_square` | F200W | 1.6797 | +1.582% | 0.623% | 0.000000 |
| `drizzle_square` | F277W | 2.1139 | -0.527% | 0.906% | 0.000000 |
| `drizzle_square` | F356W | 2.4069 | +0.786% | 0.398% | 0.000000 |
| `drizzle_square` | F444W | 2.7853 | -1.079% | 0.738% | 0.000000 |
| `swarp_lanczos3` | F090W | 1.3503 | -2.273% | 5.181% | 0.002375 |
| `swarp_lanczos3` | F150W | 1.4773 | -0.464% | 0.725% | 0.000041 |
| `swarp_lanczos3` | F200W | 1.6119 | +1.486% | 1.201% | 0.000000 |
| `swarp_lanczos3` | F277W | 1.8834 | +0.259% | 0.088% | 0.005681 |
| `swarp_lanczos3` | F356W | 2.1887 | -0.072% | 0.023% | 0.000028 |
| `swarp_lanczos3` | F444W | 2.6032 | -0.034% | 0.216% | 0.000000 |
| `montagepy_mproject` | F090W | 1.4513 | -2.516% | 5.716% | 0.000000 |
| `montagepy_mproject` | F150W | 1.5469 | -1.013% | 0.549% | 0.000000 |
| `montagepy_mproject` | F200W | 1.6810 | +1.892% | 0.883% | 0.000000 |
| `montagepy_mproject` | F277W | 2.1177 | -0.530% | 4.703% | 0.000000 |
| `montagepy_mproject` | F356W | 2.4085 | +0.760% | 0.472% | 0.000000 |
| `montagepy_mproject` | F444W | 2.7857 | -1.114% | 0.693% | 0.000000 |

Original result: [smacs-classical-sixband-metrics-20261011-r4](/Users/fabbros/src/astroai/torchsky/benchmarks_results/smacs-inr-20261010/runs/smacs-classical-sixband-metrics-20261011-r4/scoring/model_selection.json).
