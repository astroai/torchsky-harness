# Foam signed injection response audit

The fixed-budget Foam fit has a valid signed-flux diagnostic, but its effective PSF and matched detection score are unavailable because some injected-minus-blank source stamps have nonpositive total response. No positive-response clipping or forced normalization was applied. This rejects this fitted configuration on the response priorities; it is not a general conclusion about the model family.

Fluxes below use bilinearly registered 19×19 stamps with the border median subtracted, matching the existing effective-PSF helper. They measure local point-source response and cannot establish diffuse flux conservation.

| Batch | Band | Median flux bias | Flux NMAD | All stamp totals positive? |
| --- | --- | ---: | ---: | --- |
| validation-000 | F150W | -101.064% | 0.217% | False |
| validation-000 | F277W | -94.216% | 0.306% | True |
| dropout-000 | F150W | zero truth; unmeasured | unmeasured | False |
| dropout-000 | F277W | -91.034% | 5.272% | False |

The engineering dropout has zero injected F150W flux. Median absolute blue-band response is 1.109e-09 Jy and maximum absolute response is 3.057e-08 Jy. This is cross-band aperture leakage, not independent false-source performance.

Scoring status: `unavailable_nonpositive_injection_response`. The diagnostic job exited 0; the unavailable positive-PSF/detection measurement is explicitly recorded rather than called successful.

Original [signed-response.json](/Users/fabbros/src/astroai/torchsky/benchmarks_results/smacs-inr-20261010/runs/smacs-foam-response-audit-20261011-r1/signed-response.json) and [scoring-status.json](/Users/fabbros/src/astroai/torchsky/benchmarks_results/smacs-inr-20261010/runs/smacs-foam-response-audit-20261011-r1/scoring-status.json) preserve all source hashes and comparison controls.
