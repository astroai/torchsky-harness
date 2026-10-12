# Latest SMACS session status

Last read-only filtered CANFAR process snapshot: **2026-10-11 22:17 UTC**. It contains only SMACS ledger session IDs and their id/status/name/requested-resource fields; see [canfar-ps-smacs-20261011-2217.json](canfar-ps-smacs-20261011-2217.json). Bounded model-log checks continued through 22:14 UTC. Captured result artifacts for the ordinary-response scorer were checked at 22:28 UTC; no newer process snapshot was read.

The ledger has 70 records (52 launched, 18 setup failures); six SMACS sessions were listed Running in the 22:17 snapshot. The ready-model scorer (z5d4ls9p) exited 0. Its sine implicit and Gaussian basis scores are scientifically unavailable because their ordinary injection responses do not produce positive finite effective-PSF stamps; signed response records are retained. The joint score completed over coadd plus eleven classical methods, with `fitted_gaussian__swarp_lanczos3` in the actual `selected` field. This is a provisional control-only selection: five other model families are excluded and dropout response was not measured. The six-band r4 and matched-classical r3 scorer retries and the Foam signed-response audit also have exit-code-0 artifacts. Matched-metrics recovery r3 (wkgscklq) remains separate and is still waiting for Sine/Gaussian dropout products and SASNet/Sky Lambda ordinary products before its all-seven joint score. Recovery r2 (wmclm1ba) and r3 were Running in the 22:17 snapshot; an attempted deletion of r2 was rejected while r3 remained partially validated, and no retry was made.

Bounded fit-log reads at 22:14 UTC give the latest observed batches: Gaussian basis, dropout-000 step 90/120 (loss 0.100691); SASNet, validation-000 step 60/120 (loss 0.0345071); sine implicit, dropout-000 step 90/120 (loss 0.0239293); Sky Lambda, validation-000 step 90/120 (loss 0.046131). Each exit-status artifact returned HTTP 404, while all four fit sessions were listed Running in the 22:17 snapshot. Recovery r3 uses the frozen source and 8 CPU / 64 GB / 0 GPU. Recovery r2 remains Running in the same snapshot.

| Session | Run | GPU | Latest observed stage / status |
| --- | --- | ---: | --- |
| ch2c9hfj | smacs-gaussian-basis-response128-20261011-r2 | 1 | Running at 22:17; latest bounded log 22:14: dropout-000 step 90/120, loss 0.100691. |
| ksahtkqk | smacs-sasnet-response128-20261011-r2 | 1 | Running at 22:17; latest bounded log 22:14: validation-000 step 60/120, loss 0.0345071. |
| sc05fk39 | smacs-sine-implicit-response128-20261011-r2 | 1 | Running at 22:17; latest bounded log 22:14: dropout-000 step 90/120, loss 0.0239293. |
| oq1zxrx2 | smacs-sky-lambda-feature-field-response128-20261011-r3 | 0 | Running at 22:17; latest bounded log 22:14: validation-000 step 90/120, loss 0.046131. |
| wmclm1ba | smacs-matched-metrics-recovery-20261011-r2 | 0 | Running at 22:17; prior progress has partial model scores and a per-model Foam scoring failure. |
| wkgscklq | smacs-matched-metrics-recovery-20261011-r3 | 0 | Running at 22:17; two models scored, signed Foam response retained, four models and joint score pending. |

The r3 progress records Foam as scientifically unavailable because an effective-PSF injection response is nonpositive; signed diagnostics are retained and no positive normalization was forced. Its four remaining models are waiting for fit products, and joint scoring is pending. The r2 coordinator remains a separate Running session in the latest snapshot.

## Verified completions, failures, partial scores, and new launches

| Run / session | Result |
| --- | --- |
| smacs-classical-sixband-metrics-20261011-r1 / guue8u7s | Exit 1 after all six bands were staged; strict JSON serialization rejected a NaN summary value. |
| smacs-classical-sixband-metrics-20261011-r2 / jc3l6re3 | Exit 1 after all 168 products were staged; same strict JSON NaN failure. |
| smacs-classical-sixband-metrics-20261011-r3 / fwmpnfgj | Retry stopped before scoring with ModuleNotFoundError: bench_multiband_depth. |
| smacs-classical-sixband-metrics-20261011-r4 / wqu9ol11 | Exit 0; seven methods scored over 234,535 common-support pixels. No method was selected because all m50 bootstrap summaries had zero finite draws. |
| smacs-matched-classical-metrics-20261011-r1 / afy19t3d | Exit 0; provenance records 240 product hashes. No standalone scorer JSON is present in this local capture. |
| smacs-matched-classical-metrics-20261011-r2 / dm04xvh4 | Retry verified prior provenance, then stopped before scoring with ModuleNotFoundError: bench_multiband_depth. |
| smacs-matched-classical-metrics-20261011-r3 / o7pzta4u | Exit 0; twelve methods scored over 12,095 common-support pixels; SWarp Lanczos3 selected under the resolution gate. |
| Raw-aligned MontagePy r1, F090W/F150W/F200W/F277W/F356W/F444W | All six per-band runs have exit-code-0 artifacts and validation score JSONs. |
| Raw-aligned classical F150W r3 / lccgys6p | Exit 0 with blank, three validation batches, and effective-PSF response. All six bands now have complete r3 classical response products. |
| Explicit spectral-grid response128 r2 / diq799a6 | Exit 1 after the fixed-budget fit; null-response assembly lacked the blank F150W coadd-bilinear prediction. A separate recovery score covers this exact configuration. |
| Spectral-power Foam response128 r2 / hzwzjvpp | Exit 1 after the fixed-budget fit; null-response assembly also lacked the blank coadd-bilinear prediction. Recovery progress separately records a model-specific PSF-scoring failure. |
| Foam response audit r1 / pd5mvc6h | Exit 0 after retaining signed response. Positive-response scoring is unavailable because the F150W validation stamp also had nonpositive total flux; see foam-response-audit.md. |
| Matched-metrics recovery r3 / wkgscklq | Running at 22:17 UTC. Linear and explicit model score JSONs and signed Foam diagnostics are present; four models await fit products and joint scoring is pending. Provenance verifies 240 control and 30 model science products by matching source/staged hashes. |
| smacs-ready-model-responses-20261011-r1 / z5d4ls9p | Exit 0 in captured artifact. Both model scores are `unavailable_nonpositive_injection_response`; joint score selected SWarp Lanczos3 among 12 coadd/classical controls. Dropout response was not measured. |

Detailed score tables and limitations are in [completed-results.md](completed-results.md). Paired detector-residual results are in [detector-validation.md](detector-validation.md). The seven model families' blank detector reports use identical frozen calibration and validation exposure identities, but remain validation-only, use diagonal pipeline ERR without pixel-correlation modeling, and have no SWarp detector-forward score.

This inventory update is read-only with respect to CANFAR sessions. Session creation alone does not prove completion; missing from ps alone does not prove exit.
