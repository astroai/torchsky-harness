# SMACS metrics-expansion run inventory

Last read-only filtered CANFAR process snapshot: **2026-10-11 22:17 UTC**. The capture contains only SMACS ledger session IDs and their id/status/name/requested-resource fields; see [canfar-ps-smacs-20261011-2217.json](canfar-ps-smacs-20261011-2217.json). Local model-log checks continued through 22:14 UTC.

Inventory: **70 records** — 52 launched, 18 setup failures before scientific work, and 0 prepared but unlaunched. Six SMACS sessions were listed Running in the 22:17 snapshot. The ready-model ordinary scorer z5d4ls9p now has an exit-code-0 artifact: its provisional joint selection contains only the 12 matched control methods, while sine implicit and Gaussian basis are scientifically unavailable because their injection response does not yield positive finite effective-PSF stamps. The six-band and matched-classical scorer retries and the Foam signed-response audit also have exit-code-0 artifacts. Matched-metrics recovery r3 (wkgscklq) is still waiting for dropout products for sine/Gaussian and ordinary products for SASNet/Sky Lambda before its all-seven joint score. Both recovery r2 (wmclm1ba) and r3 were Running in the 22:17 snapshot; a deletion request for r2 was rejected while r3 was only partially validated, so no session was removed or retried.

All launched records use frozen source SHA-256 3fa81d201d234a0e17c49ef0234addb485bc85c03cc33d6e7073bbf1dc72c5c7 (manifest identity a0ed9e15a793ba28a751ac46cd8d43372134a7375c7d2149eb6a9b04cf9d25e4). Per-run cohorts, bands, batches, resources, planned metrics, input/source hashes, frozen specs, exit evidence, and artifact paths are in jobs-ledger.json. See latest-status.md for the last active-session snapshot and completed-results.md for verified measurements.

## Sessions listed Running at the 22:17 UTC snapshot

| Session | Run | GPU | Observed stage |
| --- | --- | ---: | --- |
| ch2c9hfj | smacs-gaussian-basis-response128-20261011-r2 | 1 | Latest bounded log 22:14: dropout-000 step 90/120, loss 0.100691. |
| ksahtkqk | smacs-sasnet-response128-20261011-r2 | 1 | Latest bounded log 22:14: validation-000 step 60/120, loss 0.0345071. |
| sc05fk39 | smacs-sine-implicit-response128-20261011-r2 | 1 | Latest bounded log 22:14: dropout-000 step 90/120, loss 0.0239293. |
| oq1zxrx2 | smacs-sky-lambda-feature-field-response128-20261011-r3 | 0 | Latest bounded log 22:14: validation-000 step 90/120, loss 0.046131. |
| wmclm1ba | smacs-matched-metrics-recovery-20261011-r2 | 0 | Running; previous progress has linear/explicit scores and a per-model Foam scoring failure. No retirement recorded. |
| wkgscklq | smacs-matched-metrics-recovery-20261011-r3 | 0 | Running; linear and explicit model scores written, Foam response retained as scientifically unavailable, four fits and joint score pending. |

The six-band and matched-classical scorer retries and the Foam signed-response audit are verified from local exit/result artifacts. The filtered 22:17 snapshot records both recovery coordinators Running; current r3 progress and evidence are summarized below and in completed-results.md.

The matched-metrics recovery r3 scorer (wkgscklq) was launched at 22:11 UTC with the frozen source and 8 CPU / 64 GB / 0 GPU. It has scored linear_band_grid_field and explicit_spectral_grid_field, retained signed Foam diagnostics with positive-PSF scoring unavailable, and is waiting for four fit product sets before joint scoring. Provenance verifies 240 control products and 30 model science products against their source hashes. Recovery r2 (wmclm1ba) also remains Running in the 22:17 snapshot; its deletion request was rejected during r3 partial validation, and no retry was made. Bounded reads of four fit logs at 22:14 UTC found: Gaussian basis on dropout-000, step 90/120 (loss 0.100691); SASNet on validation-000, step 60/120 (loss 0.0345071); sine implicit on dropout-000, step 90/120 (loss 0.0239293); and Sky Lambda on validation-000, step 90/120 (loss 0.046131). All four exit-status fetches returned HTTP 404, and CANFAR listed all four sessions Running at 22:17 UTC.

The ready-model ordinary scorer (z5d4ls9p) was launched at 22:24 UTC with 8 CPU / 64 GB / 0 GPU and a 10,800-second limit; its captured exit artifact reports code 0. It covers blank and three validation batches for sine implicit and Gaussian basis, but both model scores are scientifically unavailable because the unchanged scorer rejects nonpositive effective-PSF stamp totals. The signed validation responses remain recorded. Its joint selection used only coadd plus the eleven classical methods and selected SWarp Lanczos3; the five other model families were excluded from this provisional comparison. Dropout response remains pending and was not measured. See completed-results.md for the signed flux biases and validation limits.

## Recent scientific and operational outcomes

- The raw-aligned six-band scorer r4 (wqu9ol11) exited 0 after auditing the prior staged products and scoring seven methods on 234,535 common-support pixels. Every method had target-purity detections, but the three-batch m50 bootstrap had zero finite draws for every method, so the selection is null; this does not support a depth ranking. The earlier r1/r2 scorers reached the same numerical state but failed strict JSON serialization on NaN. r3 stopped before scoring with a missing bench_multiband_depth import.
- The matched calibrated response128 classical scorer r3 (o7pzta4u) exited 0 and scored twelve methods on 12,095 common-support pixels. SWarp Lanczos3 was selected under the effective-PSF resolution gate. Matched-metrics r1 exited 0 and recorded 240 product hashes, but its local capture has no standalone scorer JSON; r2 stopped before scoring on the same missing module import.
- The prior matched-model recovery coordinator (wmclm1ba) remains Running in the 22:17 snapshot with its earlier partial scores and Foam per-model failure. Replacement r3 (wkgscklq) is also Running, has scored linear and explicit models, retained signed Foam diagnostics while marking positive-response scoring unavailable, and is waiting for four models plus joint scoring.
- All six raw-aligned MontagePy band scorers and the F150W classical r3 run now have exit-code-0 artifacts. The raw MontagePy and SWarp comparisons are in the raw-aligned 512-pixel cohort; their validation m50 results are unavailable or descriptive only as detailed in completed-results.md.
- Explicit spectral-grid response128 r2 completed its fixed-budget fit, then its own null-response assembly stopped because the blank coadd-bilinear F150W product was missing. A separate recovery score exists for that exact configuration; it fails resolution and bright-source flux-response checks and is not a family-wide conclusion.
- The Foam response audit (pd5mvc6h) exited 0 and retained signed response plus scoring-status artifacts. Positive-response scoring was unavailable because a registered effective-PSF stamp had nonpositive total flux; the audit did not force normalization. See foam-response-audit.md for the signed-flux measurements and limits.
- Published UPDC static maps, historical raw-aligned six-band 512-pixel injections, and calibrated matched F150W/F277W response128 experiments remain separate cohorts. depth_claim remains false throughout.

## All records

| Run ID | Status | Session | CPU / memory GB / GPU | Exit | Latest observed stage |
| --- | --- | --- | ---: | ---: | --- |
| smacs-classical-f090w-inj512-20261011-r1 | setup_failed | jc04gdy4 | 8 / 64 / 0 | 1 | Montage bootstrap failed before stacker benchmark; no scientific measurements. |
| smacs-classical-f090w-inj512-20261011-r2 | launched | b6k14rjb | 8 / 64 / 0 | 1 | Exited nonzero; inspect local session log. |
| smacs-classical-f090w-inj512-20261011-r3 | launched | fm7zg9en | 8 / 64 / 0 | 0 | Completed blank + three validation batches and brightest-source effective PSF; exit 0. |
| smacs-classical-f150w-inj512-20261011-r1 | setup_failed | mu1rvcdy | 8 / 64 / 0 | 1 | Montage bootstrap failed before stacker benchmark; no scientific measurements. |
| smacs-classical-f150w-inj512-20261011-r2 | launched | stgnj5p3 | 8 / 64 / 0 | 1 | Exited nonzero; inspect local session log. |
| smacs-classical-f150w-inj512-20261011-r3 | launched | lccgys6p | 8 / 64 / 0 | 0 | Completed blank plus three validation batches and brightest-source effective PSF; six non-Montage stackers completed. |
| smacs-classical-f200w-inj512-20261011-r1 | setup_failed | e4mrpp95 | 8 / 64 / 0 | 1 | Montage bootstrap failed before stacker benchmark; no scientific measurements. |
| smacs-classical-f200w-inj512-20261011-r2 | launched | fi9yl4fh | 8 / 64 / 0 | 1 | Exited nonzero; inspect local session log. |
| smacs-classical-f200w-inj512-20261011-r3 | launched | zybi2t8m | 8 / 64 / 0 | 0 | Completed blank + three validation batches and brightest-source effective PSF; exit 0. |
| smacs-classical-f277w-inj512-20261011-r1 | setup_failed | vjmegb8i | 8 / 64 / 0 | 1 | Montage bootstrap failed before stacker benchmark; no scientific measurements. |
| smacs-classical-f277w-inj512-20261011-r2 | launched | lhshqh72 | 8 / 64 / 0 | 1 | Exited nonzero; inspect local session log. |
| smacs-classical-f277w-inj512-20261011-r3 | launched | tq4wgqqc | 8 / 64 / 0 | 0 | Completed blank + three validation batches and brightest-source effective PSF. |
| smacs-classical-f356w-inj512-20261011-r1 | setup_failed | t0b2bmz5 | 8 / 64 / 0 | 1 | Montage bootstrap failed before stacker benchmark; no scientific measurements. |
| smacs-classical-f356w-inj512-20261011-r2 | launched | dvmd43nn | 8 / 64 / 0 | 1 | Exited nonzero; inspect local session log. |
| smacs-classical-f356w-inj512-20261011-r3 | launched | wdgiwq1k | 8 / 64 / 0 | 0 | Completed blank + three validation batches and brightest-source effective PSF. |
| smacs-classical-f444w-inj512-20261011-r1 | setup_failed | o7zvva10 | 8 / 64 / 0 | 1 | Montage bootstrap failed before stacker benchmark; no scientific measurements. |
| smacs-classical-f444w-inj512-20261011-r2 | launched | n8ui6njp | 8 / 64 / 0 | 1 | Exited nonzero; inspect local session log. |
| smacs-classical-f444w-inj512-20261011-r3 | launched | njdw0lc8 | 8 / 64 / 0 | 0 | Completed blank + three validation batches and brightest-source effective PSF. |
| smacs-classical-sixband-metrics-20261011-r1 | launched | guue8u7s | 8 / 64 / 0 | 1 | All six bands staged; scorer failed only while strict JSON serialization rejected NaN summary value. |
| smacs-classical-sixband-metrics-20261011-r2 | launched | jc3l6re3 | 8 / 64 / 0 | 1 | All 168 products staged across six bands; scorer then failed strict JSON serialization on NaN. |
| smacs-classical-sixband-metrics-20261011-r3 | launched | fwmpnfgj | 8 / 64 / 0 | 1 | Serializer recovery retry stopped before scoring because bench_multiband_depth could not be imported. |
| smacs-classical-sixband-metrics-20261011-r4 | launched | wqu9ol11 | 8 / 64 / 0 | 0 | Exit 0; scored all seven methods on 234,535 common-support pixels. No method selected because m50 had zero finite bootstrap draws for every method. |
| smacs-explicit-spectral-grid-field-response128-20261011-r1 | setup_failed | hweducpz | 8 / 64 / 1 | 1 | Setup/preflight failed before scientific work; no planned measurements obtained. |
| smacs-explicit-spectral-grid-field-response128-20261011-r2 | launched | diq799a6 | 8 / 64 / 1 | 1 | 120-step fit completed; null-response assembly stopped because blank fitted_gaussian/F150W/coadd_bilinear/prediction.fits was missing. |
| smacs-foam-response-audit-20261011-r1 | launched | pd5mvc6h | 4 / 16 / 0 | 0 | Audit exited 0; signed-response artifacts captured; positive-response score unavailable due to nonpositive effective-PSF flux. |
| smacs-gaussian-basis-response128-20261011-r1 | setup_failed | e1jkq0tg | 8 / 64 / 1 | 1 | Setup/preflight failed before scientific work; no planned measurements obtained. |
| smacs-gaussian-basis-response128-20261011-r2 | launched | ch2c9hfj | 8 / 64 / 1 | — | Latest bounded log read 22:14 UTC: dropout-000 step 90/120; loss 0.100691; exit artifact not found. |
| smacs-linear-band-grid-field-response128-20261011-r1 | setup_failed | cbifhmfn | 8 / 64 / 1 | 1 | Setup/preflight failed before scientific work; no planned measurements obtained. |
| smacs-linear-band-grid-field-response128-20261011-r2 | launched | osnqkbtr | 8 / 64 / 1 | 1 | Exited nonzero; inspect captured session log. |
| smacs-map-truth-classical-20261011-r1 | launched | a644ry7v | 8 / 32 / 0 | 0 | Completed; see captured result artifacts. |
| smacs-map-truth-classical-20261011-r2 | setup_failed | ej6d53e3 | 8 / 32 / 0 | 1 | Montage bootstrap failed before benchmark; no truth-comparison measurements. |
| smacs-map-truth-classical-20261011-r3 | launched | qt42i4qu | 8 / 32 / 0 | 0 | Completed; see captured result artifacts. |
| smacs-map-truth-montagepy-20261011-r1 | launched | ewpr0sg8 | 8 / 32 / 0 | — | Not listed in latest CANFAR snapshot; no exit artifact captured (completion unknown). |
| smacs-matched-classical-metrics-20261011-r1 | launched | afy19t3d | 8 / 64 / 0 | 0 | Exit 0; provenance verifies 240 source products. No standalone scorer JSON is present in this local run capture. |
| smacs-matched-classical-metrics-20261011-r2 | launched | dm04xvh4 | 8 / 64 / 0 | 1 | Retry verified prior 240-product provenance, then stopped before scoring because bench_multiband_depth could not be imported. |
| smacs-matched-classical-metrics-20261011-r3 | launched | o7pzta4u | 8 / 64 / 0 | 0 | Exit 0; scored all twelve classical/coadd methods on 12,095 common-support pixels; SWarp Lanczos3 was selected under the resolution gate. |
| smacs-matched-classical-response128-20261011-r1 | launched | ujcx1r9o | 8 / 64 / 0 | — | No local completion artifact; see current session status. |
| smacs-matched-classical-response128-20261011-r2 | launched | aeqfyr1n | 8 / 64 / 0 | 1 | Exited nonzero; inspect captured session log. |
| smacs-matched-classical-response128-20261011-r3 | launched | ipx8xqra | 8 / 64 / 0 | 1 | Wrote 100 product rows and 20 null-response rows; later scorer stopped on support-mask dtype mismatch. |
| smacs-matched-metrics-recovery-20261011-r1 | launched | me9vj96v | 8 / 64 / 0 | 1 | Coordinator stopped validating classical exit state; no recovered scoring output. |
| smacs-matched-metrics-recovery-20261011-r2 | launched | wmclm1ba | 8 / 64 / 0 | — | Running at 22:17; earlier partial scores retained; deletion rejected while r3 remains partially validated. |
| smacs-matched-metrics-recovery-20261011-r3 | launched | wkgscklq | 8 / 64 / 0 | — | Running at 22:17; two models scored, Foam positive-response score unavailable with signed record retained, four fits and joint score pending. |
| smacs-matched-montagepy-response128-20261011-r1 | launched | vchvvoij | 8 / 64 / 0 | 0 | Completed two-method MontagePy/SWarp validation scorer; 4 effective-PSF rows and model-selection JSON captured; depth_claim remains false. |
| smacs-model-known-truth-20261011-r1 | setup_failed | acknhz34 | 8 / 64 / 1 | 1 | Input setup failed before model fit or truth scoring. |
| smacs-model-known-truth-20261011-r2 | launched | algzfqy0 | 8 / 64 / 0 | 0 | Completed; see captured result artifacts. |
| smacs-montage-projection-diagnostic-20261011-r1 | launched | cp6sqo0t | 8 / 32 / 0 | 0 | Completed; see captured result artifacts. |
| smacs-montagepy-f090w-inj512-20261011-r1 | launched | e3tqoh96 | 8 / 64 / 0 | 0 | Completed raw-aligned MontagePy band run and common-support validation score; exit 0. |
| smacs-montagepy-f150w-inj512-20261011-r1 | launched | cspxdgh1 | 8 / 64 / 0 | 0 | Completed raw-aligned MontagePy band run and common-support validation score; exit 0. |
| smacs-montagepy-f200w-inj512-20261011-r1 | launched | gdbc5bkn | 8 / 64 / 0 | 0 | Completed raw-aligned MontagePy band run and common-support validation score; exit 0. |
| smacs-montagepy-f277w-inj512-20261011-r1 | launched | pzrqorud | 8 / 64 / 0 | 0 | Completed raw-aligned MontagePy band run and common-support validation score; exit 0. |
| smacs-montagepy-f356w-inj512-20261011-r1 | launched | uspinfb6 | 8 / 64 / 0 | 0 | Completed raw-aligned MontagePy band run and common-support validation score; exit 0. |
| smacs-montagepy-f444w-inj512-20261011-r1 | launched | u3rjykq6 | 8 / 64 / 0 | 0 | Completed raw-aligned MontagePy band run and common-support validation score; exit 0. |
| smacs-montagepy-projection-check-20261011-r1 | launched | luc9qu9o | 8 / 32 / 0 | 0 | MontagePy projection calls succeeded on both diagnostic inputs. |
| smacs-native-inputs-response128-20261011-r1 | setup_failed | j6328qtc | 8 / 64 / 0 | 1 | Setup/preflight failed before scientific work; no planned measurements obtained. |
| smacs-native-inputs-response128-20261011-r2 | setup_failed | u104sh4v | 8 / 64 / 0 | 1 | Manifest/preflight failed before native-input export. |
| smacs-native-inputs-response128-20261011-r3 | launched | bn0odyoc | 8 / 64 / 0 | 0 | Completed; see captured result artifacts. |
| smacs-published-static-integrity-20261011-r1 | launched | qlvja5yb | 8 / 32 / 0 | 0 | Completed 12-row unit-aware map diagnostics including Jy sums and offset/contrast fits. |
| smacs-published-updc-best-stack-20261011-r1 | launched | w433ta2s | 8 / 32 / 0 | 0 | Completed 12-row UPDC-vs-archived-SWarp common-support comparison. |
| smacs-published-updc-map-metrics-20261011-r1 | setup_failed | zzhdpyds | 8 / 64 / 0 | 1 | Import/preflight failed before scientific comparison. |
| smacs-published-updc-map-metrics-20261011-r2 | launched | tmm6jorq | 8 / 64 / 0 | 0 | Completed; see captured result artifacts. |
| smacs-ready-model-responses-20261011-r1 | launched | z5d4ls9p | 8 / 64 / 0 | 0 | Exit 0; both model scores unavailable from nonpositive effective-PSF stamp totals; signed responses retained; provisional control-only joint score selected SWarp Lanczos3; dropout not measured. |
| smacs-sasnet-response128-20261011-r1 | setup_failed | whp3mfgf | 8 / 64 / 1 | 1 | Setup/preflight failed before scientific work; no planned measurements obtained. |
| smacs-sasnet-response128-20261011-r2 | launched | ksahtkqk | 8 / 64 / 1 | — | Latest bounded log read 22:14 UTC: validation-000 step 60/120; loss 0.0345071; exit artifact not found. |
| smacs-sine-implicit-response128-20261011-r1 | setup_failed | j6lpkbj6 | 8 / 64 / 1 | 1 | Setup/preflight failed before scientific work; no planned measurements obtained. |
| smacs-sine-implicit-response128-20261011-r2 | launched | sc05fk39 | 8 / 64 / 1 | — | Latest bounded log read 22:14 UTC: dropout-000 step 90/120; loss 0.0239293; exit artifact not found. |
| smacs-sky-lambda-feature-field-response128-20261011-r1 | setup_failed | h9dzb2vi | 8 / 64 / 1 | 1 | Setup/preflight failed before scientific work; no planned measurements obtained. |
| smacs-sky-lambda-feature-field-response128-20261011-r2 | launched | h558e8dh | 8 / 64 / 1 | 1 | Exited nonzero; inspect captured session log. |
| smacs-sky-lambda-feature-field-response128-20261011-r3 | launched | oq1zxrx2 | 8 / 64 / 0 | — | Latest bounded log read 22:14 UTC: validation-000 step 90/120; loss 0.046131; exit artifact not found. |
| smacs-spectral-power-foam-response128-20261011-r1 | setup_failed | fxbe2wpf | 8 / 64 / 1 | 1 | Setup/preflight failed before scientific work; no planned measurements obtained. |
| smacs-spectral-power-foam-response128-20261011-r2 | launched | hzwzjvpp | 8 / 64 / 1 | 1 | 120-step fit completed; null-response assembly lacked blank fitted_gaussian/F150W/coadd_bilinear prediction. Recovery progress marks this model's scoring_failed status. |

Prepared specs have no session IDs or measurement evidence. The ledger distinguishes a captured exit, a session observed Running, and a session absent from the latest process snapshot.
