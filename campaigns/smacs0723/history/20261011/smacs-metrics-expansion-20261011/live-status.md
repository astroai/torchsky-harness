# SMACS job status snapshot

> This is an earlier audit record. The current read-only snapshot is in [latest-status.md](latest-status.md), and the complete launch/status ledger is [jobs-ledger.json](jobs-ledger.json).

Read-only CANFAR session inventory captured 2026-10-11 15:26:38 UTC. Active session rows include only names matching `smacs-*-20261011`, as requested. Session statuses may change after this snapshot.

## Active CANFAR sessions

| Session ID | Name | Status | Type |
| --- | --- | --- | --- |
| `t0b2bmz5` | `smacs-classical-f356w-inj512-20261011-r1` | Running | headless |
| `o7zvva10` | `smacs-classical-f444w-inj512-20261011-r1` | Running | headless |
| `ej6d53e3` | `smacs-map-truth-classical-20261011-r2` | Running | headless |

The earlier known-truth classical session ID `a644ry7v` was absent from this latest active-session list; this audit did not retrieve its remote log or establish its final result.

## Response128 field jobs

All seven response128 job logs and `exit_status.json` records were fetched to `benchmarks_results/smacs-inr-20261010/runs/<run-id>/`. Every job exited with code 1 before the benchmark started. Setup reached the editable `torchsky` wheel build/install; command 0 then imported `benchmarks/bench_multiband_depth.py` and failed at line 28. No model fit, response measurement, or training step ran.

| Model | Run ID | Exit | Failure |
| --- | --- | ---: | --- |
| `explicit_spectral_grid_field` | `smacs-explicit-spectral-grid-field-response128-20261011-r1` | 1 | `ModuleNotFoundError: No module named 'torchsky.map'` |
| `gaussian_basis` | `smacs-gaussian-basis-response128-20261011-r1` | 1 | `ModuleNotFoundError: No module named 'torchsky.map'` |
| `linear_band_grid_field` | `smacs-linear-band-grid-field-response128-20261011-r1` | 1 | `ModuleNotFoundError: No module named 'torchsky.map'` |
| `sasnet` | `smacs-sasnet-response128-20261011-r1` | 1 | `ModuleNotFoundError: No module named 'torchsky.map'` |
| `sine_implicit` | `smacs-sine-implicit-response128-20261011-r1` | 1 | `ModuleNotFoundError: No module named 'torchsky.map'` |
| `sky_lambda_feature_field` | `smacs-sky-lambda-feature-field-response128-20261011-r1` | 1 | `ModuleNotFoundError: No module named 'torchsky.map'` |
| `spectral_power_foam` | `smacs-spectral-power-foam-response128-20261011-r1` | 1 | `ModuleNotFoundError: No module named 'torchsky.map'` |

The sine session was associated with short CANFAR ID `j6lpkbj6`; its saved `exit_status.json` also reports exit code 1. The common traceback tail is:

```text
  File ".../benchmarks/bench_multiband_depth.py", line 28, in <module>
    from torchsky.map import ImageObservation, coadd_image_plane
ModuleNotFoundError: No module named 'torchsky.map'
exit_code= 1
```

No jobs were launched, retried, deleted, or modified during this audit. Active statuses above are a snapshot; no inference is made about the completion state of jobs omitted from the active list.

## Classical stacker and map-truth audit

Follow-up read-only audit captured 2026-10-11 15:29:19 UTC. A fresh `canfar ps` query showed no active session matching the six classical injection run names or map-truth r2. Fetched each run's `session.log` and `exit_status.json` through the existing control script; all seven exit with code 1.

| Run | Exit | Last reached stage |
| --- | ---: | --- |
| `smacs-classical-f090w-inj512-20261011-r1` | 1 | Montage bootstrap, CFITSIO build failure |
| `smacs-classical-f150w-inj512-20261011-r1` | 1 | Montage bootstrap, CFITSIO build failure |
| `smacs-classical-f200w-inj512-20261011-r1` | 1 | Montage bootstrap, CFITSIO build failure |
| `smacs-classical-f277w-inj512-20261011-r1` | 1 | Montage bootstrap, CFITSIO build failure |
| `smacs-classical-f356w-inj512-20261011-r1` | 1 | Montage bootstrap, CFITSIO build failure |
| `smacs-classical-f444w-inj512-20261011-r1` | 1 | Montage bootstrap, CFITSIO build failure |
| `smacs-map-truth-classical-20261011-r2` | 1 | Montage bootstrap, CFITSIO build failure |

Montage source clone and configure completed, but the bundled Montage build failed in CFITSIO 3.49 `putcol.c`: `nullptr` failed to parse, followed by undeclared `cptr` and lvalue errors; `make` returned 2. The runner reports `Montage build failed before mProject was produced`. This means the Montage bootstrap did not succeed, and no SWarp/Drizzle/Montage comparison or truth measurement ran.

The six stacker specs point at `/arc/projects/ots/torchsky-data/updec_webb/<band>b_UPDC-v2_iter0_i2d.fits.fz` and the shared target header `/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940/validation-000/common-support-r9-f0p99.fits`. The failure happened at command 0 before command 1, so neither reference path was opened or validated; their existence in the remote runtime remains unconfirmed.

Fetched logs and exit records are under `benchmarks_results/smacs-inr-20261010/runs/<run-id>/`.

## Live run audit: r2 classical and response128 cohort

Read-only `canfar ps --json` snapshot and artifact fetch around 2026-10-11 15:42 UTC. The existing controller fetched the requested `session.log` and `exit_status.json` artifacts; missing `exit_status.json` files were HTTP 404 and are recorded as not yet available, not as success. No job was launched, stopped, or modified.

| Run | Session | Snapshot / result |
| --- | --- | --- |
| `smacs-classical-f090w-inj512-20261011-r2` | `b6k14rjb` | Running; blank stacker command active; no `exit_status.json` or result JSON yet. |
| `smacs-classical-f150w-inj512-20261011-r2` | `stgnj5p3` | Running; blank stacker command active; no `exit_status.json` or result JSON yet. |
| `smacs-classical-f200w-inj512-20261011-r2` | `fi9yl4fh` | Running; blank stacker command active; no `exit_status.json` or result JSON yet. |
| `smacs-classical-f277w-inj512-20261011-r2` | `lhshqh72` | Running; blank stacker command active; no `exit_status.json` or result JSON yet. |
| `smacs-classical-f356w-inj512-20261011-r2` | `dvmd43nn` | Running; blank stacker command active; no `exit_status.json` or result JSON yet. |
| `smacs-classical-f444w-inj512-20261011-r2` | `n8ui6njp` | Running; blank stacker command active; no `exit_status.json` or result JSON yet. |
| `smacs-published-updc-map-metrics-20261011-r1` | `zzhdpyds` | Exit 1; `ModuleNotFoundError: No module named 'torchsky.map'`; no metrics JSON. |
| `smacs-model-known-truth-20261011-r1` | `acknhz34` | Exit 1; simulated observation failed on mixed `cuda:0` and CPU tensors; no `truth.json`. |
| `smacs-native-inputs-response128-20261011-r3` | `bn0odyoc` | Running; generated manifests with 12,100 support pixels and advanced to validation-001 export; no exit status yet. |
| `smacs-explicit-spectral-grid-field-response128-20261011-r2` | `diq799a6` | Running; blank explicit-grid fit started; no exit status yet. |
| `smacs-gaussian-basis-response128-20261011-r2` | `ch2c9hfj` | Running; blank Gaussian-basis fit started; no exit status yet. |
| `smacs-linear-band-grid-field-response128-20261011-r2` | `osnqkbtr` | Running; blank linear-band fit started; no exit status yet. |
| `smacs-sasnet-response128-20261011-r2` | `ksahtkqk` | Running; blank SASNet fit started; no exit status yet. |
| `smacs-sine-implicit-response128-20261011-r2` | `sc05fk39` | Running; blank sine-implicit fit started; no exit status yet. |
| `smacs-sky-lambda-feature-field-response128-20261011-r2` | `h558e8dh` | Exit 1; CUDA allocation OOM, followed by a PyTorch NVML allocator assertion. |
| `smacs-spectral-power-foam-response128-20261011-r2` | `hzwzjvpp` | Running; blank spectral-power fit started; no exit status yet. |

The six classical r2 logs show the Montage build returned `make rc=2`, but `mProjectPP` was compiled and linked, the bootstrap printed its install path, and the stacker command began. This partial build is not proof that Montage works on the real SMACS data. At this snapshot all six jobs are still in the blank stacker command, and their `stackers.json` files are absent; **no real-SMACS Montage row has yet been observed as either successful or failed**.

The response128 r2 sky-lambda job's compact error excerpt is: `memory allocation failed ... trying to allocate 46137344 bytes (free: 12582912)`; it then raised `NVML_SUCCESS == r INTERNAL ASSERT FAILED` and exited 1. The other six model-family r2 jobs remained active with blank-batch fitting underway.

Truth r3 is separately complete (`exit_status.json` exit 0, 36 result rows). Its Montage row for `tan_aligned_high_noise` is `OK`, but that aligned case takes a direct-copy path and does not exercise Montage. The non-aligned `tan_flux_multires` Montage row is `SKIPPED` with reason `Montage failed: Montage command failed (mProjectPP):` (empty diagnostic). This synthetic result therefore contains no successful Montage tool verification and must not be counted as a real-SMACS Montage row.

All refreshed logs and available exit records are under `benchmarks_results/smacs-inr-20261010/runs/<run-id>/`; the real-SMACS stacker result JSONs were not yet available at this snapshot.
