# SMACS INR discovery bake-off — 2026-10-10

The owner authorized autonomous CANFAR headless experiments on 2026-10-10,
with priorities in this order:

1. Discovery SNR and faint-source recovery at a calibrated false-positive rate,
   including sources absent in blue bands. There is no frozen redshift list;
   undiscovered high-redshift objects are the intended beneficiaries.
2. Independently measured resolution at least as good as matched Drizzle or
   UPdec. Output grid spacing and visual sharpness do not establish this.
3. Source and diffuse flux conservation, with leakage and ringing visible.

This is exploratory reconstruction work. It does not advance cosmodist's
scientific certification or authorize downstream cosmological estimators.
Use Luna 6 at xhigh for simple changes and Sol 6.1 at xhigh for numerical work.
No commit or push is authorized.

## Recovered numerical comparator

The October 7 cold-start PCG run
`smacs-linear-val-footprint512-20261007-r4` improves the reported smooth-continuum
injection completeness and detector fit. Its selection report used the legacy
Adam grid as its effective-resolution reference. On the same report's matched
bilinear coadd comparison, its effective FWHM fails in both bands:

| Band | PCG grid FWHM (px) | Same-input coadd FWHM (px) | Ratio |
| --- | ---: | ---: | ---: |
| F150W | 3.6524 | 1.4289 | 2.5561 |
| F277W | 3.5299 | 2.1067 | 1.6756 |

These are Gaussian-equivalent positive-response second moments, not direct
core-width measurements. The reported negative-lobe fractions are about 0.62
and 0.55, respectively. A high peak or a small grid cell cannot compensate for
an unevaluated or failed response gate. The original artifacts are retained;
the numerical grid remains a comparator rather than an accepted INR winner.

## First controlled experiment

Fit one static INR directly to the training detector pixels under fixed
stationary calibration, measured passbands, WCS, masks, inverse variances and
declared pixel response. Compare an Adam fit with the identical Adam fit
followed by damped Gauss–Newton updates solved by PCG in **neural parameter
space**. No deconvolved image becomes a training target. Calibration and
validation exposures stay outside the parameter updates.

The initial convergence screen uses the same 128-square target crop, F150W and
F277W, seed 101, and the existing PATT train/validation roles. It is a feasibility
screen, not a depth claim. Only after numerical checks and real detector
validation should it expand to injection batches, multiresolution INR models,
six bands and full cluster coverage. Keep equal-compute Adam controls when
interpreting any optimizer advantage.

Use an explicit measured classical resolution reference when scoring. Missing
response measurements yield no selected winner. Smooth-continuum and band-limited
dropout controls are distinct strata; a central-band cutoff is an engineering
control, not an IGM/redshift simulation. Never assign a redshift to an injection
from that cutoff. Zero-flux bands measure leakage rather than fractional flux
bias.

Negative extrema use the same signed matched responses, whitening and peak rule
as positive detections. They are a sign-symmetry diagnostic. Neural fields enforce
positive intensity, so their negative tail cannot by itself calibrate false alarms
or establish a Gaussian SNR advantage over Drizzle. Discovery ranking needs
independent detector evidence and noise controls that respect that constraint.

The recovered input inventory identifies all 135 registered detector products.
An independent headless audit compares their SCI/ERR/DQ planes with the original
CRFs and records WCS changes separately. The pilot's calibration report requested
35 exposures; it does not establish six-band calibration coverage. Expansion
requires explicit matching calibration rather than a nominal fallback.

## Reproducibility and continuation

Runtime source is a content-hashed working-tree archive, with the base Git
commit and per-file hashes recorded separately. This preserves the existing
uncommitted changes without claiming they are a pushed commit. Each headless
job verifies the archive, builds its Pixi environment under `/scratch`, and
persists job configuration, logs, detector reports, fields and products under
the isolated ARC root:

`/arc/projects/ots/torchsky-runs/smacs-inr-autonomous-20261010/`

Local control configurations and recovered evidence are under
`benchmarks_results/smacs-inr-20261010/` (ignored runtime artifacts). The local
`pcg_objective_audit.json` records the source-report hash and the stricter
same-input coadd comparison. Never silently replace failed jobs or overwrite
different artifacts at a previously used run identity.

The allowed base image is a mutable tag: Skaha rejected a digest reference.
The registry digest observed at launch is recorded, but does not prove the image
used by the running session. Source, bootstrap, installer, configuration and
declared frozen-input hashes are verified separately. An attempt ledger retains
startup failures and corrected run identities; only completed measurements count.

Selection remains on validation evidence. Freeze settings and thresholds
before a blind test. Report missing measurements, failed runs, sample size,
background covariance, operator uncertainty and full-field coverage explicitly.
