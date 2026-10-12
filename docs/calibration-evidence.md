# NIRCam calibration evidence

The NIRCam evidence workflow prepares reproducible point-spread-function (PSF)
and background calibration samples for the runtime operator models in
[`torchsky`](https://github.com/astroai/torchsky). This guide owns exposure
acquisition assumptions, sample discovery, frozen split policy, and the
manifest-building commands. Torchsky owns the differentiable calibration terms
and the validated runtime loader described in its
[continuous-field guide](https://github.com/astroai/torchsky/blob/main/docs/field.md).

## Freeze aligned exposure inputs

Use the complete aligned exposure set selected for the experiment. For the
SMACS0723 UPDec run, the intended products are the 135 aligned
`*_crf_rshift_tweakreg.fits` files. The 27 plain CRFs cover only the long
channel; the 135 CAL products are the reproducibility fallback. Record the
source paths and checksums with the run so the selected inputs can be recovered.

Build a manifest from an explicit evidence table:

```console
torchsky-harness calibration-manifest \
  '/path/to/aligned/*.fits' \
  --evidence calibration-evidence.csv \
  --output calibration-manifest.json
```

The CSV columns are `exposure,detector_x,detector_y,role` and optional
`source_id`; `role` is `psf` or `background`. Coordinates are zero-indexed
NIRCam detector coordinates, not aligned-mosaic pixels. The command reads
primary and SCI headers through Torchfits, maps the FITS one-indexed
`SUBSTRT*`/`SUBSIZE*` footprint into the detector frame, freezes the
`PATT_NUM` train/validation/test partition, and records canonical MJD UTC
midpoint provenance. It fails on inconsistent timing aliases, footprints, or
evidence coordinates.

## Extract evidence samples

Without an external catalog, the extractor fits a heteroscedastic Gaussian-PSF
amplitude while analytically marginalizing a local constant sky, robustly
whitens the response, retains isolated compact peaks, refines their subpixel
centroids, and lays out source-excluded background controls across each
detector. Thresholds and selection diagnostics are persisted so the same
selection rule applies to every frozen split.

```console
torchsky-harness calibration-extract \
  '/path/to/aligned/*.fits' \
  --output-dir calibration-evidence
```

An explicitly reviewed astrometric catalog is an optional higher-assurance
override. Its columns are
`source_id,role,ra_deg,dec_deg,reviewed`. `role` is `psf` for an isolated,
unsaturated calibration star or `background` for a source-free location.
Optional columns are
`pmra_mas_yr,pmdec_mas_yr,ref_epoch_jyear,sample_weight`; proper motions use
the Gaia `pm_ra_cosdec` convention and default to zero at J2016.0. Every row
must explicitly set `reviewed=true`.

```console
torchsky-harness calibration-extract \
  '/path/to/aligned/*.fits' \
  --catalog reviewed-calibrators.csv \
  --output-dir calibration-evidence-reviewed
```

The command reads `SCI`, `ERR`, and `DQ` through Torchfits. Catalog positions
are propagated to exposure midpoint and transformed with Torchsky WCS. PSF
stamps are bilinearly registered at fractional detector phase, fitted with a
weighted sigma-clipped local plane, normalized to unit flux, and stored with
per-pixel inverse-variance weights. Blank-sky targets are the centre value of
the same robust local plane, weighted by detrended variance. Only `DQ == 0` is
accepted; incomplete and low-S/N stamps are rejected with their reasons
persisted. The manifest links target indices, product SHA-256 values,
instrument metadata, detector coverage, and the fixed train/validation/test
split. Bilinear stamps are the initial detector-sampled ePSF contract; promote
them to oversampled spatial ePSFs only if held-out encircled-energy residuals
show detector-phase structure.

For each role, band, detector, and split, the manifest records a certified
interval for the two-dimensional rectangular-domain fill distance. Its lower
endpoint is the maximum sampled grid-centre distance; its upper endpoint adds
the grid-cell half-diagonal using the 1-Lipschitz property of distance to the
nearest sample. Refinement continues to the requested error or fails closed.
Twice the upper endpoint is a conservative coverage guard, not a
scattered-sampling Nyquist theorem or a physical PSF/background correlation
length. The manifest includes a deterministic SHA-256 over its canonical
payload.

For exposure-specific evaluation, preserve each exposure's exact identity and
band with the sample rows. Do not mix exposure IDs when constructing the
train/validation/test sample. Torchsky's loader requires both `exposure_id` and
`band` together and verifies target product checksums by default.
