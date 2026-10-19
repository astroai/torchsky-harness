# High-level system review — torchsky 0.4.0 / harness 0.1.0

Status: working review for the release branch. It records the target ownership
model and the refactors that have landed; it is not a claim that the release is
complete.

## Two repositories, one evidence chain

| Repository | Owns | Must not own |
|---|---|---|
| `torchfits` | FITS/table parsing, HDUs, headers, checksums, compression, file caches, datasets | sky geometry, mapmaking, scientific metrics |
| `torchsky` | WCS, sphere, catalogs, mapmaking, scenes, field/native modelling, bounded package tests | FITS implementation, comparison campaigns, scientific scorecards |
| `torchsky-harness` | scientific gates, comparator adapters, scorecards, campaigns, provenance, release evidence | production algorithms or a second FITS layer |

The release is paired: `torchsky 0.4.0` is the product and
`torchsky-harness 0.1.0` is its evidence layer. A subject is always installed
explicitly; a checkout is never injected as the subject.

## Target public layers

| Layer | Canonical surface | Tier |
|---|---|---|
| world | `wcs`, `sphere`, `catalogs`, `coords` | core |
| map | `map` (image-plane/cube/HEALPix coadds, operators, products) | core |
| 1D physics | `spectral`, `spectrophotometry`, `redshift`, `log_lambda`, `extinction`, `cosmology` | core |
| scene | `scene` representations, target specs, operator adapters | experimental |
| field | `field` model/fit/render/sample verbs | experimental |
| native | `map.likelihood`, `map.query`, `map.evidence`, `map.field.native_*`, native operators | experimental/beta |
| ml/viz | `transforms`, `encoders`, `viz` | experimental/optional |
| FITS/table I/O | `torchfits` only | external boundary |

Rules:

1. One public name per concept. Implementation helpers stay in explicit
   submodules (`map.image_plane`, `map.product`, `map.field`, ...).
2. Root `torchsky` exposes science namespaces and `WCS` only. No I/O, HDU,
   table, cache, dataset, or dataloader aliases.
3. `scene` does not duplicate map verbs or field fit/render verbs.
4. The field model zoo lives under `torchsky.map.field`; `field` exposes the
   composition verbs and recommended presets.
5. Native response/likelihood stays experimental until its synthetic
   instrument–sky gate and M1 evidence exist.
6. Compatibility namespaces (`sphere.compat`, `sphere.healpy`) remain explicit
   and are not counted as core.

## Refactors landed in this pass

1. **Checkpoint:** preserved the dirty migration and paused native
   field-response work in a release-branch commit.
2. **I/O boundary reset:** removed root I/O aliases and the
   `io`, `table`, `hdu`, `cache`, `datasets`, `dataloader`, `header_parser`, and
   `core` modules; all internal/test/example callers use `torchfits` directly.
3. **Scene contract:** reduced the experimental facade to representations,
   target specs, and operator adapters; removed duplicated map/field verbs and
   result aliases.
4. **Map product names:** renamed `MapProductSpec` → `ProductSpec`,
   `MapBandpassSpec` → `BandpassSpec`, `MapProductResult` → `ProductResult`, and
   `MapUncertaintyProduct` → `UncertaintyProduct`; renamed
   `stack_map_product` → `stack_product` and removed duplicate result aliases.
   The public `map.__all__` is now the workflow set only.
5. **Native work included:** docs, CI shards, and the release scope now include
   the native response/likelihood modules; it is explicitly experimental/beta.

## Remaining consolidation

1. Remove the remaining root-level map implementation aliases
   (`coadd_image_plane*`, `evaluate_image_plane_aperture_uncertainty`) by
   migrating callers to `coadd*`/`aperture_uncertainty` or to the owning
   implementation submodule.
2. Curate the 99-name `field` facade to composition verbs, recommended presets,
   and documented diagnostics; keep the model zoo under `map.field`.
3. Collapse the three sphere SHT alias surfaces and remove duplicate region
   names; keep `compat`/`healpy` as the only compatibility surface.
4. Split `map/image_plane.py`, `wcs/core.py`, `sphere/spectral.py`, and
   `map/field/core.py` at the boundaries recorded in the audit.
5. Done: cube, matched-sampling IFU, HEALPix multi-band, and three native
   gates exist with committed baselines and completeness checks.
6. Bump versions and add publish workflows. Done: `execution.json` carries a
   real, declared gate verdict (see [evidence-verdicts.md](../../evidence-verdicts.md)).

## Verification state

- torchsky full suite was green after the I/O reset and scene/API changes;
  the map cleanup is being verified by a fresh full-suite run.
- harness preflight and targeted boundary/gate tests pass.
- six synthetic release gates now record `scientific_validity: established` with
  `comparison_eligibility: baseline_compared`; they are bounded claims, not a
  general validity claim, and real-data rows remain open.
