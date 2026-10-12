# Torchsky second-pass documentation inventory

Inventory only; no repository files were edited. Source is `/Users/fabbros/src/astroai/torchsky`; destination is `/Users/fabbros/src/astroai/torchsky-harness` on clean `main` at `4313545` (`Merge pull request #1 from astroai/codex/extract-scientific-evaluation`). The source worktree is heavily dirty (54 modified, 228 deleted, 9 untracked paths). Preserve its current documentation edits before further rewrites.

## Preserve before editing

Current source docs with working-tree modifications: `AGENTS.md`, `docs/README.md`, `docs/benchmarks.md`, `docs/catalogs.md`, `docs/contributing.md`, `docs/map.md`, `docs/release.md`, `docs/roadmap/active.md`, `docs/roadmap/next.md`, `docs/roadmap/status.md`, and `docs/wcs.md`. Untracked docs include `docs/plans/torchsky-harness.md`, `docs/projects/smacs_inr_autonomous_bakeoff.md`, and `docs/roadmap/ecosystem_parity_and_uv_ir_plan.md`. In particular, preserve complete copies of the changed roadmap and WCS/catalog/map docs before stripping evaluation detail. The report is based on the current working-tree contents, including these user changes.

## Existing destination copies (exact SHA-256 matches)

These source documents already have byte-identical copies in the harness; the destination path changed for several. No second copy is needed unless a later edit intentionally changes content.

| Torchsky source | Harness destination | SHA-256 |
|---|---|---|
| `docs/archive/benchmark-history.md` | `docs/archive/benchmark-history.md` | `a2d543e8abfb5ef54009db2fb5612b5c2300fa5de0c9a11f82934c63898de330` |
| `docs/compatibility.md` | `docs/compatibility.md` | `000bfbfe678db6700e5cc5fe37175a4af6e1f5ca4ee01735bed418d8a98a96c7` |
| `docs/projects/adass_updec_depth_week.md` | `docs/adass_updec_depth_week.md` | `488a98f219397577da7ed33f2d6394e4d7dbc9232f4fe43b74a9ff3290c04d37` |
| `docs/projects/inr_foam_problem_formulation.md` | `docs/inr_foam_problem_formulation.md` | `6f510434eb827fb6f66a76aaa882c5634a89b12f5f9238496413857fa053fbbd` |
| `docs/projects/smacs_inr_autonomous_bakeoff.md` | `docs/smacs_inr_autonomous_bakeoff.md` | `c2d5b233f986b35b70fbee5f8272fd7efd6f758ece344c0a0496895967487944` |
| `docs/projects/wcs-aware-smacs0723-pipeline.md` | `docs/wcs-aware-smacs0723-pipeline.md` | `06334f9e9a9bd76972e3183f9a5d853bac2f98b1e15333da5c75cd66a5954638` |
| `docs/plans/torchsky-harness.md` | `docs/migration-plan.md` | `e41ad20a379ebe9a26f6c824a0265d28986f7f53dd5a4fe518f474f1003e920e` |

The first, third through seventh are scientific/evaluation or migration material and can be removed from the core after references are updated. `docs/compatibility.md` is a mixed runtime/API guide and should remain in torchsky in a reduced form; keep the harness copy for evaluation coverage, but do not treat the source duplicate as wholly relocatable.

No exact destination copy was found for `docs/projects/wcs-small-n-investigation.md`, `docs/projects/hyperspectral-fusion.md`, `docs/archive/development-handoff.md`, or `docs/roadmap/reconstruction_execution_plan.md`.

## Relocation and split recommendations

### Move as evaluation records or campaign documents

- `docs/projects/adass_updec_depth_week.md` → already present as `torchsky-harness/docs/adass_updec_depth_week.md` (exact hash above). It is a 938-line SMACS/UPdec campaign ledger with run IDs, claims, baselines, acquisition and CANFAR commands.
- `docs/projects/smacs_inr_autonomous_bakeoff.md` → already present as `torchsky-harness/docs/smacs_inr_autonomous_bakeoff.md` (exact hash above). It is a result-bearing experiment record, not a package API guide.
- `docs/projects/wcs-aware-smacs0723-pipeline.md` → already present as `torchsky-harness/docs/wcs-aware-smacs0723-pipeline.md` (exact hash above). Its table and runbook point to moved `benchmarks/updc` scripts.
- `docs/projects/wcs-small-n-investigation.md` → harness historical WCS benchmark/evidence docs. It is a 122-line performance investigation and links to removed scorecards.
- `docs/archive/development-handoff.md` → harness campaign archive or provenance docs. Its data recreation and real-benchmark runbooks invoke deleted downloaders, `PYTHONPATH=src:benchmarks`, and removed benchmark modules.
- `docs/projects/hyperspectral-fusion.md` → either a campaign-specific tutorial in the harness or split: retain only generic, runnable torchsky API examples in core and move SMACS acquisition, benchmark, and result workflows. Its 569-line student workflow currently invokes removed scripts.
- `docs/roadmap/reconstruction_execution_plan.md` → harness SMACS campaign/workqueue history. This 432-line plan is dominated by calibration, injection, held-out comparisons, real-data runs, run gates, and artifact contracts. Keep only a short pointer in the core roadmap if useful.
- `.cursor/harness/proposals/multiband-depth-demo.md` → harness campaign design; it is a pre-registered scientific experiment.
- `.cursor/harness/proposals/map-tool-parity-2026.md` → harness comparator methodology/capability audit; it compares SWarp, Drizzle, and Montage and defines evaluation contracts.
- `docs/archive/benchmark-history.md` → already present byte-identically in harness. It is a pure historical benchmark snapshot.
- `docs/plans/torchsky-harness.md` → already present byte-identically as harness `docs/migration-plan.md`. It documents the extraction, including evaluation suites, run schemas, campaigns and evidence ownership; retain one canonical copy in harness.

`docs/projects/inr_foam_problem_formulation.md` is also an exact existing harness copy. It contains valuable mathematical/model formulation, but its active SMACS assumptions, evaluation hypotheses, split policy, metric targets and campaign backlog make the long research plan harness-owned. Keep the actual implemented field/operator concepts in `docs/field.md` and `docs/implicit_forward.md`; do not move those API guides.

### Keep core runtime/API architecture, but remove benchmark history and point to harness

- `README.md`: keep product scope and package performance policy; remove the invalid `[bench-all]` install extra and replace concrete performance claims with a short harness evidence link.
- `AGENTS.md`: retain package ownership, test tiers and developer workflow. Its core/harness distinction is already good. Correct stale statements such as old MEF fixture debt when the files were moved, and make all performance/evidence references explicitly point to the sibling harness rather than imply local `benchmarks/` directories.
- `docs/README.md`: keep the package guide index; replace the local archive benchmark link and experiment project index with the harness methodology/campaign links.
- `docs/benchmarks.md`: this is already a good short pointer to harness methodology. Keep it, after verifying its CLI example against current harness CLI docs.
- `docs/contributing.md`, `docs/release.md`: retain package lint/build/bounded-test/release steps; remove local benchmark directory descriptions, `bench-all` and old `bench-map-acceptance-smoke` / `bench-sphere` commands. Route scientific acceptance/release evidence to the harness.
- `docs/compatibility.md`: retain the actual public compatibility surface, optional adapter boundaries, and any behavior contract users need. Move detailed machine/run-specific ratios, hashes, claims, scorecard/status snapshots, and exhaustive upstream suite inventory to harness. Link capability claims to evidence there.
- `docs/map.md`, `docs/catalogs.md`, `docs/wcs.md`, `docs/sphere.md`, `docs/encoders.md`, and `docs/field.md`: retain public API and implementation semantics. Move throughput ratios, CANFAR results, acceptance status, and scientific comparison narratives to harness. Replace lists of in-repo benchmark modules with links to harness suites. Keep a bounded package regression pointer where it describes an actual retained test.
- `docs/roadmap/active.md`, `docs/roadmap/next.md`, and `docs/roadmap/status.md`: keep only active package implementation/architecture work and the concise ownership/status summary. Move run-by-run scientific progress, session IDs, scorecard links, metric results, acceptance narratives, data protocols and scientific workqueues to harness. Because these files are modified, first copy the complete current versions to harness or another explicit preserved snapshot before rewriting core summaries.
- `docs/roadmap/history.md`: preserve shipped package/API history, but replace removed benchmark commands and detailed performance evidence with harness evidence links. This document is an implementation history with embedded evaluation evidence, not a good whole-file move.
- `docs/roadmap/ecosystem_parity_and_uv_ir_plan.md`: preserve a full copy because it is a new untracked plan. Split its comparator-equivalence matrix and all `[x]` parity/performance assertions into harness evidence/contracts; keep only realistic package implementation priorities in core. The file currently asserts broad parity with multiple external stacks and includes future `psf`, detection and segmentation surfaces; those claims need evidence/implementation qualification before becoming a core roadmap contract. Do not create future placeholder package directories.
- `.cursor/harness/proposals`: keep implementation/API proposals and harness memory in core. `angular-motion-covariance.md`, `inr-foam-next.md`, `map-cli-next.md`, `map-coadd-next.md`, `phase-space-consumers.md`, `phase-space-propagation-review.md`, `power-foam-full.md`, and both phase-space covariance proposals are principally implementation design or code-agent working context. `map-coadd-next.md` may replace its benchmark plan with a harness link, but its seam/API/test design belongs in torchsky.
- `.cursor/harness/trajectories/` and `.cursor/harness/failures/`: retain as coding-agent memory and engineering provenance per scope. Do not bulk-move scientific-looking names without separating the memory function from published reports. The new `20261012-scientific-harness-extraction.md` trajectory is extraction memory, not a user-facing harness report.

`docs/projects/fossil*`, `source_unmixing_problem_formulation.md`, the USPM/Cosmodist roadmap, and other model/mission formulation docs are research and product planning rather than an existing scientific harness suite. Do not bulk-move them solely because they contain science or mention future tests. Reassess only their result tables, benchmark protocols, and stale runnable commands individually.

## References that become broken or misleading after the extraction

Update these before removing or moving the referenced paths:

- `README.md:94-96` — removed `bench-all` extra and install command.
- `docs/README.md:52-53,92` — current description promises local methodology, SMACS runs, snapshots and a local benchmark-history file.
- `docs/contributing.md:40,45,58,111-114,133-145` — describes local `benchmarks/`, `data/` real-data store, and removed comparator tasks.
- `docs/release.md:29-54` — old parity/acceptance/bench commands (`bench-map-acceptance-smoke`, `bench-all`) and CANFAR procedure.
- `docs/map.md:278-286`; `docs/catalogs.md:94-95`; `docs/wcs.md:214-219`; `docs/encoders.md:45`; `docs/compatibility.md:60,67,73-75` — old in-repo benchmark module paths / upstream evaluation inventory.
- `docs/sphere.md:521` — `scripts/run_canfar_cuhpx_sht_spike.sh` is an evaluation runner now in the harness. `docs/sphere.md:516` refers to the removed `scripts/fetch_healpix_pixel_weights.sh`; decide whether this supported user data-provision helper belongs in the harness or should be rehomed as a torchsky user-facing utility, then link accordingly.
- `docs/examples.md:63-84`, `docs/field.md:34-36,616`, and `docs/projects/hyperspectral-fusion.md:179-240` — removed SMACS build/download/quicklook scripts. Either move the documented workflow to the harness or provide still-supported package examples.
- `docs/projects/wcs-aware-smacs0723-pipeline.md:35-41` — moved `benchmarks/updc` paths (destination copy has same stale in-source paths and should be checked in the harness too).
- `docs/projects/wcs-small-n-investigation.md:20,38,42,121` — removed benchmark commands, scorecard paths and a relative Markdown link to `../../benchmarks/scorecards/README.md`.
- `docs/archive/development-handoff.md:25-87` — removed acquisition scripts and benchmark paths; its archive status does not make the commands runnable.
- `docs/roadmap/active.md` (many refs, e.g. lines 40-104, 188, 558, 1511, 1900-1917, 2004, 2367, 2511, 2580, 2883, 2964, 3260), `docs/roadmap/next.md:123-127`, `docs/roadmap/status.md:68-197,354`, `docs/roadmap/history.md:20-26,102,156-231`, and `docs/roadmap/reconstruction_execution_plan.md:123,195,313` — removed benchmark, scorecard, script and CANFAR paths. Replace with harness evidence links or move the specific evidence/workqueue sections.
- `docs/projects/adass_updec_depth_week.md` has old commands in multiple sections (`:113,314,340,357,713,895,905`); its destination copy is identical and should be audited there as well.
- `AGENTS.md:110` describes leftover map/SMACS MEF fixture writers still needing migration; verify against the completed extraction and remove if stale. `AGENTS.md:297` names a sibling-relative `torchsky-harness/benchmarks/` path; a direct link or checkout-aware wording is clearer.

Inline code paths are included here as well as clickable Markdown links; both can leave users with unusable commands. A repository-wide reference pass is still needed after the final move set is decided.

## Classification boundary

A compatibility/API guide states what torchsky implements and how callers use it; it may summarize a capability but should not own run-specific evidence. A harness document defines comparison contracts, fixtures, acceptance matrices, datasets/splits, performance methods, run history, claim eligibility or campaign coordination. Keep production algorithm design and public API in torchsky, even when those APIs later appear as a harness target.
