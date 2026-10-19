# Release procedure — torchsky 0.4.0 + torchsky-harness 0.1.0

The first public release is a pair: `torchsky` 0.4.0 (the scientific subject)
and `torchsky-harness` 0.1.0 (validation and evidence). `torchsky` cannot take
the 0.1.0 number because 0.1.x–0.3.1 are already on PyPI, so 0.4.0 is its public
launch and 0.1.0 is the harness's first release. Both must be cut from commits
whose evidence bundle is `release_ready`.

## Prerequisites

1. `pixi run preflight` passes in both repositories. Harness:
   `pixi run test-bounded` (the `test` task also runs the opt-in
   `performance`/`benchmark`/`slow`/`integration` cases, which are not part of
   the release suite). torchsky: `pixi run test` plus the map shards
   (`pixi run -e test -- pytest -n auto -q 'tests/test_map_*.py'`).
2. Every declared gate records `scientific_validity: established`; the
   synthetic/truth gates additionally record
   `comparison_eligibility: baseline_compared`, and
   `pixi run evidence-bundle --fail-on-missing` reports `release_ready: true`
   with a single subject identity.
3. The subject identity in the bundle is the artifact being published: rebuild
   the gates against the release commit and compare
   `subject.package_content_sha256` with the built wheel after install
   (`torchsky-harness subject`).
4. Docs state the delivered claims and their limitations
   (`docs/releases/0.1.0/gate-matrix.md`); no gate claims real-data accuracy,
   SOTA, or universal device support.

## Gate bundle

Synthetic/truth gates (`test` environment):

| Gate | Task |
|---|---|
| multi-band cube assembly | `pixi run bench-map-cube-gate` |
| matched-sampling IFU cube | `pixi run bench-map-ifu-gate` |
| IFU spectral axis (irregular/localization/envelope) | `pixi run bench-map-ifu-spectral-axis-gate` |
| HEALPix multi-band coaddition | `pixi run bench-map-healpix-multiband-gate` |
| native Gaussian response | `pixi run bench-native-response-gate` |
| native Gaussian likelihood + stale evidence | `pixi run bench-native-likelihood-gate` |
| native nuisance precision | `pixi run bench-native-nuisance-gate` |

Reference-comparison gates (`bench-all`/`sphere-bench` environments; install the
release wheel and the harness there, then run; `map-compat` is needed only for
the supporting `bench-map-compat-gate` comparison):

| Gate | Task |
|---|---|
| 30-projection WCS parity vs astropy | `pixi run bench-wcs-release-gate` |
| HEALPix core parity vs healpy | `pixi run bench-healpix-release-gate` |
| HEALPix neighbors/interpolation parity vs healpy | `pixi run bench-healpix-advanced-release-gate` |
| sphere geometry vs healpy | `pixi run bench-sphere-matrix-release-gate` |
| spin-2 SHT vs healpy | `pixi run bench-spin-matrix-release-gate` |
| catalog crossmatch vs astropy | `pixi run bench-catalog-release-gate` |

Then `pixi run evidence-bundle` writes
`benchmark_results/evidence_bundle.{json,md}` (git-ignored by design). Attach
the bundle to the tag/release notes. The raw run directories stay in durable
storage; the bundle hashes them.

## Version and tag

| Repository | Version | Tag |
|---|---|---|
| torchsky | `0.4.0` | `v0.4.0` |
| torchsky-harness | `0.1.0` | `v0.1.0` |

Versions live in `pyproject.toml`, `pixi.toml` (torchsky), and the package
`__version__`. The publish workflows fail the build when the tag does not match
`pyproject.toml`.

## Publish order

1. **torchsky 0.4.0** — `.github/workflows/publish.yml` builds sdist+wheel,
   checks the compiled extension is present, installs the wheel in a clean
   environment with CPU torch, and runs a WCS + `coadd` smoke before publishing
   through PyPI trusted publishing (`environment: pypi`).
2. **torchsky-harness 0.1.0** — builds the pure-Python wheel, checks that the
   evaluators, baselines, and fixtures are packaged, installs it in a clean
   environment with the published `torchsky==0.4.0`, then checks
   `torchsky-harness list` and `torchsky-harness subject` before publishing.

PyPI trusted publishing requires, for each repository, an `environment: pypi`
configured on PyPI as a trusted publisher for the workflow
`.github/workflows/publish.yml`. No API tokens are stored in the repository.

## Post-release verification

```sh
python -m venv /tmp/release && . /tmp/release/bin/activate
pip install "torch>=2.13" --index-url https://download.pytorch.org/whl/cpu
pip install "torchsky==0.4.0" "torchsky-harness==0.1.0"
torchsky-harness subject
torchsky-harness list --domain map
```

Re-run the release gates against the published wheel using the runner's
`--subject-wheel` option and keep that bundle as the published-artifact
evidence. If a gate fails on the published artifact, yank the affected release
on PyPI rather than silently re-tagging, fix forward, and cut a patch release.

## Claim boundaries

A passing bundle certifies the declared synthetic claims only. Open limitations
recorded in the gate matrix (physical wavelength-space spectral interpolation for
irregular axes, flux-conserving spectral redistribution, HEALPix harmonic
resampling, comparator throughput, real-data calibration, and the experimental
native/scene/field surfaces) remain open and must be repeated in release notes.
