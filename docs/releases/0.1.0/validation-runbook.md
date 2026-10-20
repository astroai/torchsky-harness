# Post-rework subject validation runbook

**Repository boundary.** This runbook lives in `torchsky-harness` because it
describes benchmark, gate, evidence, and campaign work.

- `torchsky` owns the core Python/native implementation, the public API, package
  tests, docs, and packaging metadata. It does **not** contain benchmark
  runners, gate tasks, evidence bundles, campaign code, or comparator harnesses.
- `torchsky-harness` owns `benchmarks/`, evaluation `scripts/`, gate tasks,
  subject installation, evidence claims/bundles, campaigns, scorecards, and this
  runbook. It evaluates an explicitly installed subject and never embeds a
  torchsky source checkout.

Status: prepared while the torchsky native/WCS/sphere/map/field rework is
still in the working tree. Run this **only after the rework is committed and
the working tree is clean**.

The subject will have widely changed native code, WCS, sphere and map/field
paths, so this is a full evidence refresh, not a partial gate rerun.

## 0. Freeze the subject

- [ ] `git -C torchsky status --porcelain` is clean; note `git rev-parse HEAD`.
- [ ] `git -C torchsky-harness status --porcelain` is clean.
- [ ] Confirm the subject version is still 0.4.0 and harness 0.1.0.
- [ ] Do not renumber gates, loosen thresholds, or reuse run IDs.

## 1. Build and record the wheel

```bash
cd /Users/fabbros/src/astroai/torchsky
rm -rf /tmp/torchsky-rework /tmp/wheels-rework && mkdir -p /tmp/wheels-rework
git archive HEAD | tar -x -C /tmp/torchsky-rework
CMAKE_PREFIX_PATH=$PWD/.pixi/envs/default \
  ./.pixi/envs/default/bin/python -m pip wheel /tmp/torchsky-rework \
  --no-build-isolation --no-deps -w /tmp/wheels-rework
sha256sum /tmp/wheels-rework/*.whl
```

The wheel is the subject artifact. Record its sha256; the harness also records
`package_content_sha256`.

## 2. Install the subject in every gate environment

```bash
cd /Users/fabbros/src/astroai/torchsky-harness
WHEEL=$(ls /tmp/wheels-rework/*.whl)
for env in test bench-all sphere-bench map-compat; do
  pixi run -e "$env" -- python -m pip install --force-reinstall --no-deps "$WHEEL"
  pixi run -e "$env" -- python -c "import torchsky; print(torchsky.__version__, torchsky.__file__)"
done
```

If the harness editable loader is stale after package/file moves:

```bash
pixi run -e test -- python -m pip install -e . --no-deps --no-build-isolation
```

For `map-compat`, also install the comparator stack if the env lacks it
(`scripts/bootstrap_map_compat_tools.py --tool swarp|montage`).

## 3. Subject-side sanity before the gates

```bash
cd /Users/fabbros/src/astroai/torchsky
pixi run -e test -- pytest -n auto -m "not performance" -q
pixi run examples
pixi run typecheck
pixi run docs-contract
```

Expected green: full suite, 31/31 examples, pyright 0, docs contract.

## 4. Run the eighteen declared gates

Run each task and keep the JSON outputs named in the release procedure. Do not
overwrite the previous release artifacts; use a fresh output directory if a
benchmark supports it.

Synthetic/truth (`test` env):

- [ ] `pixi run bench-map-cube-gate`
- [ ] `pixi run bench-map-ifu-gate`
- [ ] `pixi run bench-map-ifu-spectral-axis-gate`
- [ ] `pixi run bench-map-healpix-multiband-gate`
- [ ] `pixi run bench-native-response-gate`
- [ ] `pixi run bench-native-likelihood-gate`
- [ ] `pixi run bench-native-nuisance-gate`

Contract (`test` env):

- [ ] `pixi run bench-native-contracts-gate`
- [ ] `pixi run bench-domain-fuzz-gate`

Reference comparison (`bench-all` / `sphere-bench`):

- [ ] `pixi run bench-wcs-release-gate`
- [ ] `pixi run bench-healpix-release-gate`
- [ ] `pixi run bench-healpix-advanced-release-gate`
- [ ] `pixi run bench-sphere-matrix-release-gate`
- [ ] `pixi run bench-spin-matrix-release-gate`
- [ ] `pixi run bench-catalog-release-gate`

Real data (`bench-all`):

- [ ] `pixi run bench-wcs-real-headers-gate`
- [ ] `pixi run bench-sphere-real-map-gate`
- [ ] `pixi run bench-map-real-cutout-gate`

Expected: 16/16 `established`; any failure is a finding, not a threshold to
relax. Record the exact comparison rows and limits.

## 5. Native-contract probes for the reworked C++

These are the audit counterexamples; run them against the installed subject
before trusting the scientific gates. Use subprocess isolation and timeouts.

- [ ] empty HEALPix ring synthesis: zero rings/pixels -> return empty or raise,
      never signal.
- [ ] undersized ALM buffers -> raise, not NaN.
- [ ] concurrent `map2alm_spin`/`alm2map_spin` (2+ threads, timeout) -> no hang.
- [ ] fused WCS: wrong dtype/device/contiguity/shape/undersized output -> raise.
- [ ] TAN at RA 91/180/270 deg from CRVAL (0,0) -> invalid where WCSLIB is
      invalid.
- [ ] MOL inverse/round trip at dec 89.9/89.99/89.999.
- [ ] HPX H=6,K=4 (and other permitted pairs) vs WCSLIB.
- [ ] TPV zero/constant/singular metadata -> fail closed.
- [ ] malformed TPV metadata (rank, `(N,3)`, exponents, lengths) -> raise.
- [ ] HEALPix NaN/Inf/out-of-range latitude -> raise or documented sentinel.
- [ ] direct scalar SHT at `lmax=mmax=256` high-order coefficients -> finite.
- [ ] `requires_grad=True` on TPV inverse/interpolation/native geometry.
- [ ] tiny arcs (half-length ~2e-8 rad) and coincident arcs.
- [ ] max nside/order: checked overflow, no OOM/unbounded allocation.

## 6. GPU / CUDA validation

If CANFAR credentials and network are available:

```bash
cd /Users/fabbros/src/astroai/torchsky-harness
# dry-run the launch script first; flags are revision-pinned
bash scripts/launch_canfar_wcs_sphere.sh gpu-validate
bash scripts/launch_canfar_wcs_sphere.sh cuda
```

For each finished run: fetch the session log and benchmark outputs, record
session id, subject sha, harness sha, device, and rc. Expected: 0 failures on
the device-named tests and on the WCS CUDA matrix; a sphere comparator deficit
is reported, not claimed as a win.

## 7. Bundle and docs

```bash
cd /Users/fabbros/src/astroai/torchsky-harness
pixi run evidence-bundle
```

- [ ] `release_ready: true`, `subject_consistent: true`.
- [ ] `verdict_counts` has 16 established, no missing/incomplete claims.
- [ ] Update `docs/releases/0.1.0/gate-matrix.md`,
      `docs/releases/0.1.0/release-procedure.md`, and
      `docs/benchmarks.md` with the new measured numbers and session ids.
- [ ] Update torchsky `README.md`/`docs/benchmarks.md` if numbers changed.
- [ ] Re-run `pixi run preflight`, docs checks, and `pixi run test-bounded`.

## 8. Evidence refresh preconditions

- The subject wheel identity in the bundle must be the wheel installed in every
  gate env.
- All gate runs must be at the same harness revision and subject.
- No gate may pass only because a comparator was missing or skipped; report
  coverage gaps instead.
- The bundle is git-ignored; attach it to the release notes and keep raw run
  directories in durable storage.

## Known open items from the rework audit

Resolved during earlier validation (recheck after rework): stale harness import
of `torchsky.field.frequency_adaptive`; removed `max_entries` option; removed
`torchsky[catalog-qa]` hints. Still open elsewhere: publishing-workflow overlap
(`build_wheels.yml` vs `publish.yml`), CLI `--inr-*` flag rename, and the
native-contract gate family (design in `torchsky/notes/native-contract-gate-plan.md`).
