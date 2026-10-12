# Extraction wiring audit

Audit date: 2026-10-12. This is a read-only audit of the staged harness CI,
Pixi entries, and runner wiring. No CANFAR commands or remote jobs were run.

## Findings

### CANFAR depth uses the harness and subject as separate checkouts

`launch_depth()` clones `astroai/torchsky-harness`, checks out the harness
commit, then invokes `run_canfar_adass_depth.sh` and passes both
`TORCHSKY_HARNESS_COMMIT` and `TORCHSKY_COMMIT` ([launcher, lines 248–266](../scripts/launch_canfar_wcs_sphere.sh)).
The depth runner requires the harness SHA, clones the harness repository, and
loads `scripts/subject_install.sh` ([runner, lines 4–58](../scripts/run_canfar_adass_depth.sh)).
Its `ensure_native_build` installs the selected subject into the chosen Pixi
environment; the later editable install at line 399 is the harness checkout,
not torchsky ([runner, lines 79–83 and 398–400](../scripts/run_canfar_adass_depth.sh)).
This path no longer builds torchsky as the harness.

### CANFAR subject repository defaults disagree with the workflows

The WCS and depth launchers pass the subject commit and checkout path but do
not pass `TORCHSKY_SUBJECT_REPO` ([launcher, lines 93–95 and 264–266](../scripts/launch_canfar_wcs_sphere.sh)).
The shared subject installer therefore defaults to `sfabbro/torchsky`
([subject installer, lines 11–21](../scripts/subject_install.sh)). Both manual
scientific workflows explicitly check out the private subject from
`astroai/torchsky` ([WCS workflow, lines 14–20](../.github/workflows/wcs_sphere_weekly.yml);
[map workflow, lines 14–20 and 46–52](../.github/workflows/map_compat_weekly.yml)).
This makes subject identity depend on which repository happens to expose the
requested SHA and whether the runner's GitHub credentials can read it.

Exact fix: make the repository selection explicit and consistent. Either pass
`--env TORCHSKY_SUBJECT_REPO=astroai/torchsky` in both CANFAR launcher argument
arrays, or change the subject installer default to `astroai/torchsky` and keep
the override for forks. If the intended CANFAR subject is the fork instead,
make the workflows use that same repository. Preserve `TORCHSKY_COMMIT` as the
exact subject SHA.

### The standalone cuHPX runner still uses torchsky as its harness

`scripts/run_canfar_cuhpx_sht_spike.sh` accepts only `TORCHSKY_COMMIT`, clones
`sfabbro/torchsky`, then uses that checkout's `pixi.toml`, editable package,
benchmark, and comparator script ([runner, lines 5–8 and 30–65](../scripts/run_canfar_cuhpx_sht_spike.sh)).
That is stale after extraction: it conflates the subject SHA with the harness
SHA and expects benchmark infrastructure to live in the core checkout.

Exact fix: require separate `TORCHSKY_HARNESS_COMMIT` and `TORCHSKY_COMMIT`;
clone and check out `astroai/torchsky-harness` at the former; source
`scripts/subject_install.sh` and install the explicit subject at the latter;
then run the cuHPX benchmark and comparator-version module from the harness
checkout. Use its `cuda-bench` environment only if that feature continues to
contain the required CUDA dependencies. The main CANFAR runner already has an
integrated `cuhpx` mode, so the standalone script is not referenced by Pixi or
CI; `docs/benchmarks.md` still documents it as an entry point and should be
updated when the runner is repaired or retired.

The standalone script also unconditionally removes `/scratch/${run_id}-src`
before cloning. This audit did not execute it; retain that cleanup only if the
path is guaranteed to be the runner's dedicated harness checkout.

### CI and scientific workflow triggers

The infrastructure workflow is correctly independent of a scientific subject:
it installs the harness editable, runs `preflight`, executes the runner and
local-benchmark mock tests, then builds and uploads the pure-Python wheel
([CI workflow, lines 13–23](../.github/workflows/ci.yml)).

Both scientific workflows have only `workflow_dispatch`; each requires an
exact `subject_ref` and uses `SUBJECT_READ_TOKEN` to check out
`astroai/torchsky`. They currently have no schedule trigger, so they cannot be
activated on a timer before the private-repository credential is configured.
Their job display names still say “weekly” or “scheduled”, which is stale
labeling but does not change trigger behavior.

One Pixi task has an implicit environment: `bench-pytest` runs
`pytest benchmarks/ --benchmark-only` without selecting an environment
([Pixi task, lines 123–124](../pixi.toml)). The `pytest-benchmark` plugin is
only present in the `bench` feature ([Pixi, lines 185–192](../pixi.toml));
running the task in the default environment will not provide the plugin.
Exact fix: make the task invoke `pixi run -e bench -- pytest benchmarks/
--benchmark-only`, or document that callers must run `pixi run -e bench
bench-pytest`.

## Verification boundary

This audit inspected the staged files and references only. It did not launch
remote jobs, install additional environments, or change runner, CI, or Pixi
files.

## Resolution

The root integration fixed the standalone cuHPX dual-repository runner, aligned the default subject repository with astroai/torchsky, set it explicitly in CANFAR payloads, and selected the bench environment for bench-pytest. Manual workflow labels were updated. No remote jobs were launched.
