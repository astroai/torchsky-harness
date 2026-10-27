# Paired release-performance campaign

The runner executes the three frozen release tasks for all six declared claims. For pair 0 it runs baseline then candidate; pair 1 reverses that order, and the pattern repeats for at least five complete pairs. Baseline and candidate use separate harness checkouts and separate Pixi environments. Any failed checkout, build, install, subject identity check, task, row selection, status check, timing check, artifact build, or readiness validation stops the campaign.

The output root must not exist. It contains raw task reports, one-row extracted reports, strict SHA-256 identities, artifacts, the generated claims configuration, readiness output, measured environment fingerprints, subject identities, and the execution-order manifest. Every internal path is relative to this root, so the complete tree can be copied to a new durable root and revalidated with `performance --artifact-root NEW_ROOT`. Readiness reopens each raw report, recomputes its digest, reselects exactly one row, and verifies the selected timing. It does not supply or generate a real baseline unless the operator explicitly provides the baseline SHA or wheel.

Python, Torch, torchsky version/revision, OS, CPU, hostname, and device are measured independently inside each baseline and candidate task environment. Each claim adds its validated frozen task-command hash, effective arguments, and declared dtype (`float64` for WCS/HEALPix and `float32` for Drizzle). Operator-authored fingerprints are not accepted.

## Local dry run

```sh
pixi run -e test -- python campaigns/release_performance/run.py \
  --harness-sha <40_HEX_HARNESS_SHA> \
  --baseline-sha <40_HEX_BASELINE_TORCHSKY_SHA> \
  --candidate-sha <40_HEX_CANDIDATE_TORCHSKY_SHA> \
  --output-root /tmp/torchsky-release-performance-<RUN_ID> \
  --dry-run
```

Replace either `--baseline-sha` or `--candidate-sha` with the corresponding `--baseline-wheel` or `--candidate-wheel` path to evaluate an immutable wheel. Wheel SHA-256 values are recorded automatically.

## Exact CANFAR submission template

The launcher is a dry run unless `--submit` is present. CANFAR launches the public `base:latest` tag, and the runner verifies that tag's live Harbor digest against the immutable campaign digest before executing; there is no operator-supplied digest field.

```sh
bash campaigns/release_performance/launch_canfar.sh \
  --output-root /arc/projects/<PROJECT>/torchsky-release-performance/<RUN_ID> \
  --harness-sha <40_HEX_HARNESS_SHA> \
  --baseline-sha <40_HEX_BASELINE_TORCHSKY_SHA> \
  --candidate-sha <40_HEX_CANDIDATE_TORCHSKY_SHA>
```

After reviewing the printed command, repeat it with `--submit`. The requested job is one headless CPU session using the digest-verified AstroAI base image, 8 CPUs, 32 GB RAM, no GPU, one benchmark process, and no automatic partial-pair retry. The exit trap copies and byte-compares every scratch artifact and log. Missing required outputs or any persistence mismatch exits 70 and writes a failure record where possible. A failed attempt must use a new nonexisting destination and restart the complete campaign.
