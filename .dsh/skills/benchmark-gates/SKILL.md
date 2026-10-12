---
name: benchmark-gates
description: Run torchsky-harness scientific evaluations, compare against explicit evidence, and record subject and harness identities
metadata:
  project: torchsky-harness
---

# Scientific benchmark gates

This skill applies in the `torchsky-harness` checkout. Read its
[`AGENTS.md`](../../../AGENTS.md) and
[`docs/benchmarks.md`](../../../docs/benchmarks.md) before running an
evaluation. FITS image/table I/O performance belongs to `torchfits`.

## Select an evaluation and subject

Use the installed CLI to inspect registered work and the currently imported
subject:

```bash
pixi run torchsky-harness list --domain map
pixi run torchsky-harness subject
pixi run torchsky-harness run bench_map_compat -- --size 32 --n-obs 2 --runs 1 \
  --cases synthetic_shifted_wcs --torchsky-kernels bilinear
```

The harness evaluates an explicitly installed torchsky subject. Do not inject
a torchsky checkout through `PYTHONPATH`, `sys.path`, or direct source-file
imports. Record both repository identities separately:

- `TORCHSKY_HARNESS_COMMIT` identifies the code, data policy, and evaluation
  definitions used to run the campaign.
- `TORCHSKY_COMMIT` identifies the torchsky subject being evaluated. If an
  installed wheel is used, record the wheel SHA-256 and import path as well.

Do not use one repository's SHA as the other repository's identity. The run
record must say which subject was loaded and whether the evaluation completed;
process success by itself does not establish scientific acceptance.

## Work with gate evidence

Baselines, scorecards, comparator versions, and run artifacts live in the
harness repository. Use its current methodology for exact gate names, Pixi
profiles, tolerance changes, and archival. Change a baseline only after a
representative, reviewable run and preserve the previous evidence.

Performance changes should identify the algorithmic bottleneck, explain the
scaling benefit, and compare realistic sizes. Sphere and catalog reports should
include parity status and candidate-reduction effects where relevant. Missing
comparators and unavailable metrics must remain explicit; do not report them as
passes.

## Remote evaluation

CANFAR runs are explicit, revision-pinned campaigns. Use the harness launchers
only when the task authorizes a remote run, and pass separate exact harness and
subject revisions. Do not launch, cancel, or clean up remote sessions as part
of a local documentation or code change.
