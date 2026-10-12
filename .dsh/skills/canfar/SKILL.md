---
name: canfar
description: Run revision-pinned torchsky-harness scientific evaluations on CANFAR and collect their artifacts
metadata:
  project: torchsky-harness
---

# CANFAR evaluation workflow

Read the harness
[`docs/benchmarks.md`](../../../docs/benchmarks.md), the selected campaign
guide, and the runner scripts before using CANFAR. This skill covers remote
scientific evaluation owned by the harness. Package implementation and bounded
tests remain in torchsky; FITS I/O performance remains in torchfits.

## Revision identity

Every run must pin two independent revisions:

- `TORCHSKY_HARNESS_COMMIT` is the exact pushed harness revision containing
  the campaign, comparison code, and runner.
- `TORCHSKY_COMMIT` is the exact pushed torchsky subject revision under test.

Record both in the run artifacts. Never copy the harness SHA into the subject
field or infer the subject from the checkout used to submit the job. If a wheel
is the subject, record its SHA-256 and installed import path in addition to the
source revision.

## Runner use

The harness repository contains the current WCS/sphere, map/depth, and cuHPX
launch/fetch tools. Consult their `--help` and campaign documentation for
current mode names and resource profiles. Do not reuse one scientific campaign
runner for another domain merely because both execute on CANFAR.

Each run should write a unique immutable run directory with command, revisions,
environment/device, comparator versions, and output manifests. Fetch only the
run identified by its run ID and review the recorded criteria before treating
it as accepted evidence.

Long-running jobs require explicit task authorization. Do not launch,
cancel, delete, or clean up sessions unless the task specifically includes that
action. Core package verification should use the Pixi commands in the subject
repository's `AGENTS.md` and should not launch remote jobs.
