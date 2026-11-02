# torchsky-harness

This repository owns scientific validation, upstream parity, benchmarks,
comparison adapters, reports, and CANFAR evaluation orchestration for torchsky.
Torchsky owns production implementations and bounded package tests. Torchfits
owns FITS/table I/O implementations and their correctness/performance gates.

- Use Pixi. Do not guess a Python interpreter or install into user site packages.
- Evaluate an explicitly installed subject; never inject a torchsky checkout into
  sys.path, load individual implementation files, or silently build the harness
  as if it were torchsky. Record the subject identity and actual import path.
  Install the subject with `pixi run -e <env> install-subject <checkout|sdist|wheel>`
  (non-editable; writes the identity record); CANFAR campaigns use
  `scripts/canfar/run_campaign.sh` (see `docs/canfar.md`).
- Use public torchfits APIs at FITS boundaries. Astropy metadata/WCS and other
  ecosystem packages are reference comparators, not production backends.
- Preserve required gates, tolerances, fixtures, licenses and historical evidence.
  Missing required comparators fail coverage; unavailable metrics have reasons.
- Keep changes focused. Do not change scientific algorithms during extraction.
- Run `pixi run preflight` and targeted tests. Full benchmark/scientific suites
  are opt-in; no remote job submission or session cleanup without task scope.
- Preserve frozen active job sources/artifacts. Large data and fitted products
  stay in durable external storage with manifests, not Git.
- Add modules only when there is existing implementation or evaluation to move.
  Do not scaffold future PSF/detection/segmentation/SSL suites.
