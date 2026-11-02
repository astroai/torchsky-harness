"""Discover existing evaluations without importing optional comparators."""
from importlib import import_module
from pathlib import Path


def evaluations() -> dict[str, str]:
    root = Path(import_module("torchsky_harness.benchmarks").__file__).parent
    # Evaluations are the ``bench_*`` entry points plus the two campaign runners
    # below. Diagnostics (``diagnose_*``), scorers (``score_*``), fitters, stagers,
    # probes and shared helpers are tools, not evaluations, and are not listed.
    helpers = {"bench_contract"}
    runners = {"run_ground_truth_competitive_large_union", "aggregate_ground_truth_competitive_large_union"}
    return {
        p.stem: "torchsky_harness.benchmarks." + p.stem
        for p in sorted(root.glob("*.py"))
        if (p.stem.startswith("bench_") and p.stem not in helpers) or p.stem in runners
    }


def domain(name: str) -> str:
    if "catalog" in name or "pipeline_table" in name:
        return "catalogs"
    if any(part in name for part in ("sphere", "healpix", "spin", "sht")):
        return "sphere"
    if any(part in name for part in ("wcs", "observed_place", "transforms")):
        return "wcs"
    if any(part in name for part in ("map", "smacs", "depth", "field", "inr", "solver", "strainer", "operator", "updec", "ground_truth")):
        return "map"
    return "cross_domain"


def resolve(name: str) -> str:
    available = evaluations()
    if name not in available:
        raise ValueError(f"Unknown evaluation {name!r}; use torchsky-harness list")
    return available[name]
