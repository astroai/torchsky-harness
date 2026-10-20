"""Keep torchsky-harness's repository boundary explicit.

The harness owns benchmarks, gates, evidence, campaigns, and evaluation
scripts. It must not embed a torchsky source checkout or core implementation.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_embedded_torchsky_source_tree() -> None:
    assert not (ROOT / "src" / "torchsky").exists()
    assert not (ROOT / "torchsky").exists()


def test_benchmark_tree_is_present() -> None:
    assert (ROOT / "benchmarks").is_dir()
    assert (ROOT / "benchmarks" / "bench_native_contracts.py").is_file()


def test_no_installed_subject_inside_repo() -> None:
    # The subject must be installed into a Pixi environment, not unpacked into
    # the harness checkout.
    assert not (ROOT / "site-packages").exists()
