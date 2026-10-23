import json
from types import SimpleNamespace

import pytest

from torchsky_harness import runner
from torchsky_harness.benchmarks.config import benchmark_subprocess_env
from torchsky_harness.registry import domain, resolve


def test_subprocess_cannot_inject_subject_checkout(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/torchsky/src")
    env = benchmark_subprocess_env()
    assert "PYTHONPATH" not in env
    assert env["PYTHONNOUSERSITE"] == "1"


def test_unknown_evaluation_fails_closed():
    with pytest.raises(ValueError):
        resolve("../../src/torchsky/wcs/core")


def test_optimization_benchmarks_have_owned_domains():
    assert domain("bench_solver_pcg") == "map"
    assert domain("bench_field_triplane_sampling") == "map"
    assert domain("bench_wcs_coupled_calibration") == "wcs"


@pytest.mark.parametrize("code,state", [(0, "completed"), (2, "failed")])
def test_execution_state_does_not_claim_scientific_validity(tmp_path, monkeypatch, code, state):
    monkeypatch.setattr(runner, "identify_subject", lambda: {"version": "test", "import_path": "/installed/torchsky"})
    monkeypatch.setattr(runner, "git_identity", lambda _: {"revision": "test"})
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=code))
    run_dir = tmp_path / "run"
    assert runner.run("bench_sphere_sparse", [], run_dir) == code
    record = json.loads((run_dir / "execution.json").read_text())
    assert record["execution"] == state
    assert record["scientific_validity"] == "not_assessed"
    assert record["comparison_eligibility"] == "not_assessed"
    with pytest.raises(FileExistsError):
        runner.run("bench_sphere_sparse", [], run_dir)


def test_interrupted_run_retains_failure_record(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "identify_subject", lambda: {})
    monkeypatch.setattr(runner, "git_identity", lambda _: {})

    def fail(*args, **kwargs):
        raise OSError("executor unavailable")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(OSError):
        runner.run("bench_sphere_sparse", [], tmp_path / "interrupted")
    record = json.loads((tmp_path / "interrupted/execution.json").read_text())
    assert record["execution"] == "interrupted"
    assert record["returncode"] is None
