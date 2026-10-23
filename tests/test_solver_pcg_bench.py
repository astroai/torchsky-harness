from __future__ import annotations

import json
from pathlib import Path

from torchsky_harness.benchmarks import bench_solver_pcg


def test_diagonal_case_recovers_known_solution_and_reports_counters() -> None:
    row = bench_solver_pcg.run_case(
        "diagonal",
        size=128,
        max_iterations=8,
        runs=1,
        device="cpu",
    )

    assert row["known_solution_correct"] is True
    assert row["final_true_relative_residual"] < 1.0e-5
    assert row["operator_applications"] == row["observed_operator_applications"]
    assert row["preconditioner_applications"] == row["observed_preconditioner_applications"]
    assert row["iterations"] >= 1


def test_quick_smoke_artifact_has_controlled_systems() -> None:
    artifact = bench_solver_pcg.run_benchmark(
        profile="quick",
        device="cpu",
        cases=("diagonal", "sparse", "image"),
        iteration_budgets=(32,),
        runs=1,
        include_subject=False,
    )

    assert artifact["profile"] == "quick"
    assert {row["case"] for row in artifact["rows"]} == {
        "sparse",
        "image",
        "diagonal",
    }
    assert all(row["known_solution_correct"] for row in artifact["rows"])
    json.dumps(artifact)


def test_cli_writes_subject_identity_json(tmp_path) -> None:
    output = tmp_path / "pcg.json"

    assert (
        bench_solver_pcg.main(
            [
                "--profile",
                "quick",
                "--cases",
                "diagonal",
                "--iteration-budgets",
                "8",
                "--runs",
                "1",
                "--json-out",
                str(output),
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text())
    subject = artifact["subject"]
    assert subject["distribution"] == "torchsky"
    assert subject["version"]
    assert subject["import_path"] == str(torchsky_path())
    assert subject["package_content_sha256"]


def torchsky_path() -> Path:
    import torchsky

    return Path(torchsky.__file__).resolve()
