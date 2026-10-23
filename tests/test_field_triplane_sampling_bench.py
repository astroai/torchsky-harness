from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from torchsky_harness.benchmarks.bench_field_triplane_sampling import (
    TriplaneCase,
    main,
    profile_cases,
    run_case,
    run_suite,
)


@pytest.mark.parametrize(
    ("spatial_dim", "resolutions", "layout"),
    ((2, (8,), "contiguous"), (3, (4, 8), "broadcast")),
)
def test_triplane_candidate_matches_forward_and_gradients(
    spatial_dim: int,
    resolutions: tuple[int, ...],
    layout: str,
) -> None:
    case = TriplaneCase(
        spatial_dim=spatial_dim,
        features=2,
        resolutions=resolutions,
        points=12,
        coordinate_layout=layout,
        execution="forward_backward",
    )
    rows = run_case(case, warmup=0, runs=1)

    assert [row["implementation"] for row in rows] == [
        "current_per_plane",
        "batched_per_scale_candidate",
    ]
    assert all(row["status"] == "OK" for row in rows)
    assert rows[1]["forward_max_abs_error"] < 1.0e-6
    assert rows[1]["gradient_max_abs_error"] < 1.0e-5
    assert len(rows[1]["plane_gradient_max_abs_error"]) == (
        len(resolutions) * (3 if spatial_dim == 2 else 6)
    )
    assert set(rows[1]["coordinate_gradient_max_abs_error"]) == (
        {"x", "y", "lam"} if spatial_dim == 2 else {"x", "y", "lam", "z"}
    )


def test_profiles_cover_h6_axes() -> None:
    quick = profile_cases("quick")
    full = profile_cases("full")

    assert {case.spatial_dim for case in quick} == {2, 3}
    assert {case.coordinate_layout for case in quick} == {"contiguous", "broadcast"}
    assert {case.execution for case in quick} == {"forward", "forward_backward"}
    assert {len(case.resolutions) > 1 for case in quick} == {False, True}
    assert {case.features for case in full} >= {4, 8, 16, 32}
    assert {resolution for case in full for resolution in case.resolutions} >= {
        16,
        32,
        64,
        128,
        256,
    }
    assert {case.points for case in full} >= {1_000, 10_000, 100_000, 1_000_000}


def test_quick_suite_is_json_serializable_and_reports_cuda_capability() -> None:
    payload = run_suite(profile="quick", devices=("cpu", "cuda"), warmup=0, runs=1)
    restored = json.loads(json.dumps(payload))

    assert restored["schema"] == "torchsky.field_triplane_sampling/v1"
    assert restored["subject"]["import_path"]
    assert any(row.get("implementation") == "current_per_plane" for row in restored["results"])
    cuda_rows = [row for row in restored["results"] if row.get("device") == "cuda"]
    assert cuda_rows
    if cuda_rows[0]["status"] == "SKIPPED":
        assert cuda_rows[0]["reason"] == "torch.cuda.is_available() is false"


def test_main_writes_quick_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "triplane.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_field_triplane_sampling.py",
            "--profile",
            "quick",
            "--devices",
            "cpu",
            "--warmup",
            "0",
            "--runs",
            "1",
            "--json-out",
            str(output),
        ],
    )

    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["profile"] == "quick"
    assert json.loads(capsys.readouterr().out)["schema"] == (
        "torchsky.field_triplane_sampling/v1"
    )
