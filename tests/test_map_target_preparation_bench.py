from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import torchsky_harness.benchmarks.bench_map_image_plane as image_plane_bench
import torchsky_harness.benchmarks.bench_map_projection_mosaic_suite as projection_bench


def _result(*, provenance=None):
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    return SimpleNamespace(
        data=data,
        weight=torch.ones_like(data),
        inverse_variance=torch.full_like(data, 2.0),
        context=torch.ones((1, 4, 4), dtype=torch.int32),
        overlap_count=torch.ones((4, 4), dtype=torch.int16),
        coverage_moc=None,
        provenance=provenance,
    )


def test_target_preparation_row_is_complete_without_cache_api(monkeypatch) -> None:
    monkeypatch.setattr(
        image_plane_bench.image_plane_mod, "_prepare_target", lambda: None
    )
    monkeypatch.setattr(
        image_plane_bench.image_plane_mod,
        "_project_observation_to_target",
        lambda: None,
    )

    def fake_coadd(observations, **kwargs):
        image_plane_bench.image_plane_mod._prepare_target()
        for _ in observations:
            image_plane_bench.image_plane_mod._project_observation_to_target()
        return _result()

    monkeypatch.setattr(image_plane_bench, "coadd_image_plane", fake_coadd)
    row = image_plane_bench._profile_target_preparation_workload(
        case="tan_quick",
        projection="TAN",
        observations=[SimpleNamespace()],
        target_header={"NAXIS1": 4, "NAXIS2": 4},
        tile_size=2,
        exposure_count=4,
        value_semantics="surface_brightness",
        footprint_order=None,
        runs=2,
        warmups=1,
        device="cpu",
        cache_capacity=None,
        environment={"subject": {"import_path": "/installed/torchsky"}},
    )

    assert row["status"] == "OK"
    assert row["target_preparation_calls"] == 1
    assert row["projection_calls"] == 4
    assert row["cache_metrics_available"] is False
    assert row["cache_metrics_unavailable_reason"]
    assert set(row["cache"]) == {
        "hits",
        "misses",
        "evictions",
        "live_entries",
        "peak_bytes",
    }
    assert all(
        row["checksums"][name]
        for name in ("data", "weight", "variance", "context", "overlap")
    )
    assert row["checksums"]["footprint"] is None
    assert all(
        difference in (None, 0.0)
        for difference in row["max_abs_difference_from_uncached"].values()
    )
    assert row["environment"]["subject"]["import_path"] == "/installed/torchsky"


def test_target_preparation_row_records_optional_cache_counters(monkeypatch) -> None:
    monkeypatch.setattr(
        image_plane_bench.image_plane_mod, "_prepare_target", lambda: None
    )
    monkeypatch.setattr(
        image_plane_bench.image_plane_mod,
        "_project_observation_to_target",
        lambda: None,
    )

    capacities = []

    def fake_coadd(observations, *, target_cache_capacity=None, **kwargs):
        capacities.append(target_cache_capacity)
        image_plane_bench.image_plane_mod._prepare_target()
        for _ in observations:
            image_plane_bench.image_plane_mod._project_observation_to_target()
        return _result(
            provenance={
                "target_cache": {
                    "hits": 3,
                    "misses": 1,
                    "evictions": 0,
                    "live_entries": 1,
                    "peak_bytes": 4096,
                }
            }
        )

    monkeypatch.setattr(image_plane_bench, "coadd_image_plane", fake_coadd)
    row = image_plane_bench._profile_target_preparation_workload(
        case="tan_cached",
        projection="TAN",
        observations=[SimpleNamespace()],
        target_header={"NAXIS1": 4, "NAXIS2": 4},
        tile_size=2,
        exposure_count=1,
        value_semantics="surface_brightness",
        footprint_order=None,
        runs=1,
        warmups=0,
        device="cpu",
        cache_capacity=4,
    )

    assert row["cache_capacity_argument"] == "target_cache_capacity"
    assert row["cache_capacity_requested"] == 4
    assert capacities == [0, 4]
    assert row["cache_metrics_available"] is True
    assert row["cache"] == {
        "hits": 3,
        "misses": 1,
        "evictions": 0,
        "live_entries": 1,
        "peak_bytes": 4096,
    }


def test_image_plane_target_preparation_cli_writes_json(
    tmp_path: Path, monkeypatch
) -> None:
    row = {
        "case": "tan_target_preparation_quick",
        "tile_size": 4,
        "n_obs": 1,
        "total_time_ms": 1.0,
        "target_preparation_calls": 1.0,
        "target_preparation_time_ms": 0.2,
        "projection_time_ms": 0.7,
        "checksums": {"data": "a" * 64},
        "max_abs_difference_from_uncached": {"data": 0.0},
        "environment": {"subject": {"import_path": "/installed/torchsky"}},
    }
    monkeypatch.setattr(
        image_plane_bench,
        "_run_target_preparation_quick",
        lambda *args, **kwargs: [row],
    )
    json_out = tmp_path / "image-plane.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_map_image_plane.py",
            "--profile",
            "target-preparation-quick",
            "--json-out",
            str(json_out),
        ],
    )

    assert image_plane_bench.main() == 0
    assert json.loads(json_out.read_text(encoding="utf-8")) == [row]


def test_projection_target_preparation_cli_writes_complete_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    row = {
        "suite": "target_preparation",
        "case": "SIN:test@8",
        "projection": "SIN",
        "n_obs": 1,
        "tile_size": 8,
        "value_semantics": "surface_brightness",
        "footprint_order": 8,
        "status": "OK",
        "target_width": 8,
        "target_height": 8,
        "total_time_ms": 1.0,
        "target_preparation_calls": 1,
        "target_preparation_time_ms": 0.2,
        "projection_calls": 1,
        "projection_time_ms": 0.7,
        "mpix_s": 0.1,
        "cache_capacity_requested": None,
        "cache_metrics_available": False,
        "cache_metrics_unavailable_reason": "unavailable",
        "checksums": {"data": "a" * 64},
        "max_abs_difference_from_uncached": {"data": 0.0},
        "environment": {"subject": {"import_path": "/installed/torchsky"}},
    }
    monkeypatch.setattr(
        projection_bench, "_run_target_preparation_suite", lambda *args, **kwargs: [row]
    )
    json_out = tmp_path / "result.json"
    markdown_out = tmp_path / "result.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_map_projection_mosaic_suite.py",
            "--target-preparation-profile",
            "quick",
            "--target-preparation-only",
            "--target-preparation-cases",
            "SIN:tan_aligned_low_noise@8",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ],
    )

    assert projection_bench.main() == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload == [row]
    markdown = markdown_out.read_text(encoding="utf-8")
    assert "Target Preparation Quick Profile" in markdown
    assert "target_preparation" in markdown
