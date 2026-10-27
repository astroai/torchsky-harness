from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "bench_cfht_mef_mosaic.py"
SPEC = importlib.util.spec_from_file_location("bench_cfht_mef_mosaic", BENCHMARK)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_comparison_mask_uses_finite_common_positive_weight_support() -> None:
    torchsky_data = np.array([[1.0, 2.0], [np.nan, -5.0]], dtype=np.float32)
    swarp_data = np.array([[1.5, 2.0], [3.0, -7.0]], dtype=np.float32)
    torchsky_weight = np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    swarp_weight = np.array([[3.0, 4.0], [5.0, 0.0]], dtype=np.float32)
    metrics = BENCH.comparison_metrics(
        torchsky_data, torchsky_weight, swarp_data, swarp_weight
    )
    assert metrics["common_pixels"] == 1
    assert metrics["rmse"] == 0.5


def test_cfht_benchmark_uses_matched_combine_mask_and_timing() -> None:
    source = BENCHMARK.read_text()
    assert "COMBINE_TYPE           WEIGHTED" in source
    assert "COMBINE_TYPE           MEDIAN" not in source
    assert "swarp_weight > 0" in source
    assert "sw_data > 0" not in source
    assert "sw_elapsed - decomp_time" not in source
    assert '"claim": None' in source
    assert "--require-swarp" in source


def test_canfar_runner_is_scratch_bounded_and_persists_on_exit() -> None:
    runner = (ROOT / "campaigns/cfht_union_pitcairn/run_canfar.sh").read_text()
    assert "export PYTHONNOUSERSITE=1" in runner
    assert "unset PYTHONPATH" in runner
    assert "TORCHSKY_HARNESS_COMMIT" in runner
    assert "TORCHSKY_COMMIT" in runner
    assert "work_root=/scratch/" in runner
    assert "CFHT_PERSISTENT_DEST must be under /arc/projects" in runner
    assert "trap persist EXIT" in runner
    assert 'cp -a "${artifact_root}/." "${persistent_dest}/"' in runner
    assert "bench_cfht_megacam.py" in runner
    assert "--require-swarp" in runner
    assert '--band "${band}"' in runner


def test_command_manifest_declares_no_claims() -> None:
    import json

    payload = json.loads(
        (ROOT / "campaigns/cfht_union_pitcairn/command.json").read_text()
    )
    assert payload["claims"] == []
    assert payload["scratch_only_downloads"] is True
    assert payload["persistent_artifacts"] == [
        "logs",
        "results",
        "manifests",
        "products",
    ]
