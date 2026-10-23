from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "bench_wcs_coupled_calibration.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "bench_wcs_coupled_calibration", BENCHMARK
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profiles_cover_required_graph_families() -> None:
    module = _load_benchmark()
    for cases in module.PROFILES.values():
        assert {case.anchoring for case in cases} >= {"all", "none", "alternating"}
        assert {case.row_order for case in cases} == {"grouped", "shuffled"}
        assert any(case.components == 1 for case in cases)
        assert any(case.components > 1 for case in cases)


def test_component_metrics_are_stable_under_row_permutation() -> None:
    module = _load_benchmark()
    common = dict(
        components=3,
        exposures_per_component=2,
        sources_per_component=3,
        bands=1,
        anchoring="alternating",
        seed=17,
    )
    grouped = module.run_case(
        module.CaseSpec("grouped", row_order="grouped", **common), runs=1
    )
    shuffled = module.run_case(
        module.CaseSpec("shuffled", row_order="shuffled", **common), runs=1
    )

    assert grouped["component_signature"] == shuffled["component_signature"]
    assert grouped["component_reference_anchored"] == [True, False, True]
    assert grouped["gauge_exposure_indices"] == [2]
    assert grouped["deterministic_metrics"]["fit_row_count"] == 18
    assert grouped["deterministic_metrics"]["photometric_log_residual_rms"] < 1.0e-8


def test_main_writes_json_with_installed_subject(monkeypatch, tmp_path: Path) -> None:
    module = _load_benchmark()
    output = tmp_path / "coupled.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_wcs_coupled_calibration",
            "--profile",
            "quick",
            "--cases",
            "many_components_mixed_grouped,many_components_mixed_shuffled",
            "--runs",
            "1",
            "--json-out",
            str(output),
        ],
    )

    assert module.main() == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["benchmark"] == "wcs_coupled_calibration"
    assert Path(artifact["subject"]["import_path"]).name == "__init__.py"
    assert len(artifact["rows"]) == 2
    assert all(row["row_order_stable"] for row in artifact["rows"])
    assert all(
        set(row["median_ms"])
        >= {
            "graph_construction_total",
            "photometric_component_discovery",
            "coupled_solve",
        }
        for row in artifact["rows"]
    )
