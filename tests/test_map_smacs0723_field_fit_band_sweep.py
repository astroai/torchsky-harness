from __future__ import annotations

from pathlib import Path

from torchsky_harness.benchmarks import bench_map_smacs0723_field_fit_band_sweep as sweep

_format_markdown = sweep._format_markdown


def test_band_sweep_markdown_lists_band_and_model() -> None:
    text = _format_markdown(
        [
            {
                "band": "F277W",
                "model": "sky_encoder_resmlp",
                "status": "OK",
                "support_rmse": 0.1,
                "aligned_rmse": 0.09,
                "affine_slope": 0.95,
                "uniform_rmse": 0.2,
                "support_flux_ratio": 1.01,
                "support_frac": 0.9,
                "fit_time_ms": 12.3,
            }
        ]
    )
    assert "F277W" in text
    assert "sky_encoder_resmlp" in text


def test_band_sweep_source_mentions_best_overlap_strategy() -> None:
    source = Path(sweep.__file__).read_text()
    assert "--crop-strategy" in source
    assert "best_overlap" in source
