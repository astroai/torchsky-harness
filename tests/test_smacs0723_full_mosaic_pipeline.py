from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("requests")

pytest.importorskip("requests")

from torchsky_harness.scripts import smacs0723_full_mosaic_pipeline as pipeline

ROOT = pipeline.ROOT
_band_filename_contains = pipeline._band_filename_contains
_ensure_local_cpp = pipeline._ensure_local_cpp
build_parser = pipeline.build_parser


def test_band_filename_contains_matches_long_wave_only() -> None:
    assert _band_filename_contains("F277W", "b") == "nrcblong"
    assert _band_filename_contains("F356W", "a") == "nrcalong"
    assert _band_filename_contains("F200W", "b") is None


def test_parser_defaults_are_simple() -> None:
    args = build_parser().parse_args([])
    assert str(args.root).endswith("/data")
    assert args.input_product == "cal"
    assert args.train_model == "spectral_basis_field"
    assert args.train_steps == 12


def test_ensure_local_cpp_installs_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"install_calls": 0, "probe_calls": 0}
    _real_import_module = importlib.import_module

    def _fake_import_module(name: str):
        if name == "torchsky":
            return object()
        if name == "torchsky.cpp":
            raise ModuleNotFoundError(name)
        return _real_import_module(name)

    def _fake_check_call(cmd, cwd):
        del cmd, cwd
        state["install_calls"] += 1

    def _fake_call(cmd, cwd):
        assert cmd == [sys.executable, "-c", "import torchsky; import torchsky.cpp"]
        assert cwd == ROOT
        state["probe_calls"] += 1
        return 0

    monkeypatch.setattr(
        "torchsky_harness.scripts.smacs0723_full_mosaic_pipeline.importlib.import_module",
        _fake_import_module,
    )
    monkeypatch.setattr(
        "torchsky_harness.scripts.smacs0723_full_mosaic_pipeline.subprocess.check_call",
        _fake_check_call,
    )
    monkeypatch.setattr(
        "torchsky_harness.scripts.smacs0723_full_mosaic_pipeline.subprocess.call",
        _fake_call,
    )

    _ensure_local_cpp()

    assert state == {"install_calls": 1, "probe_calls": 1}
