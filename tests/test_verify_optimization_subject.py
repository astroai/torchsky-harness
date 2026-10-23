from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _verification_module():
    path = Path(__file__).parents[1] / "scripts" / "verify_optimization_subject.py"
    spec = importlib.util.spec_from_file_location("verify_optimization_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subject(*, revision: str = "abc", dirty: bool = False) -> dict:
    return {
        "import_path": "/installed/torchsky/__init__.py",
        "source": {"revision": revision, "dirty": dirty},
    }


def test_subject_check_accepts_clean_expected_revision(monkeypatch) -> None:
    verify = _verification_module()
    monkeypatch.setattr(verify, "identify_subject", lambda: _subject())
    monkeypatch.setenv("TORCHSKY_EXPECTED_COMMIT", "abc")
    assert verify.main() == 0


def test_subject_check_rejects_dirty_source(monkeypatch) -> None:
    verify = _verification_module()
    monkeypatch.setattr(verify, "identify_subject", lambda: _subject(dirty=True))
    with pytest.raises(SystemExit, match="dirty"):
        verify.main()


def test_subject_check_rejects_wrong_revision(monkeypatch) -> None:
    verify = _verification_module()
    monkeypatch.setattr(verify, "identify_subject", lambda: _subject())
    monkeypatch.setenv("TORCHSKY_EXPECTED_COMMIT", "def")
    with pytest.raises(SystemExit, match="expected"):
        verify.main()
