from __future__ import annotations

from pathlib import Path

import pytest

from torchsky_harness import subject as subject_module
from torchsky_harness.subject import source_identity

WHEEL_SHA = "f" * 64


def test_registry_install_has_no_source_identity() -> None:
    assert source_identity(None) == {
        "kind": "registry",
        "revision": None,
        "dirty": None,
    }


def test_wheel_install_records_archive_digest_not_enclosing_repo() -> None:
    identity = source_identity(
        {
            "archive_info": {"hash": f"sha256={WHEEL_SHA}", "hashes": {"sha256": WHEEL_SHA}},
            "url": "file:///tmp/dist/torchsky-0.4.0-cp313-cp313-macosx_11_0_arm64.whl",
        }
    )

    assert identity["kind"] == "archive"
    assert identity["sha256"] == WHEEL_SHA
    assert identity["revision"] is None
    assert identity["dirty"] is None
    assert identity["url"].endswith(".whl")


def test_source_tree_install_resolves_git_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def fake_git_identity(path: Path) -> dict:
        calls.append(path)
        return {"revision": "a" * 40, "dirty": False}

    monkeypatch.setattr(subject_module, "git_identity", fake_git_identity)
    identity = source_identity(
        {
            "dir_info": {"editable": True},
            "url": "file:///Users/example/torchsky",
        }
    )

    assert calls == [Path("/Users/example/torchsky")]
    assert identity["kind"] == "source_tree"
    assert identity["revision"] == "a" * 40
    assert identity["dirty"] is False


def test_unknown_source_shape_is_reported() -> None:
    identity = source_identity({"url": "https://example.invalid/torchsky"})

    assert identity["kind"] == "unknown"
    assert identity["revision"] is None
