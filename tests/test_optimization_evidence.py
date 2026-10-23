from __future__ import annotations

import json

import pytest
import torch

from torchsky_harness import optimization_evidence as evidence


def test_measure_records_complete_distribution() -> None:
    calls = []
    result = evidence.measure(
        lambda: calls.append(len(calls)),
        device="cpu",
        warmups=2,
        runs=3,
    )
    assert len(calls) == 5
    assert len(result["samples_seconds"]) == 3
    assert result["minimum_seconds"] <= result["median_seconds"]
    assert result["median_seconds"] <= result["maximum_seconds"]


def test_measure_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="warmups"):
        evidence.measure(lambda: None, device="cpu", warmups=-1, runs=1)
    with pytest.raises(ValueError, match="runs"):
        evidence.measure(lambda: None, device="cpu", warmups=0, runs=0)


def test_tensor_checksum_includes_shape_and_values() -> None:
    base = torch.tensor([1.0, 2.0])
    assert evidence.tensor_checksum(base) == evidence.tensor_checksum(base.clone())
    assert evidence.tensor_checksum(base) != evidence.tensor_checksum(base.reshape(1, 2))
    assert evidence.tensor_checksum(base) != evidence.tensor_checksum(base + 1.0)


def test_payload_requires_complete_installed_subject(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        evidence,
        "identify_subject",
        lambda: {
            "version": "test",
            "import_path": "/installed/torchsky/__init__.py",
            "package_content_sha256": "abc",
        },
    )
    monkeypatch.setattr(evidence, "git_identity", lambda _: {"revision": "harness"})
    payload = evidence.build_payload(
        evaluation="bench_example",
        workstream="O0",
        profile="quick",
        device="cpu",
        rows=[{"case": "example", "status": "ok"}],
    )
    output = tmp_path / "evidence.json"
    evidence.write_payload(output, payload)
    assert json.loads(output.read_text()) == payload


def test_payload_rejects_incomplete_subject(monkeypatch) -> None:
    monkeypatch.setattr(evidence, "identify_subject", lambda: {"version": "test"})
    with pytest.raises(ValueError, match="subject identity"):
        evidence.build_payload(
            evaluation="bench_example",
            workstream="O0",
            profile="quick",
            device="cpu",
            rows=[{"case": "example"}],
        )
