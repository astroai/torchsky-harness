"""Shared contracts for focused optimization evidence."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any

import torch
from torch import Tensor

from torchsky_harness import __version__
from torchsky_harness.subject import git_identity, identify_subject

SCHEMA_NAME = "torchsky-harness.optimization-evidence.v1"


def synchronize(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps":
        torch.mps.synchronize()


def measure(
    function: Callable[[], Any],
    *,
    device: torch.device | str,
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if runs < 1:
        raise ValueError("runs must be positive")
    for _ in range(warmups):
        function()
    synchronize(device)
    samples = []
    for _ in range(runs):
        synchronize(device)
        started = time.perf_counter()
        function()
        synchronize(device)
        samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
        "warmups": warmups,
        "runs": runs,
    }


def tensor_checksum(*values: Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def reset_peak_memory(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)


def peak_memory_bytes(device: torch.device | str) -> int | None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        return int(torch.cuda.max_memory_allocated(resolved))
    return None


def build_payload(
    *,
    evaluation: str,
    workstream: str,
    profile: str,
    device: torch.device | str,
    rows: Sequence[Mapping[str, Any]],
    limitations: Sequence[str] = (),
) -> dict[str, Any]:
    if not evaluation:
        raise ValueError("evaluation must be non-empty")
    if not workstream:
        raise ValueError("workstream must be non-empty")
    if not rows:
        raise ValueError("rows must be non-empty")
    root = Path(__file__).resolve().parents[2]
    payload = {
        "schema": SCHEMA_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": evaluation,
        "workstream": workstream,
        "profile": profile,
        "device": str(torch.device(device)),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "subject": identify_subject(),
        "harness": {
            "version": __version__,
            "source": git_identity(root),
        },
        "rows": [dict(row) for row in rows],
        "limitations": list(limitations),
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "generated_at",
        "evaluation",
        "workstream",
        "profile",
        "device",
        "runtime",
        "subject",
        "harness",
        "rows",
        "limitations",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"optimization evidence missing fields: {missing}")
    if payload["schema"] != SCHEMA_NAME:
        raise ValueError(f"unsupported optimization evidence schema: {payload['schema']!r}")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("optimization evidence rows must be a non-empty list")
    subject = payload["subject"]
    if not isinstance(subject, Mapping) or any(
        not subject.get(key)
        for key in ("version", "import_path", "package_content_sha256")
    ):
        raise ValueError("optimization evidence subject identity is incomplete")


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    validate_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)
