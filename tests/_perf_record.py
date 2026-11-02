"""Append normalized comparator-performance rows (record, never gate).

Rows go to ``$TORCHSKY_HARNESS_PERF_DIR`` (default ``<repo>/benchmark_results``) as JSON lines in
``comparator_perf.jsonl``; ``scripts/canfar/summarize.py`` renders them as the "comparator
performance" table. Schema: area, operation, comparator, size, device, torchsky_s,
comparator_s, throughput_per_s (size / torchsky_s), speedup (comparator_s / torchsky_s), note.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def perf_dir() -> Path:
    return Path(os.environ.get("TORCHSKY_HARNESS_PERF_DIR") or Path(__file__).resolve().parents[1] / "benchmark_results")


def record(
    area: str, operation: str, comparator: str, size: int, torchsky_s: float, comparator_s: float,
    device: str = "cpu", note: str = "",
) -> dict:  # fmt: skip
    row = {
        "area": area, "operation": operation, "comparator": comparator, "size": int(size), "device": device,
        "torchsky_s": float(torchsky_s), "comparator_s": float(comparator_s),
        "throughput_per_s": float(size) / float(torchsky_s) if torchsky_s > 0 else None,
        "speedup": float(comparator_s) / float(torchsky_s) if torchsky_s > 0 else None,
        "note": note,
    }  # fmt: skip
    path = perf_dir()
    path.mkdir(parents=True, exist_ok=True)
    with (path / "comparator_perf.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    return row
