"""Run STILTS crossmatch tasks on FITS tables written with torchfits (no astropy.io.fits)."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest
import torch
import torchfits

import _external_tools as xt


def need_stilts() -> list[str]:
    cmd = xt.stilts_command()
    if cmd is None:
        pytest.skip("stilts not on PATH (and TORCHSKY_HARNESS_STILTS_JAR/java unset)")
    return cmd


def write_table(path: Path, **columns: np.ndarray) -> Path:
    data = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in columns.items()}
    torchfits.table.write(str(path), data, overwrite=True)
    return path


def run_stilts(args: list[str], out: Path) -> tuple[dict[str, np.ndarray], float]:
    """Run ``stilts <args> out=<csv> ofmt=csv``; return (columns, wall seconds incl. JVM start)."""
    cmd = need_stilts()
    t0 = time.perf_counter()
    done = subprocess.run(
        [*cmd, *args, f"out={out}", "ofmt=csv"], capture_output=True, text=True, timeout=3000, check=False
    )
    wall = time.perf_counter() - t0
    if done.returncode != 0:
        raise RuntimeError(f"stilts {args[0]} failed:\n{done.stderr[-2000:]}")
    with out.open(newline="") as stream:
        rows = list(csv.reader(stream))
    header, body = rows[0], rows[1:]
    cols = {}
    for i, name in enumerate(header):
        vals = [r[i] for r in body]
        try:
            cols[name] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            cols[name] = np.array(vals)
    return cols, wall


def tmatch2(
    a: Path, b: Path, out: Path, *, radius: float, find: str = "best1", join: str = "1and2",
    matcher: str = "sky", values1: str = "ra dec", values2: str = "ra dec", task: str = "tmatch2",
) -> tuple[dict[str, np.ndarray], float]:  # fmt: skip
    """Pairwise STILTS match; both inputs carry an ``id`` row-index column (0-based)."""
    if task == "tskymatch2":
        args = ["tskymatch2", f"in1={a}", f"in2={b}", "ra1=ra", "dec1=dec", "ra2=ra", "dec2=dec", f"error={radius}"]
    else:
        args = [
            "tmatch2", f"in1={a}", f"in2={b}", f"matcher={matcher}", f"params={radius}",
            f"values1={values1}", f"values2={values2}",
        ]  # fmt: skip
    args += [f"find={find}", f"join={join}", "ifmt1=fits", "ifmt2=fits"]
    return run_stilts(args, out)


def pairs(cols: dict[str, np.ndarray]) -> set[tuple[int, int]]:
    return set(zip(cols["id_1"].astype(np.int64).tolist(), cols["id_2"].astype(np.int64).tolist()))
