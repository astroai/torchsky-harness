"""Locate external astronomy binaries (SWarp, Montage, SCAMP, solve-field, WCSTools, ...).

Paths come from ``PATH`` only; astrometry index files come from ``ASTROMETRY_INDEX_DIR``
(never hard-coded). ``TORCHSKY_HARNESS_REQUIRE_TOOLS=1`` (set by ``IMAGE_PROFILE=improc`` in
``scripts/canfar/run_campaign.sh``) turns a missing binary into a test failure.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

DEFAULT_INDEX_DIR = "/arc/projects/uvickbos/astrometry.net/data"
TOOL_NAMES = {
    "swarp": ("swarp", "SWarp"),
    "scamp": ("scamp",),
    "sextractor": ("source-extractor", "sex"),
    "psfex": ("psfex",),
    "solve-field": ("solve-field",),
    "mProject": ("mProject",),
    "mProjExec": ("mProjExec",),
    "mAdd": ("mAdd",),
    "xy2sky": ("xy2sky",),
    "sky2xy": ("sky2xy",),
    "stilts": ("stilts",),
}


def require_tools() -> bool:
    return os.environ.get("TORCHSKY_HARNESS_REQUIRE_TOOLS") == "1"


def find(tool: str) -> str | None:
    for name in TOOL_NAMES.get(tool, (tool,)):
        found = shutil.which(name)
        if found:
            return found
    return None


def need(tool: str) -> str:
    """Return the tool path or skip (the conftest turns a gated skip into a failure)."""
    found = find(tool)
    if found is None:
        pytest.skip(f"{tool} binary not on PATH")
    return found


def index_dir() -> Path:
    return Path(os.environ.get("ASTROMETRY_INDEX_DIR", DEFAULT_INDEX_DIR))


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600, **kw)


FLOAT = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def stilts_command() -> list[str] | None:
    """Command prefix that runs STILTS: the ``stilts`` launcher, else ``java -jar $TORCHSKY_HARNESS_STILTS_JAR``."""
    exe = find("stilts")
    if exe:
        return [exe]
    jar = os.environ.get("TORCHSKY_HARNESS_STILTS_JAR")
    java = os.environ.get("TORCHSKY_HARNESS_JAVA") or shutil.which("java")
    if jar and java and Path(jar).is_file():
        return [java, "-jar", jar]
    return None
