from __future__ import annotations

import os
import importlib.util
from pathlib import Path

# Centralized output directory for all benchmarks
DEFAULT_OUTPUT_DIR = Path(os.environ.get("TORCHSKY_HARNESS_OUTPUT_ROOT", "benchmarks_results"))
REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_src_on_path() -> None:
    """Validate the installed subject without changing import resolution."""
    if importlib.util.find_spec("torchsky") is None:
        raise RuntimeError("Install the selected torchsky build before evaluation")


def benchmark_subprocess_env() -> dict[str, str]:
    """Use installed packages, isolated from user-site/PYTHONPATH contamination."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env
