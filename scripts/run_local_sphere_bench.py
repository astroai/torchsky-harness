"""Run an evaluator against an explicitly installed torchsky subject."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from torchsky_harness.cli import main as harness_main


def _ensure_local_cpp() -> None:
    """Fail closed; installation/building is an explicit separate operation."""
    subprocess.run([sys.executable, "-c", "import torchsky; import torchsky._cpp"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    _ensure_local_cpp()
    return harness_main(["run", Path(args.script).stem, "--", *args.script_args])


if __name__ == "__main__":
    raise SystemExit(main())
