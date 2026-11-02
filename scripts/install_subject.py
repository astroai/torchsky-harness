"""Install an explicit torchsky subject into the active (Pixi) environment.

Usage (from the harness root)::

    pixi run -e test install-subject ../torchsky          # source checkout
    pixi run -e test install-subject dist/torchsky-0.5.0rc1.tar.gz
    pixi run -e test install-subject dist/torchsky-0.5.0rc1-cp313-*.whl

The install is always non-editable (``pip install --no-build-isolation
--no-deps``; build requirements come from the environment).  A checkout is
copied (tracked + untracked-not-ignored files) into a scratch directory first, so
the build never writes into the subject tree and no checkout ever lands on
``sys.path``.  The source identity (git revision, dirty flag, tracked diff
digest, archive sha256) is recorded in ``<prefix>/torchsky-subject-install.json``
and merged into every report by ``torchsky_harness.subject.identify_subject``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # harness helpers only

from torchsky_harness.subject import git_identity, sha256  # noqa: E402

RECORD_NAME = "torchsky-subject-install.json"


def _copy_checkout(src: Path, dst: Path) -> None:
    listing = subprocess.run(
        ["git", "-C", str(src), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        check=False,
    )
    if listing.returncode:
        raise SystemExit(f"{src} is not a git checkout; pass an sdist or wheel instead")
    for rel in filter(None, listing.stdout.decode().split("\0")):
        source = src / rel
        if source.is_file() or source.is_symlink():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dst / rel, follow_symlinks=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("subject", type=Path, help="checkout directory, sdist (.tar.gz) or wheel")
    args = parser.parse_args(argv)
    subject = args.subject.expanduser().resolve()
    if not subject.exists():
        raise SystemExit(f"subject not found: {subject}")
    record: dict = {"requested": str(subject)}
    pip = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--force-reinstall"]
    with tempfile.TemporaryDirectory(prefix="torchsky-subject-") as scratch:
        if subject.is_dir():
            record["kind"] = "source_tree"
            record.update(git_identity(subject))
            tree = Path(scratch) / "src"
            tree.mkdir()
            _copy_checkout(subject, tree)
            target = str(tree)
        else:
            record["kind"] = "wheel" if subject.suffix == ".whl" else "sdist"
            record["sha256"] = sha256(subject)
            target = str(subject)
        subprocess.run([*pip, target], check=True)
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import torchsky, importlib.metadata as m; print(torchsky.__file__); print(m.version('torchsky'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tempfile.gettempdir(),
    )
    path, version = probe.stdout.split()
    record.update(import_path=path, version=version)
    if subject.is_dir() and Path(path).is_relative_to(subject):
        raise SystemExit(f"subject imports from the checkout ({path}); expected site-packages")
    (Path(sys.prefix) / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
