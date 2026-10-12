"""Identify the installed scientific subject independently of the checkout."""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(path: Path) -> dict:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    if result.returncode:
        return {"revision": None, "dirty": None}
    root = Path(result.stdout.strip())
    revision_result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else None
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    diff = subprocess.check_output(["git", "-C", str(root), "diff", *(["HEAD"] if revision else []), "--binary"])
    return {"revision": revision, "dirty": bool(status), "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}


def identify_subject() -> dict:
    module = importlib.import_module("torchsky")
    package_path = Path(module.__file__).resolve()
    distribution = metadata.distribution("torchsky")
    digest = hashlib.sha256()
    for path in sorted(package_path.parent.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".so", ".pyd"}:
            digest.update(str(path.relative_to(package_path.parent)).encode())
            digest.update(bytes.fromhex(sha256(path)))
    direct_url = distribution.read_text("direct_url.json")
    return {
        "distribution": "torchsky", "version": distribution.version,
        "import_path": str(package_path), "package_content_sha256": digest.hexdigest(),
        "direct_url": json.loads(direct_url) if direct_url else None,
        "source": git_identity(package_path.parent), "python": sys.version,
        "platform": platform.platform(),
    }
