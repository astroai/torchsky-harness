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
from urllib.parse import unquote, urlparse


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


def source_identity(direct_url: dict | None) -> dict:
    """Describe where the installed subject came from.

    A wheel install must not inherit the VCS identity of the repository that
    happens to contain its site-packages directory, so only an editable/dir
    install is resolved through git.
    """
    if direct_url is None:
        return {"kind": "registry", "revision": None, "dirty": None}
    url = str(direct_url.get("url", ""))
    archive = direct_url.get("archive_info")
    if archive:
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        digest = None
        if isinstance(hashes, dict):
            digest = hashes.get("sha256")
        if digest is None and isinstance(archive, dict):
            digest = str(archive.get("hash", "")).removeprefix("sha256=") or None
        return {
            "kind": "archive",
            "revision": None,
            "dirty": None,
            "url": url,
            "sha256": digest,
        }
    if direct_url.get("dir_info") is not None:
        parsed = urlparse(url)
        root = Path(unquote(parsed.path)) if parsed.scheme == "file" else None
        identity = (
            git_identity(root)
            if root is not None
            else {"revision": None, "dirty": None}
        )
        return {"kind": "source_tree", "url": url, **identity}
    return {
        "kind": "unknown",
        "revision": None,
        "dirty": None,
        "url": url,
    }


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
        "source": source_identity(json.loads(direct_url) if direct_url else None),
        "python": sys.version,
        "platform": platform.platform(),
    }
