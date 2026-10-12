"""Fetch artifacts from the frozen pre-extraction SMACS campaign."""
import argparse
from pathlib import Path, PurePosixPath
import ssl
import urllib.request

LOCAL = Path(__file__).resolve().parent
ARC = "/projects/ots/torchsky-runs/smacs-inr-autonomous-20261010"
SERVICE = "https://ws-uv.canfar.net/arc"

def request(path, *, data=None, method="GET", content_type=None):
    context = ssl.create_default_context()
    context.load_cert_chain(Path.home() / ".ssl" / "cadcproxy.pem")
    headers = {"Content-Type": content_type} if content_type else {}
    req = urllib.request.Request(
        SERVICE + path, data=data, method=method, headers=headers
    )
    with urllib.request.urlopen(req, context=context, timeout=120) as response:
        return response.read()

def fetch(run_id, relative):
    if not run_id.replace("-", "").isalnum() or len(run_id) > 64:
        raise ValueError("invalid run id")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in relative:
        raise ValueError("artifact must be a relative path within the run")
    run_root = LOCAL.resolve() / "runs" / run_id
    local = run_root / relative
    if not local.resolve().is_relative_to(run_root):
        raise ValueError("local artifact path escapes the run through a symlink")
    payload = request("/files" + ARC + "/" + run_id + "/" + relative)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(payload)
    print(str(local))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("artifact", help="Relative path within the frozen remote run")
    args = parser.parse_args()
    fetch(args.run_id, args.artifact)
