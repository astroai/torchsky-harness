"""Write summary.md / summary.json for a CANFAR harness campaign.

    python summarize.py --out DIR --run RUN --mode cpu|gpu
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

COMPARATORS = (
    "astropy",
    "healpy",
    "hpgeom",
    "astropy-healpix",
    "mocpy",
    "spherical-geometry",
    "healsparse",
    "reproject",
    "drizzle",
    "dfreproject",
    "dust-extinction",
    "speclite",
    "synphot",
    "scipy",
    "pyerfa",
    "MontagePy",
    "gwcs",
    "sgp4",
    "jplephem",
    "mhealpy",
    "cdshealpix",
    "specutils",
    "torch-harmonics",
    "s2fft",
    "numpy",
    "torch",
    "torchfits",
)


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _gpu() -> dict:
    info: dict = {"nvidia_smi": None, "cuda_available": False}
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda"] = torch.version.cuda
        if info["cuda_available"]:
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    rows = []
    ledger = args.out / "gates.jsonl"
    if ledger.is_file():
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        from torchsky_harness.subject import identify_subject

        subject = identify_subject()
    except Exception as exc:  # noqa: BLE001
        subject = {"error": repr(exc)}
    comparators = {name: _version(name) for name in COMPARATORS}
    gpu = _gpu() if args.mode == "gpu" else {"cuda_available": False}
    payload = {
        "run": args.run,
        "mode": args.mode,
        "subject": subject,
        "comparators": comparators,
        "gpu": gpu,
        "python": sys.version,
        "platform": platform.platform(),
        "gates": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    passed = sum(r["status"] == "pass" for r in rows)
    lines = [f"# Harness campaign `{args.run}` ({args.mode})", ""]
    skipped = sum(r["status"] == "skip" for r in rows)
    failed = sum(r["status"] == "fail" for r in rows)
    lines += [f"**{passed}/{len(rows)} gates passed, {skipped} skipped (opt-in data), {failed} failed**", ""]
    lines += ["## Subject", ""]
    record = subject.get("install_record") or {}
    lines += [
        f"- torchsky `{subject.get('version')}` imported from `{subject.get('import_path')}`",
        f"- source: `{json.dumps(subject.get('source'), default=str)}`",
        f"- install record: `{json.dumps(record, default=str)}`",
        "",
        "## Gates",
        "",
        "| suite | gate | status | seconds |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['suite']} | `{row['gate']}` | {row['status']} (rc={row['returncode']}) | {row['seconds']} |")
    lines += ["", "## Comparators", "", "| package | version |", "|---|---|"]
    lines += [f"| {name} | {version} |" for name, version in comparators.items()]
    lines += ["", "## Hardware", "", "```json", json.dumps(gpu, indent=2, default=str), "```", ""]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((args.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
