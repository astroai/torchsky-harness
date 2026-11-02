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
    "starlink-pyast",
    "drizzlepac",
    "numpy",
    "torch",
    "torchfits",
)


_OUT_OF_PROCESS = {"starlink-pyast": "PyAST", "drizzlepac": "AstroDrizzle"}


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        if name in _OUT_OF_PROCESS:
            return f"not in the run venv (out-of-process oracle: see {_OUT_OF_PROCESS[name]} below)"
        return "NOT INSTALLED"


def _oracle_versions() -> dict:
    """Versions of the out-of-process oracles (their interpreters come from the campaign env)."""
    import os

    out = {}
    probes = {
        "PyAST": ("TORCHSKY_HARNESS_ORACLE_PYAST_PYTHON", "import starlink.Ast as A; v=A.version(); print('starlink-pyast %d.%d.%d' % (v//1000000, v//1000%1000, v%1000))"),
        "AstroDrizzle": ("TORCHSKY_HARNESS_ORACLE_DRIZZLEPAC_PYTHON", "import drizzlepac; print('drizzlepac', drizzlepac.__version__)"),
        "Kapteyn": ("TORCHSKY_HARNESS_ORACLE_KAPTEYN_PYTHON", "import kapteyn; print('kapteyn', kapteyn.__version__)"),
    }
    for name, (var, code) in probes.items():
        py = os.environ.get(var)
        if not py:
            reason = os.environ.get(var.replace("_PYTHON", "_UNPROVISIONED"))
            if reason:
                out[name] = f"NOT PROVISIONED: {reason}"
            elif name == "Kapteyn":
                out[name] = "not provisioned (no source)"
            else:
                out[name] = "NOT PROVISIONED"
            continue
        try:
            out[name] = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120).stdout.strip() or "NOT PROVISIONED"
        except Exception as exc:  # noqa: BLE001
            out[name] = f"NOT PROVISIONED ({exc!r})"
    try:
        out["STILTS"] = subprocess.run(["stilts", "-version"], capture_output=True, text=True, timeout=120).stdout.split("STILTS version")[-1].split()[0]
    except Exception:  # noqa: BLE001
        out["STILTS"] = "NOT AVAILABLE"
    return out


def _perf_rows(out: Path) -> list[dict]:
    """Normalized comparator-performance rows: bench JSONs plus the parity gates' jsonl records."""
    rows: list[dict] = []
    roots = [out / "benchmark_results", Path(__file__).resolve().parents[2] / "benchmark_results"]
    seen = set()
    for root in roots:
        if not root.is_dir() or root.resolve() in seen:
            continue
        seen.add(root.resolve())
        for path in sorted(root.glob("comparator_perf*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            for row in payload.get("rows", []):
                rows.append(dict(row, source=path.name))
        for path in sorted(root.glob("comparator_perf*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(dict(json.loads(line), source=path.name))
    return rows


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds * 1e3:.1f} ms" if seconds < 1.0 else f"{seconds:.2f} s"


def _perf_table(rows: list[dict]) -> list[str]:
    lines = [
        "## Comparator performance",
        "",
        "Recorded, not gated: speedup = comparator time / torchsky time (above 1: torchsky faster); throughput = size / torchsky time. STILTS wall time includes JVM start and table I/O; PyAST and Kapteyn are timed inside their oracle process. Sizes of 1 are single calls.",
        "",
        "| area | operation | size | device | torchsky | throughput | comparator | comparator time | speedup |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if not rows:
        return lines[:-2] + ["No comparator performance rows were produced (the `bench-comparator-perf-gate` gate failed or was not run).", ""]
    for r in sorted(rows, key=lambda r: (r.get("area", ""), r.get("operation", ""), r.get("size", 0), r.get("device", ""), r.get("comparator", ""))):
        if r.get("operation") in {"error", "truncated"}:
            lines.append(f"| {r.get('area')} | {str(r.get('operation')).upper()} | - | {r.get('device')} | - | - | - | {r.get('status')} | - |")
            continue
        thr = r.get("throughput_per_s")
        sp = r.get("speedup")
        cmp_time = _fmt_time(r.get("comparator_s")) if r.get("comparator_s") is not None else f"unavailable ({r.get('note', '')})"
        lines.append(
            f"| {r['area']} | {r['operation']} | {r['size']:,} | {r.get('device', '')} | {_fmt_time(r['torchsky_s'])} | "
            f"{(f'{thr:,.3g}/s' if thr else '-')} | {r['comparator']} | {cmp_time} | {(f'{sp:.2f}x' if sp is not None else '-')} |"
        )
    lines.append("")
    return lines


def _real_catalogs(out: Path) -> dict | None:
    import os

    root = os.environ.get("CATALOG_REAL_DIR") or str(out / "real-catalogs")
    path = Path(root) / "manifest.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
    return None


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
    oracles = _oracle_versions()
    perf = _perf_rows(args.out)
    real = _real_catalogs(args.out)
    gpu = _gpu() if args.mode == "gpu" else {"cuda_available": False}
    payload = {
        "run": args.run,
        "mode": args.mode,
        "subject": subject,
        "comparators": comparators,
        "oracles": oracles,
        "real_catalogs": real,
        "comparator_performance": perf,
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
    lines += [f"**{passed}/{len(rows)} gates passed, {skipped} skipped (opt-in data or oracle), {failed} failed**", ""]
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
        note = f" - {row['reason']}" if row.get("reason") else ""
        lines.append(f"| {row['suite']} | `{row['gate']}` | {row['status']} (rc={row['returncode']}){note} | {row['seconds']} |")
    lines += ["", "## Comparators", "", "| package | version |", "|---|---|"]
    lines += [f"| {name} | {version} |" for name, version in comparators.items()]
    lines += ["", "## Out-of-process oracles and binaries", "", "| comparator | version |", "|---|---|"]
    lines += [f"| {name} | {version} |" for name, version in oracles.items()]
    lines += ["", "## Real catalogues (STILTS gate)", ""]
    if real:
        lines += [f"Field: `{json.dumps(real.get('field'))}`, fetched {real.get('fetched_utc')}", "", "| table | rows | service | sha256 |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['rows']:,} | {v['service']} | `{v['sha256'][:16]}` |" for k, v in real.get("tables", {}).items()]
        lines += ["", "Queries: " + "; ".join(f"`{v['query']}`" for v in real.get("tables", {}).values()), ""]
    else:
        lines += ["NOT AVAILABLE: real catalogues were not fetched or staged (the real-data STILTS gate fails closed under `REQUIRE_TOOLS`).", ""]
    lines += _perf_table(perf)
    lines += ["", "## Hardware", "", "```json", json.dumps(gpu, indent=2, default=str), "```", ""]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((args.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
