#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TILE = "CFIS.000.229"
TILES_URI = "vos:cfis/tiles_DR6"
PITCAIRN_URI = "vos:cfis/pitcairn"
EXPECTED: dict[str, dict[str, Any]] = {
    "u": {
        "reference": f"{TILE}.u.fits",
        "reference_approx_bytes": 400006080,
        "exposures": {
            "1833406p.fits.fz": 261334080,
            "1833429p.fits.fz": 259873920,
            "1833624p.fits.fz": 256386240,
            "2513912p.fits.fz": 270737280,
            "2513914p.fits.fz": 271008000,
            "2513915p.fits.fz": 271203840,
        },
    },
    "r": {
        "reference": f"{TILE}.r.fits",
        "reference_approx_bytes": 400006080,
        "exposures": {
            "2784453p.fits.fz": 343154880,
            "2784455p.fits.fz": 344868480,
            "2784458p.fits.fz": 348096960,
            "2785688p.fits.fz": 358908480,
            "2785690p.fits.fz": 356564160,
            "2785692p.fits.fz": 353177280,
        },
    },
}
EXPOSURE_RE = re.compile(
    r"(?<!\d)(\d{7})p?(?:\.fits(?:\.fz)?)?"
    r"(?=\s+\d+\s+[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
NAME_RE = re.compile(
    r"(?:vos:[^\s]+|[^\s/]+(?:\.fits(?:\.fz)?|\.expcxy))", re.IGNORECASE
)


class DiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ListingEntry:
    name: str
    uri: str
    size_bytes: int | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_listing(text: str, base_uri: str) -> list[ListingEntry]:
    entries: dict[str, ListingEntry] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        names = NAME_RE.findall(line)
        if not names:
            continue
        token = names[-1].rstrip("/")
        uri = token if token.startswith("vos:") else f"{base_uri.rstrip('/')}/{token}"
        name = uri.rsplit("/", 1)[-1]
        fields = line.split()
        size: int | None = None
        if len(fields) == 2 and fields[1].isdigit():
            size = int(fields[1])
        elif len(fields) >= 9 and fields[0][0] in "-dl" and fields[-5].isdigit():
            size = int(fields[-5])
        current = ListingEntry(name=name, uri=uri, size_bytes=size)
        previous = entries.get(name)
        if previous is not None and previous != current:
            raise DiscoveryError(f"conflicting listing rows for {name}")
        entries[name] = current
    if not entries:
        raise DiscoveryError(f"listing for {base_uri} contained no recognized files")
    return sorted(entries.values(), key=lambda item: item.name)


def extract_exposure_ids(text: str) -> list[str]:
    return sorted(
        {f"{match.group(1)}p.fits.fz" for match in EXPOSURE_RE.finditer(text)}
    )


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise DiscoveryError(f"required command is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise DiscoveryError(f"command failed ({' '.join(command)}): {detail}") from exc
    return completed.stdout


def _read_or_list(path: Path | None, uri: str, snapshot: Path) -> tuple[str, Path, str]:
    if path is not None:
        source = path.resolve(strict=True)
        return source.read_text(encoding="utf-8"), source, "saved"
    text = _run(["vls", "-l", uri])
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(text, encoding="utf-8")
    return text, snapshot, "vls"


def _find_exact(entries: list[ListingEntry], name: str, context: str) -> ListingEntry:
    matches = [entry for entry in entries if entry.name == name]
    if len(matches) != 1:
        raise DiscoveryError(
            f"{context}: expected exactly one {name}, found {len(matches)}"
        )
    return matches[0]


def _find_expcxy(entries: list[ListingEntry], band: str) -> ListingEntry:
    prefix = f"{TILE}.{band}"
    matches = [
        entry
        for entry in entries
        if entry.name.startswith(prefix) and entry.name.lower().endswith(".expcxy")
    ]
    if len(matches) != 1:
        raise DiscoveryError(
            f"tile listing: expected exactly one {prefix}*.expcxy, found {len(matches)}"
        )
    return matches[0]


def _under_scratch(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == Path("/scratch") or Path("/scratch") in resolved.parents


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _download(uri: str, destination: Path, expected_size: int | None) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    _run(["vcp", uri, str(destination)])
    if not destination.is_file() or destination.stat().st_size == 0:
        raise DiscoveryError(f"vcp did not create a non-empty file for {uri}")
    size = destination.stat().st_size
    if expected_size is not None and size != expected_size:
        raise DiscoveryError(
            f"download size mismatch for {uri}: listing={expected_size}, downloaded={size}"
        )
    return {"path": str(destination), "size_bytes": size, "sha256": sha256(destination)}


def _file_record(
    entry: ListingEntry,
    *,
    kind: str,
    band: str,
    local_name: str,
    approximate_size: int | None,
) -> dict[str, Any]:
    if entry.size_bytes is not None and approximate_size is not None:
        relative_error = abs(entry.size_bytes - approximate_size) / approximate_size
        if relative_error > 0.25:
            raise DiscoveryError(
                f"implausible size for {entry.uri}: listed={entry.size_bytes}, expected approximately={approximate_size}"
            )
    return {
        "kind": kind,
        "band": band,
        "tile": TILE,
        "name": entry.name,
        "uri": entry.uri,
        "remote_size_bytes": entry.size_bytes,
        "approximate_size_bytes": approximate_size,
        "local_name": local_name,
        "download": None,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    campaign_dir = Path(__file__).resolve().parent
    root = campaign_dir.parents[1]
    scratch_dir = args.scratch_dir.resolve()
    if args.fetch and not _under_scratch(scratch_dir):
        raise DiscoveryError("downloads require --scratch-dir under /scratch")
    metadata_dir = scratch_dir / "metadata"
    listing_specs = {
        "tiles": (args.tiles_listing, TILES_URI),
        "pitcairn_u": (args.pitcairn_u_listing, PITCAIRN_URI),
        "pitcairn_r": (args.pitcairn_r_listing, PITCAIRN_URI),
    }
    listings: dict[str, list[ListingEntry]] = {}
    listing_provenance: dict[str, Any] = {}
    for label, (path, uri) in listing_specs.items():
        text, source, method = _read_or_list(
            path, uri, metadata_dir / f"{label}.listing"
        )
        listings[label] = parse_listing(text, uri)
        listing_provenance[label] = {
            "uri": uri,
            "method": method,
            "source": str(source),
            "sha256": sha256(source),
        }

    records: list[dict[str, Any]] = []
    expcxy_provenance: dict[str, Any] = {}
    for band in ("u", "r"):
        expected = EXPECTED[band]
        reference = _find_exact(
            listings["tiles"], expected["reference"], f"{band} reference"
        )
        records.append(
            _file_record(
                reference,
                kind="reference",
                band=band,
                local_name=f"reference/{reference.name}",
                approximate_size=expected["reference_approx_bytes"],
            )
        )
        expcxy_entry = _find_expcxy(listings["tiles"], band)
        supplied = getattr(args, f"expcxy_{band}")
        if supplied is not None:
            expcxy_path = supplied.resolve(strict=True)
            expcxy_method = "saved"
        elif args.dry_run:
            expcxy_path = None
            expcxy_method = "planned-vcp"
        else:
            expcxy_path = metadata_dir / expcxy_entry.name
            _download(expcxy_entry.uri, expcxy_path, expcxy_entry.size_bytes)
            expcxy_method = "vcp"
        if expcxy_path is None:
            extracted: list[str] = []
            content_sha = None
        else:
            extracted = extract_exposure_ids(
                expcxy_path.read_text(encoding="utf-8", errors="replace")
            )
            content_sha = sha256(expcxy_path)
            expected_names = sorted(expected["exposures"])
            if extracted != expected_names:
                raise DiscoveryError(
                    f"{band} expcxy contributors differ from bounded set: expected={expected_names}, extracted={extracted}"
                )
        expcxy_provenance[band] = {
            "uri": expcxy_entry.uri,
            "method": expcxy_method,
            "source": str(expcxy_path) if expcxy_path else None,
            "sha256": content_sha,
            "extracted_exposures": extracted,
        }
        band_entries = listings[f"pitcairn_{band}"]
        for name, approximate_size in expected["exposures"].items():
            exposure = _find_exact(band_entries, name, f"{band} Pitcairn")
            records.append(
                _file_record(
                    exposure,
                    kind="exposure",
                    band=band,
                    local_name=f"inputs/{band}/{name}",
                    approximate_size=approximate_size,
                )
            )

    if args.fetch:
        for record in records:
            destination = scratch_dir / record["local_name"]
            record["download"] = _download(
                record["uri"], destination, record["remote_size_bytes"]
            )

    fully_verified = all(
        expcxy_provenance[band]["extracted_exposures"]
        == sorted(EXPECTED[band]["exposures"])
        for band in ("u", "r")
    )
    return {
        "schema_version": 1,
        "campaign": "cfht-union-pitcairn-bounded-smoke",
        "status": "verified" if fully_verified else "dry-run-planned",
        "tile": TILE,
        "bands": ["u", "r"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "outputs": {
            band: {
                "torchsky": f"products/{TILE}.{band}.torchsky.fits",
                "swarp": f"products/{TILE}.{band}.swarp.fits",
                "reference": f"products/reference/{EXPECTED[band]['reference']}",
                "result": f"results/{TILE}.{band}.smoke.json",
            }
            for band in ("u", "r")
        },
        "provenance": {
            "immutable_inputs": {
                "expected": EXPECTED,
                "listings": listing_provenance,
                "expcxy": expcxy_provenance,
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__)),
            },
            "harness_revision": _git_revision(root),
            "argv": sys.argv,
            "python": sys.version,
            "environment": {
                "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
                "PYTHONPATH_set": bool(os.environ.get("PYTHONPATH")),
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover and optionally fetch the bounded CFIS.000.229 Union/Pitcairn u/r smoke inputs."
    )
    parser.add_argument("--tiles-listing", type=Path)
    parser.add_argument("--pitcairn-u-listing", type=Path)
    parser.add_argument("--pitcairn-r-listing", type=Path)
    parser.add_argument("--expcxy-u", type=Path)
    parser.add_argument("--expcxy-r", type=Path)
    parser.add_argument(
        "--scratch-dir", type=Path, default=Path("/scratch/cfht-union-pitcairn")
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fetch and args.dry_run:
        raise DiscoveryError("--fetch and --dry-run are mutually exclusive")
    payload = build_manifest(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
