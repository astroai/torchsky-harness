#!/usr/bin/env python
"""Fetch two real catalogues of one field for the STILTS crossmatch gate.

Gaia DR3 (ESA archive TAP) and 2MASS point sources (IRSA TAP ``fp_psc``) in one cone, queried
over plain HTTP TAP (``requests`` only), written as FITS binary tables with ``torchfits`` plus a
``manifest.json`` (queries, service URLs, row counts, sha256, UTC time). The fetch is cached: an
existing manifest in ``--out`` is reused unchanged. ``CATALOG_REAL_DIR`` may point at a pre-staged
directory with the same layout (``gaia.fits``, ``twomass.fits``, ``manifest.json``).

    python scripts/fetch_real_catalogs.py --out "$OUT/real-catalogs"

Exit status is non-zero when a catalogue cannot be fetched (fail closed).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import requests
import torch
import torchfits

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
# Baade-window bulge field: ~2e5 Gaia sources and heavy crowding (best, best1 and all differ).
DEFAULT_FIELD = (270.92, -30.03, 0.25)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def tap_csv(url: str, query: str, maxrec: int = 10_000_000) -> list[dict]:
    """Async job with a sync fallback; each retried (the archives reset connections now and then)."""
    last: Exception | None = None
    for attempt in range(3):
        for runner in (_tap_async, _tap_sync):
            try:
                return runner(url, query, maxrec)
            except Exception as exc:  # noqa: BLE001
                last = exc
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"TAP query failed after retries: {last!r}")


def _tap_sync(url: str, query: str, maxrec: int) -> list[dict]:
    reply = requests.post(
        url,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query, "MAXREC": str(maxrec)},
        timeout=1800,
    )
    reply.raise_for_status()
    return list(csv.DictReader(io.StringIO(reply.text)))


def _tap_async(url: str, query: str, maxrec: int) -> list[dict]:
    """Run ``query`` as an async TAP job (no sync row cap) and return the CSV rows."""
    base = url.rsplit("/sync", 1)[0] + "/async"
    reply = requests.post(
        base,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query, "PHASE": "RUN", "MAXREC": str(maxrec)},
        timeout=300,
        allow_redirects=False,
    )
    job = reply.headers.get("Location") or reply.url
    if reply.status_code >= 400 or not job:
        reply.raise_for_status()
    job = job.replace("http://", "https://")
    deadline = time.time() + 3600
    while True:
        phase = requests.get(job + "/phase", timeout=120).text.strip()
        if phase == "COMPLETED":
            break
        if phase in {"ERROR", "ABORTED"}:
            raise RuntimeError(f"TAP job {job} ended {phase}: {requests.get(job, timeout=120).text[-500:]}")
        if time.time() > deadline:
            raise TimeoutError(f"TAP job {job} still {phase}")
        time.sleep(3)
    result = requests.get(job + "/results/result", timeout=1800)
    result.raise_for_status()
    return list(csv.DictReader(io.StringIO(result.text)))


def tap_count(url: str, query: str) -> int:
    count_query = "SELECT COUNT(*) AS n FROM " + query.split(" FROM ", 1)[1]
    return int(tap_csv(url, count_query)[0]["n"])


def write_table(path: Path, rows: list[dict], id_key: str, ra_key: str, dec_key: str) -> int:
    keep = [r for r in rows if r[ra_key] not in ("", None) and r[dec_key] not in ("", None)]
    ra = np.array([float(r[ra_key]) for r in keep], dtype=np.float64)
    dec = np.array([float(r[dec_key]) for r in keep], dtype=np.float64)
    ident = np.arange(len(keep), dtype=np.int64)
    torchfits.table.write(
        str(path),
        {"ra": torch.from_numpy(ra), "dec": torch.from_numpy(dec), "id": torch.from_numpy(ident)},
        overwrite=True,
    )
    return len(keep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ra", type=float, default=DEFAULT_FIELD[0])
    parser.add_argument("--dec", type=float, default=DEFAULT_FIELD[1])
    parser.add_argument("--radius", type=float, default=DEFAULT_FIELD[2], help="cone radius, degrees")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.is_file():
        print(f"cached: {manifest_path}")
        return 0

    cone = f"1=CONTAINS(POINT('ICRS',{{ra}},{{dec}}),CIRCLE('ICRS',{args.ra},{args.dec},{args.radius}))"
    queries = {
        "gaia": (
            GAIA_TAP,
            "SELECT source_id, ra, dec FROM gaiadr3.gaia_source WHERE " + cone.format(ra="ra", dec="dec"),
            "source_id",
        ),
        "twomass": (
            IRSA_TAP,
            "SELECT designation, ra, dec FROM fp_psc WHERE " + cone.format(ra="ra", dec="dec"),
            "designation",
        ),
    }
    manifest: dict = {
        "field": {"ra": args.ra, "dec": args.dec, "radius_deg": args.radius},
        "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tables": {},
    }
    for name, (url, query, id_key) in queries.items():
        try:
            expected = tap_count(url, query)
            rows = tap_csv(url, query, maxrec=expected + 1000)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED to fetch {name} from {url}: {exc!r}", file=sys.stderr)
            return 1
        if expected != len(rows):
            print(f"{name}: fetched {len(rows)} rows but COUNT(*) is {expected} (truncated)", file=sys.stderr)
            return 1
        path = args.out / f"{name}.fits"
        n = write_table(path, rows, id_key, "ra", "dec")
        manifest["tables"][name] = {
            "service": url,
            "query": query,
            "rows": n,
            "file": path.name,
            "sha256": sha256(path),
        }
        print(f"{name}: {n} rows from {url}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
