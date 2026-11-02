#!/usr/bin/env python
"""Fetch three real catalogues of one field for the STILTS crossmatch gate.

Gaia DR3 (ESA archive TAP), 2MASS point sources (IRSA TAP ``fp_psc``) and the AllWISE source
catalogue (IRSA TAP ``allwise_p3as_psd``) in one cone, queried over plain HTTP TAP (``requests``
only), written as FITS binary tables with ``torchfits`` plus a ``manifest.json`` (queries, service
URLs, row counts, columns, sha256, UTC time). Each table carries ``ra``, ``dec``, a 0-based ``id``
and the catalogue's positional errors (Gaia ``ra_error``/``dec_error`` in mas, 2MASS ``err_maj`` in
arcsec, AllWISE ``sigra``/``sigdec`` in arcsec; missing values are NaN). The fetch is cached per
table: a table whose manifest entry, file, sha256 and column list all match is reused, anything
else is fetched again (and each fetch is checked against ``COUNT(*)``). ``CATALOG_REAL_DIR`` may
point at a pre-staged directory with the same layout (``gaia.fits``, ``twomass.fits``,
``allwise.fits``, ``manifest.json``).

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


def write_table(path: Path, rows: list[dict], extra: tuple[str, ...]) -> int:
    keep = [r for r in rows if r["ra"] not in ("", None) and r["dec"] not in ("", None)]

    def col(key: str) -> torch.Tensor:
        vals = [float(r[key]) if r[key] not in ("", None) else np.nan for r in keep]
        return torch.from_numpy(np.array(vals, dtype=np.float64))

    data = {"ra": col("ra"), "dec": col("dec"), "id": torch.arange(len(keep), dtype=torch.int64)}
    data.update({k: col(k) for k in extra})
    torchfits.table.write(str(path), data, overwrite=True)
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
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    field = {"ra": args.ra, "dec": args.dec, "radius_deg": args.radius}
    if manifest.get("field", field) != field:
        manifest = {}
    manifest.setdefault("field", field)
    manifest.setdefault("tables", {})
    manifest.setdefault("fetched_utc", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))

    cone = f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{args.ra},{args.dec},{args.radius}))"
    # name -> (service, table, error columns)
    sources = {
        "gaia": (GAIA_TAP, "gaiadr3.gaia_source", ("ra_error", "dec_error")),
        "twomass": (IRSA_TAP, "fp_psc", ("err_maj",)),
        "allwise": (IRSA_TAP, "allwise_p3as_psd", ("sigra", "sigdec")),
    }
    for name, (url, table, extra) in sources.items():
        query = f"SELECT ra, dec, {', '.join(extra)} FROM {table} WHERE {cone}"
        path = args.out / f"{name}.fits"
        entry = manifest["tables"].get(name)
        columns = ["ra", "dec", "id", *extra]
        if entry and path.is_file() and entry.get("sha256") == sha256(path) and entry.get("columns") == columns:
            print(f"cached: {name} ({entry['rows']} rows)")
            continue
        try:
            expected = tap_count(url, query)
            rows = tap_csv(url, query, maxrec=expected + 1000)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED to fetch {name} from {url}: {exc!r}", file=sys.stderr)
            return 1
        if expected != len(rows):
            print(f"{name}: fetched {len(rows)} rows but COUNT(*) is {expected} (truncated)", file=sys.stderr)
            return 1
        n = write_table(path, rows, extra)
        manifest["tables"][name] = {
            "service": url,
            "query": query,
            "rows": n,
            "columns": columns,
            "file": path.name,
            "sha256": sha256(path),
        }
        print(f"{name}: {n} rows from {url}")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
