"""Public archive metadata inventory; never fetches science files.

MAST observations are paged to completion within an explicit safety cap. Product
queries are a bounded lineage sample, independent of observation completeness.
CADC uses TAP VOTable QUERY_STATUS so overflow cannot masquerade as completeness.
SMOKA supplies an actionable interactive query recipe, not an invented API.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

MAST_URL = "https://mast.stsci.edu/api/v0/invoke"
CADC_URL = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"
SOURCES = {
    "mast_api": "https://mast.stsci.edu/api/v0/MastApiTutorial.html",
    "mast_products": "https://mast.stsci.edu/api/v0/_productsfields.html",
    "hst_products": "https://hst-docs.stsci.edu/acsdhb/chapter-2-acs-data-structure/2-1-types-of-acs-files",
    "jwst_products": "https://jwst-docs.stsci.edu/accessing-jwst-data/jwst-science-data-overview",
    "cadc_tap": "https://www.cadc.hia.nrc-cnrc.gc.ca/en/doc/tap/",
    "cadc_caom": "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/doc/caom2/",
    "smoka_help": "https://smoka.nao.ac.jp/help/help.jsp",
    "smoka_hsc": "https://smoka.nao.ac.jp/HSCsearch.jsp",
    "smoka_sup": "https://smoka.nao.ac.jp/SUPsearch.jsp",
}


class ArchiveQueryError(RuntimeError):
    """Malformed, changing, incomplete-protocol, or error archive response."""

    def __init__(self, message: str, provenance: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.provenance = provenance or []


@dataclass(frozen=True)
class Cone:
    ra_deg: float = 150.116321
    dec_deg: float = 2.20583
    radius_deg: float = 0.03

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("cone coordinates must be finite degrees")
        if not 0 <= self.ra_deg < 360 or not -90 <= self.dec_deg <= 90:
            raise ValueError("RA must be [0, 360), Dec [-90, 90] degrees")
        if not 0 < self.radius_deg <= 180:
            raise ValueError("radius must be (0, 180] degrees")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _response_record(response: Any, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_at_utc": request.pop("requested_at_utc"),
        "received_at_utc": utc_now(),
        "endpoint": response.url,
        "request": request,
        "http_status": response.status_code,
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
    }


def _deduplicate(
    rows: list[dict[str, Any]], key: str
) -> tuple[list[dict[str, Any]], int]:
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        identity = row.get(key)
        if identity in (None, ""):
            raise ArchiveQueryError(f"missing {key} identity")
        token = str(identity)
        if key == "dataURI":
            # One auxiliary artifact may occur under several observation and
            # association parents. Those are distinct lineage rows, not a
            # conflict in the underlying science-file identity.
            token = json.dumps([identity, row.get("obsID"), row.get("parent_obsid")])
        if token in unique:
            duplicates += 1
            if unique[token] != row:
                raise ArchiveQueryError(f"conflicting duplicate {key}: {token}")
        else:
            unique[token] = row
    return list(unique.values()), duplicates


def _checked_post(session: Any, endpoint: str, data: dict[str, Any], timeout: float):
    started = utc_now()
    attempted = {"endpoint": endpoint, "request": data, "requested_at_utc": started}
    try:
        response = session.post(endpoint, data=data, timeout=timeout)
    except OSError as exc:
        attempted["received_at_utc"] = utc_now()
        raise ArchiveQueryError(f"archive transport error: {exc}", [attempted]) from exc
    record = _response_record(response, {"requested_at_utc": started, **data})
    if response.status_code >= 400:
        raise ArchiveQueryError(
            f"archive HTTP {response.status_code}: {response.text[:2000]}", [record]
        )
    return response, record


def mast_observation_request(cone: Cone, page_size: int) -> dict[str, Any]:
    _positive(page_size, "page_size")
    return {
        "service": "Mast.Caom.Filtered.Position",
        "params": {
            "position": f"{cone.ra_deg}, {cone.dec_deg}, {cone.radius_deg}",
            "columns": "*",
            "filters": [
                {"paramName": "obs_collection", "values": ["HST", "JWST"]},
                {"paramName": "dataproduct_type", "values": ["image"]},
                {"paramName": "dataRights", "values": ["PUBLIC"]},
            ],
        },
        "format": "json",
        "pagesize": page_size,
        "page": 1,
    }


def query_mast_pages(
    session: Any,
    request: dict[str, Any],
    *,
    identity: str,
    max_pages: int = 100,
    timeout: float = 50,
) -> dict[str, Any]:
    _positive(max_pages, "max_pages")
    records, rows = [], []
    expected = None
    total_pages = None
    for page in range(1, max_pages + 1):
        query = deepcopy(request)
        query["page"] = page
        response, record = _checked_post(
            session, MAST_URL, {"request": json.dumps(query)}, timeout
        )
        record["request"] = {"mast_request": query}
        records.append(record)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ArchiveQueryError("MAST did not return JSON", records) from exc
        if not isinstance(payload, dict) or payload.get("status") != "COMPLETE":
            raise ArchiveQueryError("MAST query is not COMPLETE", records)
        data, paging = payload.get("data"), payload.get("paging")
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise ArchiveQueryError("invalid MAST data rows", records)
        if not isinstance(paging, dict):
            raise ArchiveQueryError("MAST paging metadata missing", records)
        for name in ("page", "pageSize", "pagesFiltered", "rows", "rowsFiltered"):
            value = paging.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ArchiveQueryError(f"invalid MAST paging {name}", records)
        if (
            paging["page"] != page
            or paging["pageSize"] != query["pagesize"]
            or paging["rows"] != len(data)
            or len(data) > query["pagesize"]
        ):
            raise ArchiveQueryError("MAST paging does not match returned page", records)
        count, pages = paging["rowsFiltered"], paging["pagesFiltered"]
        if pages != math.ceil(count / query["pagesize"]):
            raise ArchiveQueryError("MAST page and row totals disagree", records)
        if expected is not None and (count != expected or pages != total_pages):
            raise ArchiveQueryError(
                "MAST result changed while paging; rerun query", records
            )
        expected, total_pages = count, pages
        record["paging"] = paging
        rows.extend(data)
        if page >= pages:
            if len(rows) != count:
                raise ArchiveQueryError(
                    "MAST fetched row count differs from total", records
                )
            break
        if not data:
            raise ArchiveQueryError("empty intermediate MAST page", records)
    try:
        unique, duplicates = _deduplicate(rows, identity)
    except ArchiveQueryError as exc:
        raise ArchiveQueryError(str(exc), records) from exc
    complete = len(records) >= total_pages
    return {
        "rows": unique,
        "raw_row_count": len(rows),
        "unique_row_count": len(unique),
        "duplicate_count": duplicates,
        "reported_total": expected,
        "complete": complete and duplicates == 0,
        "truncated": not complete,
        "completeness_reason": (
            "duplicate identities reduce coverage"
            if duplicates
            else "all reported pages fetched"
            if complete
            else "max_pages reached"
        ),
        "queries": records,
    }


def classify_mast_product(row: dict[str, Any]) -> dict[str, Any]:
    """Only known exposure formats are candidates, never validated training data."""
    mission = str(row.get("obs_collection", "")).upper()
    name = str(row.get("productFilename", "")).lower()
    rights = str(row.get("dataRights", "")).upper()
    suffix = name.rsplit("_", 1)[-1].removesuffix(".fits")
    classification, reason = "unknown", "unrecognized or insufficient metadata"
    if rights != "PUBLIC":
        classification, reason = "ineligible", "product rights not explicitly PUBLIC"
    elif str(row.get("productType", "")).upper() != "SCIENCE":
        classification, reason = "auxiliary", "not a SCIENCE product"
    elif not name.endswith(".fits"):
        classification, reason = "auxiliary", "not a FITS science candidate"
    elif suffix in {"drz", "drc", "crj", "crc", "sfl", "i2d", "s3d"}:
        classification, reason = "derived_reference", "resampled or combined product"
    elif mission == "HST" and re.fullmatch(r"[a-z0-9]{9}_(flt|flc|raw)\.fits", name):
        classification, reason = "native_exposure_candidate", "HST exposure format"
    elif mission == "JWST" and re.fullmatch(
        r"jw\d{11}_\d{5}_\d{5}_[a-z0-9]+(?:_seg\d{3})?_(uncal|rate|rateints|cal|calints)\.fits",
        name,
    ):
        classification, reason = "native_exposure_candidate", "JWST exposure filename"
    if classification == "native_exposure_candidate" and row.get("type") in {"C", "D"}:
        # MAST parent association type can describe even an exposure child. Keep
        # that ambiguity explicit rather than claiming verified native lineage.
        reason += "; association/derived type requires header and association review"
    return {
        "classification": classification,
        "reason": reason,
        "verified_native": False,
    }


def query_mast(
    session: Any,
    cone: Cone,
    *,
    page_size: int = 500,
    max_pages: int = 100,
    product_observations_per_collection: int = 2,
    timeout: float = 50,
) -> dict[str, Any]:
    if product_observations_per_collection < 0:
        raise ValueError("product_observations_per_collection must be nonnegative")
    observations = query_mast_pages(
        session,
        mast_observation_request(cone, page_size),
        identity="obsid",
        max_pages=max_pages,
        timeout=timeout,
    )
    by_mission: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations["rows"]:
        if str(row.get("dataRights", "")).upper() != "PUBLIC":
            raise ArchiveQueryError(
                "MAST observation lacks explicitly public rights",
                observations["queries"],
            )
        by_mission[str(row.get("obs_collection"))].append(row)
    products, sampled_ids = [], []
    for mission in ("HST", "JWST"):
        selection = sorted(by_mission[mission], key=lambda row: str(row["obsid"]))
        for row in selection[:product_observations_per_collection]:
            sampled_ids.append(str(row["obsid"]))
            request = {
                "service": "Mast.Caom.Products",
                "params": {"obsid": str(row["obsid"])},
                "format": "json",
                "pagesize": page_size,
                "page": 1,
            }
            try:
                result = query_mast_pages(
                    session,
                    request,
                    identity="dataURI",
                    max_pages=min(max_pages, 4),
                    timeout=timeout,
                )
            except (ArchiveQueryError, ValueError, OSError) as exc:
                products.append(
                    {
                        "queried_observation": row["obsid"],
                        "status": "error",
                        "complete": False,
                        "error": str(exc),
                        "request": request,
                        "endpoint": MAST_URL,
                        "received_at_utc": utc_now(),
                        "rows": [],
                        "queries": getattr(exc, "provenance", []),
                    }
                )
                continue
            result["queried_observation"] = row["obsid"]
            for product in result["rows"]:
                product["candidate_assessment"] = classify_mast_product(product)
            products.append(result)
    product_uris = [p["dataURI"] for query in products for p in query["rows"]]
    return {
        "observations": observations,
        "collection_counts": dict(
            Counter(r["obs_collection"] for r in observations["rows"])
        ),
        "instrument_counts": dict(
            Counter(r.get("instrument_name") for r in observations["rows"])
        ),
        "proposal_counts": dict(
            Counter(str(r.get("proposal_id")) for r in observations["rows"])
        ),
        "product_queries": products,
        "sampled_product_observation_ids": sampled_ids,
        "unique_product_uri_count": len(set(product_uris)),
        "repeated_product_uri_count_across_queries": len(product_uris)
        - len(set(product_uris)),
        "product_lineage_complete": (
            len(sampled_ids) == len(observations["rows"])
            and observations["complete"]
            and all(p["complete"] for p in products)
        ),
        "product_scope": "bounded per-collection sample sorted by obsid; not a complete exposure census",
        "lineage_fields": [
            "obs_id",
            "obsID",
            "parent_obsid",
            "type",
            "proposal_id",
            "prvversion",
        ],
    }


def cadc_query(cone: Cone, public_at_utc: str) -> str:
    moment = datetime.fromisoformat(public_at_utc)
    if moment.tzinfo is None:
        raise ValueError("public_at_utc needs a timezone")
    timestamp = moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"""SELECT o.observationID, o.obsID, o.collection, o.instrument_name,
o.algorithm_name, o.intent, o.type, o.proposal_id, o.target_name,
p.planeID, p.productID, p.calibrationLevel, p.dataProductType, p.dataRelease,
p.dataReadGroups, p.position_bounds, p.time_bounds_lower, p.time_bounds_upper,
p.time_exposure, p.energy_bandpassName, p.provenance_name,
p.provenance_version, p.provenance_inputs
FROM caom2.Observation AS o JOIN caom2.Plane AS p ON o.obsID = p.obsID
WHERE o.collection = 'CFHT' AND o.instrument_name = 'MegaPrime'
AND INTERSECTS(p.position_bounds, CIRCLE('ICRS', {cone.ra_deg}, {cone.dec_deg}, {cone.radius_deg})) = 1
AND p.dataRelease <= '{timestamp}'
ORDER BY o.observationID, p.productID"""


def parse_tap_votable(text: str) -> tuple[list[dict[str, Any]], bool]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ArchiveQueryError("CADC did not return well-formed VOTable") from exc

    def local(node: ET.Element) -> str:
        return node.tag.rsplit("}", 1)[-1]

    if local(root) != "VOTABLE":
        raise ArchiveQueryError("CADC response is not a VOTABLE")
    statuses = [
        n.attrib.get("value")
        for n in root.iter()
        if local(n) == "INFO" and n.attrib.get("name") == "QUERY_STATUS"
    ]
    if (
        not statuses
        or statuses[0] != "OK"
        or any(s not in {"OK", "OVERFLOW"} for s in statuses)
    ):
        raise ArchiveQueryError(f"CADC QUERY_STATUS invalid or error: {statuses}")
    tables = [n for n in root.iter() if local(n) == "TABLE"]
    if len(tables) != 1:
        raise ArchiveQueryError("expected exactly one CADC result table")
    fields = [n.attrib.get("name") for n in tables[0] if local(n) == "FIELD"]
    if not fields or any(not f for f in fields) or len(set(fields)) != len(fields):
        raise ArchiveQueryError("CADC columns missing or duplicated")
    if not any(local(n) == "TABLEDATA" for n in tables[0].iter()):
        raise ArchiveQueryError("CADC VOTable TABLEDATA serialization required")
    rows = []
    for tr in tables[0].iter():
        if local(tr) != "TR":
            continue
        cells = [(n.text or "").strip() for n in tr if local(n) == "TD"]
        if len(cells) != len(fields):
            raise ArchiveQueryError("CADC row length differs from columns")
        rows.append(dict(zip(fields, cells, strict=True)))
    return rows, "OVERFLOW" in statuses


def classify_cadc_plane(row: dict[str, Any]) -> dict[str, Any]:
    level = int(row["calibrationLevel"])
    classification = "unknown"
    if row.get("intent") == "calibration":
        classification = "calibration_evidence_candidate"
    elif level >= 3:
        classification = "derived_reference"
    elif (
        row.get("algorithm_name") == "exposure"
        and level in {1, 2}
        and row.get("intent") == "science"
        and row.get("dataProductType") == "image"
    ):
        classification = "native_exposure_candidate"
    return {
        "classification": classification,
        "verified_native": False,
        "reason": "intent, product type, calibration level and algorithm only; verify headers and provenance",
    }


def query_cadc(
    session: Any,
    cone: Cone,
    *,
    max_records: int = 20000,
    timeout: float = 50,
) -> dict[str, Any]:
    _positive(max_records, "max_records")
    started = utc_now()
    request = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "votable",
        "MAXREC": str(max_records),
        "QUERY": cadc_query(cone, started),
    }
    response, record = _checked_post(session, CADC_URL, request, timeout)
    record["request"] = {"tap_request": request}
    try:
        rows, overflow = parse_tap_votable(response.text)
    except ArchiveQueryError as exc:
        raise ArchiveQueryError(str(exc), [record]) from exc
    try:
        unique, duplicates = _deduplicate(rows, "planeID")
    except ArchiveQueryError as exc:
        raise ArchiveQueryError(str(exc), [record]) from exc
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        try:
            level = int(row["calibrationLevel"])
            released = datetime.fromisoformat(row["dataRelease"].replace("Z", "+00:00"))
            if released.tzinfo is None:
                released = released.replace(tzinfo=UTC)
        except (KeyError, ValueError) as exc:
            raise ArchiveQueryError(
                "CADC missing calibration level or release time", [record]
            ) from exc
        if released > datetime.fromisoformat(started):
            raise ArchiveQueryError("CADC plane is not public at query time", [record])
        if row.get("collection") != "CFHT" or row.get("instrument_name") != "MegaPrime":
            raise ArchiveQueryError(
                "CADC returned unexpected collection/instrument", [record]
            )
        if level not in {0, 1, 2, 3, 4} or not row.get("observationID"):
            raise ArchiveQueryError(
                "CADC invalid level or missing observation identity", [record]
            )
        row["candidate_assessment"] = classify_cadc_plane(row)
        grouped[row["observationID"]].append(row)
    return {
        "rows": unique,
        "raw_row_count": len(rows),
        "duplicate_count": duplicates,
        "complete": not overflow and len(rows) < max_records and not duplicates,
        "truncated": overflow or len(rows) >= max_records,
        "query_status_overflow": overflow,
        "queries": [record],
        "calibration_level_counts": dict(
            Counter(r["calibrationLevel"] for r in unique)
        ),
        "raw_processed_relationships": [
            {
                "observationID": obs,
                "planes": [
                    {
                        k: p.get(k)
                        for k in (
                            "planeID",
                            "productID",
                            "calibrationLevel",
                            "provenance_inputs",
                        )
                    }
                    for p in planes
                ],
            }
            for obs, planes in grouped.items()
        ],
        "relationship_scope": "shared observation container; provenance_inputs retained, not a proven pixel lineage",
        "public_rights_rule": "Plane.dataRelease <= query UTC; null/future releases excluded",
    }


def smoka_recipe(cone: Cone) -> dict[str, Any]:
    return {
        "status": "interactive_query_required",
        "complete": False,
        "instruments": {
            "HSC": SOURCES["smoka_hsc"],
            "Suprime-Cam": SOURCES["smoka_sup"],
        },
        "fields": {
            "resolver": "Don't Resolve",
            "coordinate_system": "Equatorial",
            "equinox": "J2000",
            "ra_decimal_deg": cone.ra_deg,
            "dec_decimal_deg": cone.dec_deg,
            "radius_arcmin": cone.radius_deg * 60,
            "field_of_view": "auto",
            "date": "unrestricted",
            "proposal": "unrestricted",
            "observation_mode": "IMAG",
            "data_type": "OBJECT",
            "filters": "all",
            "output": "ASCII",
            "grouping": "SHOT then FRAME for CCD lineage",
            "columns": [
                "FRAMEID",
                "EXP_ID",
                "DATE_OBS",
                "UT_START",
                "EXPTIME",
                "RA2000",
                "DEC2000",
                "FILTER",
            ],
        },
        "limitations": [
            "Archive/FOV-expanded candidate search; inspect and record actual displayed spatial SQL, then verify individual CCD footprints. Do not assume an exact cone or a square-only search.",
            "ASCII output cap is 20000 rows; HTML cap is 5000. Check reported hit count and partition date/frame ranges if capped.",
            "Web form uses cookies and JavaScript. No stable documented metadata API verified; failed/error HTML must not be parsed as zero results.",
            "SMOKA raw data searches are public; FITS retrieval requires registration. No retrieval or email request is performed.",
            "HSC/SUP raw frames need separate calibration and detector/header validation before training; SHOT is not CCD frame count.",
            "The OBJECT UI selection can include FOCUS/FOCUSING subtypes; inspect returned data types before selecting science observations.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ra-deg", type=float, default=150.116321)
    parser.add_argument("--dec-deg", type=float, default=2.20583)
    parser.add_argument("--radius-deg", type=float, default=0.03)
    parser.add_argument(
        "--archives",
        nargs="+",
        choices=("mast", "cadc", "smoka"),
        default=["mast", "cadc", "smoka"],
    )
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--product-observations-per-collection", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--timeout", type=float, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cone = Cone(args.ra_deg, args.dec_deg, args.radius_deg)
    # requests is optional harness tooling and absent from module import path.
    import requests

    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "cone": asdict(cone),
        "metadata_only": True,
        "proposal_and_observation_dates": "unrestricted",
        "sources": SOURCES,
        "archives": {},
        "training_ready": False,
    }
    errors = False
    with requests.Session() as session:
        for archive in args.archives:
            try:
                if archive == "mast":
                    record = query_mast(
                        session,
                        cone,
                        page_size=args.page_size,
                        max_pages=args.max_pages,
                        timeout=args.timeout,
                        product_observations_per_collection=args.product_observations_per_collection,
                    )
                elif archive == "cadc":
                    record = query_cadc(
                        session,
                        cone,
                        max_records=args.max_records,
                        timeout=args.timeout,
                    )
                else:
                    record = smoka_recipe(cone)
                result["archives"][archive] = record
                errors |= any(
                    p.get("status") == "error"
                    for p in record.get("product_queries", [])
                )
            except (ArchiveQueryError, requests.RequestException, ValueError) as exc:
                result["archives"][archive] = {
                    "status": "error",
                    "complete": False,
                    "error": str(exc),
                    "queries": getattr(exc, "provenance", []),
                }
                errors = True
    result["finished_at_utc"] = utc_now()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "errors": errors, "metadata_only": True}
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
