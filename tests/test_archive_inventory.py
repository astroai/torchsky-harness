from __future__ import annotations

import json

import pytest

from torchsky_harness.archive_inventory import (
    ArchiveQueryError,
    Cone,
    cadc_query,
    classify_cadc_plane,
    classify_mast_product,
    mast_observation_request,
    parse_tap_votable,
    query_cadc,
    query_mast,
    query_mast_pages,
    smoka_recipe,
)


class Response:
    status_code = 200
    url = "https://official.example/query"

    def __init__(self, payload):
        self.payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self.content = self.text.encode()

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(next(self.responses))


def page(rows, number=1, total=None, size=2):
    total = len(rows) if total is None else total
    return {
        "status": "COMPLETE",
        "data": rows,
        "paging": {
            "page": number,
            "pageSize": size,
            "pagesFiltered": (total + size - 1) // size,
            "rows": len(rows),
            "rowsFiltered": total,
        },
    }


def test_all_pages_query_keeps_later_visits_and_provenance():
    session = Session(
        page([{"obsid": 1}, {"obsid": 2}], total=3),
        page([{"obsid": 3}], number=2, total=3),
    )
    result = query_mast_pages(
        session, mast_observation_request(Cone(), 2), identity="obsid"
    )
    assert result["complete"] and not result["truncated"]
    assert result["unique_row_count"] == 3
    assert [
        json.loads(call[1]["data"]["request"])["page"] for call in session.calls
    ] == [1, 2]
    assert all(
        q["response_sha256"] and q["received_at_utc"].endswith("+00:00")
        for q in result["queries"]
    )
    filters = result["queries"][0]["request"]["mast_request"]["params"]["filters"]
    assert {f["paramName"] for f in filters} == {
        "obs_collection",
        "dataproduct_type",
        "dataRights",
    }


def test_page_cap_reports_truncation():
    result = query_mast_pages(
        Session(page([{"obsid": 1}, {"obsid": 2}], total=3)),
        mast_observation_request(Cone(), 2),
        identity="obsid",
        max_pages=1,
    )
    assert not result["complete"] and result["truncated"]
    assert result["reported_total"] == 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(status="ERROR"),
        lambda p: p.update(data={}),
        lambda p: p.pop("paging"),
        lambda p: p["paging"].update(page=2),
        lambda p: p["paging"].update(rows=0),
        lambda p: p["paging"].update(rowsFiltered=True),
        lambda p: p["paging"].update(pagesFiltered=20),
    ],
)
def test_malformed_mast_fails_closed(mutate):
    payload = page([{"obsid": 1}])
    mutate(payload)
    with pytest.raises(ArchiveQueryError):
        query_mast_pages(
            Session(payload), mast_observation_request(Cone(), 2), identity="obsid"
        )


def test_changed_paging_and_conflicting_duplicates_fail_closed():
    with pytest.raises(ArchiveQueryError, match="changed"):
        query_mast_pages(
            Session(
                page([{"obsid": 1}, {"obsid": 2}], total=3),
                page([{"obsid": 3}], number=2, total=4),
            ),
            mast_observation_request(Cone(), 2),
            identity="obsid",
        )
    with pytest.raises(ArchiveQueryError, match="conflicting duplicate"):
        query_mast_pages(
            Session(page([{"obsid": 1, "x": 1}, {"obsid": 1, "x": 2}])),
            mast_observation_request(Cone(), 2),
            identity="obsid",
        )


def test_identical_duplicates_report_coverage_gap_and_empty_results_valid():
    result = query_mast_pages(
        Session(page([{"obsid": 1}, {"obsid": 1}])),
        mast_observation_request(Cone(), 2),
        identity="obsid",
    )
    assert result["duplicate_count"] == 1 and not result["complete"]
    empty = query_mast_pages(
        Session(page([])), mast_observation_request(Cone(), 2), identity="obsid"
    )
    assert empty["complete"] and empty["rows"] == []


@pytest.mark.parametrize(
    "mission,name,rights,expected",
    [
        ("HST", "j9cm01010_flt.fits", "PUBLIC", "native_exposure_candidate"),
        ("HST", "j9cm01010_drc.fits", "PUBLIC", "derived_reference"),
        ("HST", "j9cm01010_crc.fits", "PUBLIC", "derived_reference"),
        (
            "JWST",
            "jw01523003001_03102_00002_nrcalong_cal.fits",
            "PUBLIC",
            "native_exposure_candidate",
        ),
        (
            "JWST",
            "jw01523-o001_t001_nircam_f277w_i2d.fits",
            "PUBLIC",
            "derived_reference",
        ),
        ("JWST", "jw01523-o001_t001_nircam_f277w_cal.fits", "PUBLIC", "unknown"),
        ("HST", "j9cm01010_flt.fits", "EXCLUSIVE ACCESS", "ineligible"),
        ("HST", "j9cm01010_flt.fits", "", "ineligible"),
    ],
)
def test_conservative_classification(mission, name, rights, expected):
    result = classify_mast_product(
        {
            "obs_collection": mission,
            "productFilename": name,
            "dataRights": rights,
            "productType": "SCIENCE",
        }
    )
    assert result["classification"] == expected
    assert not result["verified_native"]


def test_product_sampling_balances_collections_and_retains_parent_lineage():
    observations = [
        {"obsid": i, "obs_collection": mission, "dataRights": "PUBLIC"}
        for i, mission in [(1, "HST"), (2, "JWST")]
    ]
    product = {"dataURI": "mast:JWST/product/x", "parent_obsid": 22, "obsID": 2}
    session = Session(page(observations), page([]), page([product]))
    result = query_mast(
        session, Cone(), page_size=2, product_observations_per_collection=1
    )
    assert result["sampled_product_observation_ids"] == ["1", "2"]
    assert result["product_queries"][1]["rows"][0]["parent_obsid"] == 22
    assert result["product_lineage_complete"]


def test_shared_artifact_across_lineage_rows_is_not_a_conflict():
    rows = [
        {"dataURI": "mast:JWST/product/shared_pool.csv", "obsID": i, "parent_obsid": i}
        for i in (1, 2)
    ]
    result = query_mast_pages(
        Session(page(rows)), mast_observation_request(Cone(), 2), identity="dataURI"
    )
    assert result["complete"] and len(result["rows"]) == 2


def test_product_error_preserves_complete_observation_census():
    observations = [{"obsid": 1, "obs_collection": "HST", "dataRights": "PUBLIC"}]
    result = query_mast(
        Session(page(observations), {"status": "ERROR"}),
        Cone(),
        page_size=2,
        product_observations_per_collection=1,
    )
    assert result["observations"]["complete"]
    product = result["product_queries"][0]
    assert product["status"] == "error" and product["queries"]
    assert not result["product_lineage_complete"]


def test_http_error_retains_exact_attempted_query_and_response_hash():
    class ErrorResponse(Response):
        status_code = 400

    class ErrorSession:
        def post(self, *args, **kwargs):
            return ErrorResponse("invalid format")

    with pytest.raises(ArchiveQueryError, match="HTTP 400") as caught:
        query_mast_pages(
            ErrorSession(), mast_observation_request(Cone(), 2), identity="obsid"
        )
    provenance = caught.value.provenance[0]
    assert provenance["http_status"] == 400
    assert provenance["response_sha256"]
    assert (
        json.loads(provenance["request"]["request"])["service"]
        == "Mast.Caom.Filtered.Position"
    )


def votable(rows="", status="OK", final=""):
    return f'''<VOTABLE xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE type="results">
    <INFO name="QUERY_STATUS" value="{status}"/><TABLE>
    <FIELD name="planeID"/><FIELD name="calibrationLevel"/>
    <DATA><TABLEDATA>{rows}</TABLEDATA></DATA></TABLE>{final}</RESOURCE></VOTABLE>'''


def test_tap_tabledata_and_post_table_overflow():
    rows, overflow = parse_tap_votable(
        votable(
            "<TR><TD>p1</TD><TD>1</TD></TR>",
            final='<INFO name="QUERY_STATUS" value="OVERFLOW"/>',
        )
    )
    assert rows == [{"planeID": "p1", "calibrationLevel": "1"}]
    assert overflow


@pytest.mark.parametrize(
    "payload",
    [
        "<html>Error</html>",
        "not xml",
        votable(status="ERROR"),
        votable().replace('<INFO name="QUERY_STATUS" value="OK"/>', ""),
        votable("<TR><TD>p1</TD></TR>"),
        votable().replace("TABLEDATA", "BINARY"),
    ],
)
def test_tap_malformed_and_error_responses_fail_closed(payload):
    with pytest.raises(ArchiveQueryError):
        parse_tap_votable(payload)


def test_cadc_query_uses_footprint_all_levels_unrestricted_program_date():
    query = cadc_query(Cone(radius_deg=1), "2026-10-12T12:30:00+00:00")
    assert (
        "INTERSECTS(p.position_bounds, CIRCLE('ICRS', 150.116321, 2.20583, 1)) = 1"
        in query
    )
    assert "instrument_name = 'MegaPrime'" in query
    assert "p.dataRelease <= '2026-10-12T12:30:00'" in query
    assert "AND p.calibrationLevel" not in query
    assert "AND o.proposal_id" not in query and "AND p.time_" not in query


def test_cadc_groups_raw_processed_planes_by_observation():
    fields = [
        "planeID",
        "calibrationLevel",
        "dataRelease",
        "collection",
        "instrument_name",
        "observationID",
        "productID",
        "algorithm_name",
        "provenance_inputs",
        "intent",
        "dataProductType",
    ]
    values = [
        [
            "p1",
            "1",
            "2000-01-01T00:00:00Z",
            "CFHT",
            "MegaPrime",
            "100",
            "100o",
            "exposure",
            "",
            "science",
            "image",
        ],
        [
            "p2",
            "2",
            "2000-01-01T00:00:00Z",
            "CFHT",
            "MegaPrime",
            "100",
            "100p",
            "exposure",
            "caom:CFHT/100/100o",
            "science",
            "image",
        ],
    ]
    payload = '<VOTABLE><RESOURCE><INFO name="QUERY_STATUS" value="OK"/><TABLE>'
    payload += "".join(f'<FIELD name="{f}"/>' for f in fields)
    payload += "<DATA><TABLEDATA>" + "".join(
        "<TR>" + "".join(f"<TD>{v}</TD>" for v in row) + "</TR>" for row in values
    )
    payload += "</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"
    result = query_cadc(Session(payload), Cone())
    assert result["complete"]
    assert len(result["raw_processed_relationships"]) == 1
    assert len(result["raw_processed_relationships"][0]["planes"]) == 2
    assert all(
        r["candidate_assessment"]["classification"] == "native_exposure_candidate"
        for r in result["rows"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"ra_deg": 360}, {"dec_deg": 91}, {"radius_deg": 0}, {"radius_deg": float("nan")}],
)
def test_invalid_cone(kwargs):
    with pytest.raises(ValueError):
        Cone(**kwargs)


@pytest.mark.parametrize(
    "intent,algorithm,expected",
    [
        ("science", "exposure", "native_exposure_candidate"),
        ("calibration", "exposure", "calibration_evidence_candidate"),
        ("science", "master_detrend", "unknown"),
        ("", "exposure", "unknown"),
    ],
)
def test_cadc_science_and_calibration_candidates_are_distinct(
    intent, algorithm, expected
):
    result = classify_cadc_plane(
        {
            "calibrationLevel": "2",
            "intent": intent,
            "algorithm_name": algorithm,
            "dataProductType": "image",
        }
    )
    assert result["classification"] == expected and not result["verified_native"]


def test_smoka_recipe_converts_degrees_to_arcmin_and_is_not_claimed_complete():
    result = smoka_recipe(Cone(radius_deg=1))
    assert result["fields"]["radius_arcmin"] == 60
    assert not result["complete"]
    assert "Suprime-Cam" in result["instruments"] and "HSC" in result["instruments"]
