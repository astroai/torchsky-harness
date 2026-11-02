from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns" / "cfht_union_pitcairn"
SCRIPT = CAMPAIGN / "discover.py"
FIXTURES = CAMPAIGN / "fixtures"
SPEC = importlib.util.spec_from_file_location("cfht_union_pitcairn_discover", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DISCOVERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DISCOVERY
SPEC.loader.exec_module(DISCOVERY)


def fixture_args(output: Path) -> list[str]:
    return [
        str(SCRIPT),
        "--tiles-listing",
        str(FIXTURES / "tiles.listing"),
        "--pitcairn-u-listing",
        str(FIXTURES / "pitcairn-u.listing"),
        "--pitcairn-r-listing",
        str(FIXTURES / "pitcairn-r.listing"),
        "--expcxy-u",
        str(FIXTURES / "CFIS.000.229.u.expcxy"),
        "--expcxy-r",
        str(FIXTURES / "CFIS.000.229.r.expcxy"),
        "--json-out",
        str(output),
    ]


def test_listing_parser_accepts_vls_shapes() -> None:
    entries = DISCOVERY.parse_listing(
        (FIXTURES / "tiles.listing").read_text(), DISCOVERY.TILES_URI
    )
    by_name = {entry.name: entry for entry in entries}
    assert by_name["CFIS.000.229.u.fits"].size_bytes == 400006080
    assert by_name["CFIS.000.229.r.fits"].uri == (
        "vos:cfis/tiles_DR6/CFIS.000.229.r.fits"
    )


def test_expcxy_parser_deduplicates_supported_names() -> None:
    text = (FIXTURES / "CFIS.000.229.u.expcxy").read_text()
    assert DISCOVERY.extract_exposure_ids(text) == [
        "1833406p.fits.fz",
        "1833429p.fits.fz",
        "1833624p.fits.fz",
        "2513912p.fits.fz",
        "2513914p.fits.fz",
        "2513915p.fits.fz",
    ]


def test_fixture_discovery_emits_verified_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    completed = subprocess.run(
        [sys.executable, *fixture_args(output)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["status"] == "verified"
    assert payload["tile"] == "CFIS.000.229"
    assert {record["band"] for record in payload["files"]} == {"u", "r"}
    assert len(payload["files"]) == 14
    assert all(record["download"] is None for record in payload["files"])
    immutable = payload["provenance"]["immutable_inputs"]
    assert immutable["expcxy"]["u"]["extracted_exposures"] == [
        "1833406p.fits.fz",
        "1833429p.fits.fz",
        "1833624p.fits.fz",
        "2513912p.fits.fz",
        "2513914p.fits.fz",
        "2513915p.fits.fz",
    ]
    assert all(item["sha256"] for item in immutable["listings"].values())


def test_discovery_fails_closed_on_contributor_mismatch(tmp_path: Path) -> None:
    bad_expcxy = tmp_path / "bad.expcxy"
    bad_expcxy.write_text("2506043 1 0.0 0.0\n")
    args = fixture_args(tmp_path / "manifest.json")
    index = args.index(str(FIXTURES / "CFIS.000.229.u.expcxy"))
    args[index] = str(bad_expcxy)
    completed = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert completed.returncode == 2
    assert "contributors differ from bounded set" in completed.stderr


def test_fetch_requires_scratch(tmp_path: Path) -> None:
    parser = DISCOVERY.build_parser()
    args = parser.parse_args(
        [
            *fixture_args(tmp_path / "manifest.json")[1:],
            "--scratch-dir",
            # Not tmp_path: CANFAR sessions put TMPDIR under /scratch, which
            # would satisfy the very guard this test expects to trip.
            "/var/tmp/torchsky-not-under-scratch",
            "--fetch",
        ]
    )
    with pytest.raises(DISCOVERY.DiscoveryError, match="under /scratch"):
        DISCOVERY.build_manifest(args)


def test_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert "--dry-run" in completed.stdout
    assert "--fetch" in completed.stdout


def test_parse_listing_uses_size_column_for_small_files() -> None:
    listing = "\n".join(
        [
            "-rw-rw---- owner read write 1842 Aug 29 2025 CFIS.000.229.u.expcxy",
            "-rw-rw---- owner read write 921 Jul 03 2025 CFIS.000.229.r.expcxy",
        ]
    )
    parsed = {
        entry.name: entry.size_bytes
        for entry in DISCOVERY.parse_listing(listing, "vos:cfis/tiles_DR6")
    }
    assert parsed == {
        "CFIS.000.229.r.expcxy": 921,
        "CFIS.000.229.u.expcxy": 1842,
    }


def test_canfar_runner_fails_closed_on_existing_destinations() -> None:
    runner = (CAMPAIGN / "run_canfar.sh").read_text()
    assert 'test ! -e "${work_root}"' in runner
    assert 'test ! -e "${persistent_dest}"' in runner
    assert 'case "${persistent_dest}" in' in runner
    assert "/arc/projects/*" in runner
    assert "trap persist EXIT" in runner
