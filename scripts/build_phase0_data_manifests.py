"""Regenerate ``campaigns/phase0/data/*.json`` data-identity manifests.

Each campaign TOML names its data manifests and declares ``[[datasets]]``; every
dataset id is pinned here to a tracked repository file (frozen matrix, vendored
fixture, or external-asset stub) by sha256. Run after changing any pinned file::

    pixi run -e test python scripts/build_phase0_data_manifests.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "campaigns" / "phase0"
FIXTURES = "benchmarks/fixtures"

# dataset id -> repository-relative file
PINS: dict[str, str] = {
    "astrometry-v1-matrix": "campaigns/phase0/matrices/astrometry-v1.json",
    "astrometry-megapipe-v1": f"{FIXTURES}/certification/astrometry_megapipe_v1.json",
    "sphere-v1-matrix": "campaigns/phase0/matrices/sphere-v1.json",
    "sphere-v1-fixture": f"{FIXTURES}/certification/sphere_v1.json",
    "deblend-v1-matrix": "campaigns/phase0/matrices/deblend-v1.json",
    "deblend-crowded-v1": f"{FIXTURES}/certification/deblend_crowded_v1.json",
    "drizzle-dq-cr-variance-fixtures-v1": "campaigns/phase0/matrices/drizzle-dq-v1.json",
    "hst-wfc3-ir-sample-ipppssoot-v1": f"{FIXTURES}/real_wcs/sample_ipppssoot_flt.fits",
    "adaptive-reprojection-matrix-v1": "campaigns/phase0/matrices/adaptive-reprojection-v1.json",
    "cfht-megapipe-g-cutout-256-v1": f"{FIXTURES}/real_map/cfht_g_cutout_256.npz",
    "jwst-ifu-calibrated-fixtures-v1": "campaigns/phase0/matrices/physical-ifu-v1.json",
    "nirspec-external-association-manifest-v1": "campaigns/phase0/external/nirspec-association-v1.json",
    "miri-external-association-manifest-v1": "campaigns/phase0/external/miri-association-v1.json",
}
for _family in ("astrometry", "sphere", "deblend"):
    for _path in sorted((PHASE0 / "external").glob(f"{_family}-*-v1.json")):
        PINS[_path.stem] = _path.relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict[Path, list[dict[str, str]]]:
    outputs: dict[Path, list[dict[str, str]]] = {}
    for config in sorted(PHASE0.glob("*.toml")):
        payload = tomllib.loads(config.read_text(encoding="utf-8"))
        manifests = payload.get("data_manifests") or []
        if not manifests:
            continue
        records = []
        for dataset in payload.get("datasets", []):
            pin = PINS.get(dataset["id"])
            if pin is None:
                raise SystemExit(f"{config.name}: no pinned file for dataset {dataset['id']!r}")
            records.append(
                {
                    "identity_type": "data",
                    "id": dataset["id"],
                    "role": dataset["role"],
                    "uri": f"file:{pin}",
                    "sha256": _sha256(ROOT / pin),
                }
            )
        for manifest in manifests:
            outputs[(PHASE0 / manifest).resolve()] = records
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if manifests are stale")
    args = parser.parse_args()
    stale = False
    for path, records in build().items():
        text = json.dumps(records, indent=2) + "\n"
        if args.check:
            stale |= not path.is_file() or path.read_text(encoding="utf-8") != text
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(records)} records)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
