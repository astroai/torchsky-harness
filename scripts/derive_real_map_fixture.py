#!/usr/bin/env python3
"""Derive a compact real-image fixture from a real CFHT MegaPipe mosaic.

Extracts a contiguous 256x256 cutout from the actual 2 GB g-band mosaic,
records its WCS metadata, and stores only the cutout plus WCS keywords. The raw
mosaic is not redistributed; the provenance file records the source path, size,
sha256, and cutout origin.

Usage:
    pixi run -e bench-all python scripts/derive_real_map_fixture.py \
        --mosaic ~/data/CFHT/MegaPipe.120.258/g/mosaic_g.fits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

WCS_KEYS = (
    "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2", "BUNIT", "WCSAXES", "RADESYS",
    "EQUINOX", "CUNIT1", "CUNIT2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks" / "fixtures" / "real_map" / "cfht_g_cutout_256.npz",
    )
    args = parser.parse_args()

    import torchfits

    t0 = time.perf_counter()
    path = str(args.mosaic)
    _, shape = torchfits.read_shape(path, hdu=0)
    ny, nx = (int(v) for v in shape[-2:])
    cy, cx = ny // 2, nx // 2
    half = args.size // 2
    image = (
        torchfits.read_subset(path, 0, cx - half, cy - half, cx + half, cy + half)
        .numpy()
        .astype(np.float32)
    )
    header = dict(torchfits.read_header(path, hdu=0))
    wcs_header = {key: header[key] for key in WCS_KEYS if key in header}
    wcs_header["NAXIS1"] = int(args.size)
    wcs_header["NAXIS2"] = int(args.size)
    wcs_header["CRPIX1"] = float(header["CRPIX1"]) - cx + half
    wcs_header["CRPIX2"] = float(header["CRPIX2"]) - cy + half

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image=image,
        header_json=np.array(json.dumps(wcs_header, sort_keys=True)),
        cutout_origin=np.array([int(cy - half), int(cx - half)]),
    )
    provenance = args.output.with_name("PROVENANCE.md")
    provenance.write_text(
        "# Real-image map fixture\n\n"
        f"- Source: `{args.mosaic}` ({args.mosaic.stat().st_size} bytes, "
        f"sha256 `{_sha256(args.mosaic)}`)\n"
        f"- Derived: CFHT MegaPipe g-band mosaic, contiguous "
        f"{args.size}x{args.size} cutout centred on the mosaic, full resolution\n"
        f"- WCS: the original TAN header recentred for the cutout\n"
        f"- Generator: `scripts/derive_real_map_fixture.py` "
        f"({time.perf_counter() - t0:.1f}s)\n\n"
        "The fixture stores one real cutout plus its WCS keywords; raw survey "
        "pixels are not redistributed beyond this small cutout.\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
