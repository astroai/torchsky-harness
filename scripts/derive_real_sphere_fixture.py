#!/usr/bin/env python3
"""Derive a compact real-sky HEALPix map fixture from a real CFHT mosaic.

The harness principle is real data: this script samples the actual CFHT
MegaPipe g-band mosaic on disk, bins every valid image sample into HEALPix
pixels at ``--nside``, and stores only the non-empty pixels plus a sample of
real sky positions. The resulting 27 KB fixture drives
``benchmarks/bench_sphere_real_map.py`` without requiring the 2 GB mosaic at
evaluation time.

Usage:
    pixi run -e bench-all python scripts/derive_real_sphere_fixture.py \
        --mosaic ~/data/CFHT/MegaPipe.120.258/g/mosaic_g.fits

The fixture is derived data: raw survey pixels are not redistributed, and the
provenance file records the source path, size, stride, and nside.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic", type=Path, required=True)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "fixtures"
        / "real_sphere"
        / "cfht_megapipe_g_nside512.npz",
    )
    args = parser.parse_args()

    import healpy as hp
    import torchfits
    from astropy.wcs import WCS

    t0 = time.perf_counter()
    full, header = torchfits.read(str(args.mosaic), hdu=0, return_header=True)
    data = full.numpy()[:: args.stride, :: args.stride].astype(np.float32)
    del full
    wcs = WCS(dict(header), relax=True)
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = wcs.all_pix2world(
        (xx.ravel() * args.stride).astype(np.float64),
        (yy.ravel() * args.stride).astype(np.float64),
        0,
    )
    pix = hp.ang2pix(
        args.nside, np.deg2rad(90.0 - dec), np.deg2rad(ra), nest=False
    )
    values = data.ravel().astype(np.float64)
    valid = np.isfinite(values) & (values != 0.0)
    pix = pix[valid]
    values = values[valid]
    npix = hp.nside2npix(args.nside)
    counts = np.bincount(pix, minlength=npix)
    sums = np.bincount(pix, weights=values, minlength=npix)
    occupied = counts > 0
    mean = np.where(occupied, sums / np.maximum(counts, 1), 0.0)[occupied]
    sample = slice(0, ra.size, max(1, ra.size // 4000))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        nside=np.array(args.nside),
        pix=np.nonzero(occupied)[0].astype(np.int64),
        value=mean,
        counts=counts[occupied],
        sample_ra=ra[sample][:4000],
        sample_dec=dec[sample][:4000],
    )
    provenance = args.output.with_name("PROVENANCE.md")
    provenance.write_text(
        "# Real-sky HEALPix fixture\n\n"
        f"- Source: `{args.mosaic}` ({args.mosaic.stat().st_size} bytes, "
        f"sha256 `{_sha256(args.mosaic)}`)\n"
        f"- Derived: CFHT MegaPipe g-band mosaic, stride {args.stride}, "
        f"HEALPix nside {args.nside} (RING), bin mean\n"
        f"- Occupied pixels: {int(occupied.sum())} of {npix}\n"
        f"- Generator: `scripts/derive_real_sphere_fixture.py` "
        f"({time.perf_counter() - t0:.1f}s)\n\n"
        "The fixture stores only per-pixel mean surface brightness and a sample "
        "of real sky positions; raw survey pixels are not redistributed.\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes, "
        f"{int(occupied.sum())} occupied pixels)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
