"""Out-of-process AstroDrizzle kernel oracle (drizzlepac ``adrizzle.do_driz`` -> ``cdriz.tdriz``).

    python drizzle_oracle.py --in in.npz --out out.npz

in.npz:  src_cards, dst_cards (80-char FITS card strings), image, weight (float32 2-D),
         kernel (str), pixfrac (float)
out.npz: sci, wht (float32, target grid), version (str)

The input WCS (``pscale`` set from its CD matrix, as stwcs does) and output WCS are astropy
WCS objects built from the cards; no FITS file is read or written.
"""
import argparse

import numpy as np


def _wcs(cards):
    from astropy.wcs import WCS

    header = {}
    for card in cards:
        card = str(card)
        key = card[:8].strip()
        if card[8:10] != "= ":
            continue
        body = card[10:]
        if body.lstrip().startswith("'"):
            header[key] = body.split("'")[1].strip()
        else:
            val = body.split("/")[0].strip()
            header[key] = float(val)
    for key in ("NAXIS", "NAXIS1", "NAXIS2"):
        if key in header:
            header[key] = int(header[key])
    w = WCS(header)
    w.pixel_shape = (int(header["NAXIS1"]), int(header["NAXIS2"]))
    cd = np.array([[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]])
    w.pscale = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = np.load(args.inp, allow_pickle=True)
    from drizzlepac import __version__, adrizzle

    src, dst = _wcs(list(data["src_cards"])), _wcs(list(data["dst_cards"]))
    ny, nx = dst.pixel_shape[1], dst.pixel_shape[0]
    outsci = np.zeros((ny, nx), dtype=np.float32)
    outwht = np.zeros((ny, nx), dtype=np.float32)
    outcon = np.zeros((ny, nx), dtype=np.int32)
    adrizzle.do_driz(
        data["image"].astype(np.float32), src, data["weight"].astype(np.float32), dst,
        outsci, outwht, outcon, 1.0, "cps", 1.0,
        wcslin_pscale=src.pscale, pixfrac=float(data["pixfrac"]), kernel=str(data["kernel"]), fillval="0",
    )  # fmt: skip
    np.savez(args.out, sci=outsci, wht=outwht, version="drizzlepac %s" % __version__)


if __name__ == "__main__":
    main()
