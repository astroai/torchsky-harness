"""Out-of-process WCS oracle (PyAST); Python 3.6+ compatible, numpy only.

Runs in a SEPARATE interpreter from torchsky so the legacy comparators may use whatever
Python/numpy they build on. Exchange is NPZ in / NPZ out; header cards arrive as 80-char card
strings (no astropy.io.fits anywhere).

    python wcs_oracle.py --engine pyast --in in.npz --out out.npz

in.npz:  cards (unicode array, no pickle), x, y (FITS 1-based pixel), lon, lat (degrees), repeat (int)
out.npz: ra, dec (degrees, from x/y), px, py (FITS 1-based pixel, from lon/lat),
         t_forward_s, t_inverse_s (best of ``repeat`` calls, comparator-only wall time),
         version (str)
"""
import argparse
import time

import numpy as np


def _best(fn, repeat):
    best = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None or dt < best else best
    return out, best


def _ast_version(Ast):
    v = int(Ast.version())
    return "%d.%d.%d" % (v // 1000000, (v // 1000) % 1000, v % 1000)


def run_pyast(cards, x, y, lon, lat, repeat):
    import starlink.Ast as Ast

    chan = Ast.FitsChan()
    for card in cards:
        chan.putfits(str(card), False)
    chan.clear("Card")
    wcs = chan.read()
    if wcs is None:
        raise RuntimeError("AST could not read a WCS from the cards")
    # Base frame is GRID (1-based, centre of first pixel = 1), current frame is the sky frame (radians).
    fwd, t_f = _best(lambda: wcs.tran(np.vstack([x, y]), True), repeat)
    inv, t_i = _best(lambda: wcs.tran(np.vstack([np.radians(lon), np.radians(lat)]), False), repeat)
    ra = np.degrees(fwd[0]) % 360.0
    dec = np.degrees(fwd[1])
    return ra, dec, inv[0], inv[1], t_f, t_i, "starlink-pyast %s" % _ast_version(Ast)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["pyast"], required=True)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = np.load(args.inp, allow_pickle=False)
    fn = run_pyast
    ra, dec, px, py, t_f, t_i, version = fn(
        list(data["cards"]), data["x"], data["y"], data["lon"], data["lat"], int(data["repeat"])
    )
    np.savez(
        args.out, ra=ra, dec=dec, px=px, py=py, t_forward_s=t_f, t_inverse_s=t_i, version=version
    )


if __name__ == "__main__":
    main()
