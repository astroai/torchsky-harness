"""Absolute astrometry of the official UNIONS tile and the torchsky coadd against Gaia DR3.

usage: unions_gaia_astrom.py --tile CFIS.000.229 --bands r,u --coadd-dir DIR --work /scratch/g --out OUT

For each band: Gaia DR3 stars (5/6-parameter solutions, RUWE < 1.4) propagated to the mean epoch of the
contributing exposures (MJD-OBS from their headers), projected with torchsky.wcs on the tile TAN grid; iterative
Gaussian-windowed centroids (SExtractor XWIN-like) on both images, from the same starting positions.
Outputs per band: offsets image - Gaia [mas] for both images, their difference, binned maps vs position and
magnitude, and a figure. Images and headers read with torchfits.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchfits
from torchsky.wcs import WCS

LOG: dict = {}
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def vcp(src, dst, head=False):
    cmd = ["vcp"] + (["--head"] if head else []) + [src, str(dst)]
    for k in range(3):
        if subprocess.run(cmd).returncode == 0:
            return
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"vcp failed {src}")


def gaia_query(ra0, dec0, radius):
    """Gaia DR3 cone; ESA archive first, VizieR as fallback. Returns dict of numpy arrays."""
    cols = "source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, ruwe, astrometric_params_solved, ra_error, dec_error"
    try:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = -1
        q = (f"SELECT {cols} FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {ra0}, {dec0}, {radius})) AND phot_g_mean_mag < 21")
        t = Gaia.launch_job_async(q).get_results()
        LOG["gaia_service"] = "ESA Gaia archive"
        get = lambda c: np.asarray(t[c], dtype=np.float64)
        return {c: get(c) for c in ["ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag", "ruwe", "astrometric_params_solved"]}
    except Exception as exc:  # noqa: BLE001
        tick(f"ESA Gaia failed: {exc!r}; trying VizieR")
    from astroquery.vizier import Vizier
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    v = Vizier(columns=["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Plx", "Gmag", "RUWE", "Solved"], row_limit=-1,
               column_filters={"Gmag": "<21"})
    t = v.query_region(SkyCoord(ra0, dec0, unit="deg"), radius=radius * u.deg, catalog="I/355/gaiadr3")[0]
    LOG["gaia_service"] = "VizieR I/355/gaiadr3 (epoch 2016.0)"
    g = lambda c: np.asarray(t[c].filled(np.nan) if hasattr(t[c], "filled") else t[c], dtype=np.float64)
    return {"ra": g("RA_ICRS"), "dec": g("DE_ICRS"), "pmra": g("pmRA"), "pmdec": g("pmDE"), "parallax": g("Plx"),
            "phot_g_mean_mag": g("Gmag"), "ruwe": g("RUWE"), "astrometric_params_solved": g("Solved")}


def xwin(img, x, y, sig, niter=16):
    """Iterative Gaussian-windowed centroid (SExtractor XWIN_IMAGE recipe), local annulus background.
    x, y: 0-based starting positions. Returns refined x, y (nan when off-image or not converged)."""
    H, W = img.shape
    R = int(np.ceil(4 * sig)) + 8
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    ox, oy = np.full(x.size, np.nan), np.full(x.size, np.nan)
    for k, (xi, yi) in enumerate(zip(x, y)):
        x0, y0 = int(round(xi)), int(round(yi))
        if x0 - R < 0 or y0 - R < 0 or x0 + R >= W or y0 + R >= H:
            continue
        st = np.asarray(img[y0 - R:y0 + R + 1, x0 - R:x0 + R + 1], np.float64)
        rr0 = np.hypot(xx - (xi - x0), yy - (yi - y0))
        ann = st[(rr0 > R - 6) & (rr0 <= R)]
        if not np.isfinite(ann).any():
            continue
        s = np.nan_to_num(st - np.nanmedian(ann))
        cx, cy = xi - x0, yi - y0
        ok = False
        for _ in range(niter):
            dx, dy = xx - cx, yy - cy
            w = np.exp(-(dx * dx + dy * dy) / (2 * sig * sig)) * (dx * dx + dy * dy < (4 * sig) ** 2)
            tw = (w * s).sum()
            if not tw > 0:
                break
            ddx, ddy = 2 * (w * s * dx).sum() / tw, 2 * (w * s * dy).sum() / tw
            cx, cy = cx + ddx, cy + ddy
            if abs(cx - (xi - x0)) > 3 or abs(cy - (yi - y0)) > 3:
                break
            if ddx * ddx + ddy * ddy < 1e-8:
                ok = True
                break
        if ok:
            ox[k], oy[k] = cx + x0, cy + y0
    return ox, oy


def nmad(v):
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def binned_map(x, y, v, n, size):
    m = np.full((n, n), np.nan)
    c = np.zeros((n, n), int)
    ix, iy = np.clip((x / size * n).astype(int), 0, n - 1), np.clip((y / size * n).astype(int), 0, n - 1)
    for i in range(n):
        for j in range(n):
            k = (iy == i) & (ix == j) & np.isfinite(v)
            c[i, j] = k.sum()
            if k.sum() >= 5:
                m[i, j] = np.median(v[k])
    return m, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="CFIS.000.229")
    ap.add_argument("--bands", default="r,u")
    ap.add_argument("--coadd-dir", required=True)
    ap.add_argument("--work", default="/scratch/g")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gmin", type=float, default=17.0)
    ap.add_argument("--gmax", type=float, default=20.5)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    LOG["args"] = vars(a)
    LOG["versions"] = {"torch": torch.__version__, "torchfits": torchfits.__version__}
    try:
        import torchsky
        LOG["versions"]["torchsky"] = torchsky.__version__
    except Exception:  # noqa: BLE001
        pass
    res = {}
    for band in a.bands.split(","):
        tb = f"{a.tile}.{band}"
        tick(f"== {tb}")
        off = work / f"{tb}.fits"
        if not off.exists():
            vcp(f"vos:cfis/tiles_DR6/{tb}.fits", off)
        rf, hdr = torchfits.read(str(off), hdu=0, return_header=True)
        hdr = {k: hdr[k] for k in hdr.keys()} if not isinstance(hdr, dict) else hdr
        ts = torchfits.read(str(Path(a.coadd_dir) / f"{tb}.torchsky.fits"), hdu=0)
        rf, ts = rf.numpy(), ts.numpy()
        H, W = rf.shape
        # epochs of the contributing exposures (tile HISTORY lines, read as text)
        hp = work / f"{tb}.head.txt"
        if not hp.exists():
            vcp(f"vos:cfis/tiles_DR6/{tb}.fits", hp, head=True)
        hist = open(hp, errors="replace").read()
        exps = sorted(set(re.findall(r"input image (\d+)p\.fits", hist)))
        mjds = []
        for e in exps:
            hp = work / f"{e}.head.txt"
            if not hp.exists():
                vcp(f"vos:cfis/pitcairn/{e}p.fits.fz", hp, head=True)
            mm = re.search(r"^MJD-OBS\s*=\s*([0-9.Ee+-]+)", open(hp, errors="replace").read(), re.M)
            if mm:
                mjds.append(float(mm.group(1)))
        mjd = float(np.mean(mjds))
        epoch = 2000.0 + (mjd - 51544.5) / 365.25
        tick(f"{len(exps)} exposures, MJD {min(mjds):.1f}..{max(mjds):.1f}, mean epoch {epoch:.3f}")
        wcs = WCS(hdr)
        cx, cy = (W - 1) / 2, (H - 1) / 2
        ra0, dec0 = (float(v) for v in wcs.pixel_to_world_values(torch.tensor([cx], dtype=torch.float64), torch.tensor([cy], dtype=torch.float64), origin=0))
        scale_mas = abs(float(hdr["CD2_2"])) * 3.6e6
        g = gaia_query(ra0, dec0, 0.75 * H * scale_mas / 3.6e6)
        good = (g["ruwe"] < 1.4) & (g["astrometric_params_solved"] > 3) & (g["phot_g_mean_mag"] > a.gmin) & (g["phot_g_mean_mag"] < a.gmax)
        g = {k: v[good] for k, v in g.items()}
        dt = epoch - 2016.0
        pmra, pmdec = np.nan_to_num(g["pmra"]), np.nan_to_num(g["pmdec"])
        dec_t = g["dec"] + pmdec * dt / 3.6e6
        ra_t = g["ra"] + pmra * dt / 3.6e6 / np.cos(np.radians(g["dec"]))
        px, py = (t.numpy() for t in wcs.world_to_pixel_values(torch.tensor(ra_t), torch.tensor(dec_t), origin=0))
        inside = (px > 30) & (py > 30) & (px < W - 30) & (py < H - 30)
        px, py, gm = px[inside], py[inside], g["phot_g_mean_mag"][inside]
        tick(f"{inside.sum()} Gaia stars G {a.gmin}-{a.gmax} on the tile (dt = {dt:.2f} yr)")
        # window from the image PSF: median second moment of bright stars is overkill; use the tile seeing keyword or 0.7"
        fwhm_as = float(hdr.get("IQFINAL", hdr.get("SEEING", 0.7)) or 0.7)
        sig = max(fwhm_as / 2.3548 / (scale_mas / 1000), 1.0)
        out_b = {"exposures": exps, "mjd_mean": mjd, "epoch": epoch, "n_gaia": int(inside.sum()), "window_sigma_px": sig,
                 "pixel_mas": scale_mas}
        cent = {}
        for nm, im in (("official", rf), ("torchsky", ts)):
            xo, yo = xwin(im, px, py, sig)
            cent[nm] = (xo, yo)
            # RA increases to -x on this grid (CD1_1 < 0): report offsets on the sky axes, east and north
            de = -(xo - px) * scale_mas
            dn = (yo - py) * scale_mas
            ok = np.isfinite(de) & np.isfinite(dn)
            out_b[nm] = {"n": int(ok.sum()), "median_east_mas": float(np.median(de[ok])), "median_north_mas": float(np.median(dn[ok])),
                         "nmad_east_mas": nmad(de[ok]), "nmad_north_mas": nmad(dn[ok])}
            tick(f"{nm}: {out_b[nm]}")
        de_d = -(cent["torchsky"][0] - cent["official"][0]) * scale_mas
        dn_d = (cent["torchsky"][1] - cent["official"][1]) * scale_mas
        ok = np.isfinite(de_d) & np.isfinite(dn_d)
        out_b["torchsky_minus_official"] = {"n": int(ok.sum()), "median_east_mas": float(np.median(de_d[ok])),
                                            "median_north_mas": float(np.median(dn_d[ok])),
                                            "nmad_east_mas": nmad(de_d[ok]), "nmad_north_mas": nmad(dn_d[ok])}
        tick(f"torchsky - official: {out_b['torchsky_minus_official']}")
        maps = {}
        for nm in ("official", "torchsky"):
            de = -(cent[nm][0] - px) * scale_mas
            dn = (cent[nm][1] - py) * scale_mas
            maps[f"{nm}_east"], cnt = binned_map(px, py, de, 8, W)
            maps[f"{nm}_north"], _ = binned_map(px, py, dn, 8, W)
        maps["diff_east"], _ = binned_map(px, py, de_d, 8, W)
        maps["diff_north"], _ = binned_map(px, py, dn_d, 8, W)
        out_b["map_counts"] = cnt.tolist()
        out_b["maps_mas"] = {k: np.round(v, 2).tolist() for k, v in maps.items()}
        np.savez_compressed(out / f"{tb}.gaia_astrom.npz", px=px, py=py, gmag=gm,
                            x_off=cent["official"][0], y_off=cent["official"][1], x_ts=cent["torchsky"][0], y_ts=cent["torchsky"][1])
        res[band] = out_b
        figure(out / f"{tb}.gaia_astrom.png", tb, px, py, gm, cent, scale_mas, maps)
    LOG["bands"] = res
    json.dump(LOG, open(out / f"{a.tile}.gaia_astrom.json", "w"), indent=1, default=float)
    tick("done")


def figure(path, tb, px, py, gm, cent, scale_mas, maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    for r, nm in enumerate(("official", "torchsky")):
        de = -(cent[nm][0] - px) * scale_mas
        dn = (cent[nm][1] - py) * scale_mas
        axs[r, 0].scatter(de, dn, s=3, alpha=0.4)
        axs[r, 0].axhline(0, color="k", lw=0.5); axs[r, 0].axvline(0, color="k", lw=0.5)
        axs[r, 0].set_xlim(-150, 150); axs[r, 0].set_ylim(-150, 150); axs[r, 0].set_aspect("equal")
        axs[r, 0].set_title(f"{nm} - Gaia [mas] (E, N)")
        axs[r, 1].scatter(gm, de, s=2, alpha=0.3, label="east"); axs[r, 1].scatter(gm, dn, s=2, alpha=0.3, label="north")
        axs[r, 1].set_ylim(-150, 150); axs[r, 1].set_xlabel("G"); axs[r, 1].legend()
        for c, comp in enumerate(("east", "north")):
            im = axs[r, 2 + c].imshow(maps[f"{nm}_{comp}"], origin="lower", cmap="RdBu_r", vmin=-40, vmax=40)
            axs[r, 2 + c].set_title(f"{nm} {comp} offset map [mas]")
            plt.colorbar(im, ax=axs[r, 2 + c], shrink=0.8)
    fig.suptitle(tb)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
