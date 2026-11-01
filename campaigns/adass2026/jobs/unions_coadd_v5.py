"""v5: CCD frames declared ICRS when the astrometry is Gaia-based (see ccd_observations).
torchsky coadd of a UNIONS/CFIS tile from the pitcairn MegaCam exposures, compared with the official
MegaPipe/SWarp tile (vos:cfis/tiles_DR6).

usage: unions_coadd.py --tile CFIS.000.229 --band r --work /scratch/u --out OUT
Steps: contributing exposures from the tile's .expcxy -> fetch pitcairn MEFs -> per CCD: TPV WCS (headers say
TAN + PV), PHOTZP -> ZP 30 scaling, median background, saturation/zero masking, inverse-variance weight ->
torchsky coadd onto the official tile grid (Lanczos-3, flux semantics, sigma-clipped weighted mean) ->
compare with the official tile: pixel differences, aperture photometry and centroids at the tile catalogue
positions, background noise, PSF moments; figures.
"""
from __future__ import annotations

import argparse
import os
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

LOG: dict = {"timings": {}}
T0 = time.perf_counter()


def tick(msg):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def vcp(src, dst):
    if Path(dst).exists() and Path(dst).stat().st_size > 0:
        return
    subprocess.run(["vcp", src, str(dst)], check=True)


def contributing_exposures(expcxy: Path):
    txt = expcxy.read_text()
    ids = re.findall(r"\s(\d{7})\s+\d{1,2}\s+[0-9.]+\s+[0-9.]+", txt)
    u, c = np.unique(ids, return_counts=True)
    return [x for _, x in sorted(zip(-c, u))], dict(zip(u.tolist(), c.tolist()))


def robust_sigma(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    med = np.median(a)
    return 1.4826 * np.median(np.abs(a - med))


WM_CONFIG = """streak_masking:
  enable: false
  enable_sparse_ransac: false
"""
FILL_EPS = 1e-4
# SATURATED is kept at full weight, as in the official SWarp stack: the stack interpolates data with their weights,
# so zero-weight cores get filled from the wings and bright stars lose flux.
EXCLUDE = (1 << 0) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 6)  # BAD CR STREAK INVALID NO_DATA


def run_weightmask(mef: Path, flat: Path, work: Path):
    """weightmask per exposure: flat-based BAD mask, saturation + bleed, cosmic rays, object-masked SEP sky,
    theoretical inverse variance (gain, read noise, flat). Streak detection disabled."""
    cfg = work / "weightmask_overrides.yml"
    cfg.write_text(WM_CONFIG)
    stem = mef.name.replace(".fits.fz", "")
    outs = {k: work / f"{stem}.wm.{k}.fits" for k in ("weight", "mask", "invvar", "sky")}
    if all(o.exists() and o.stat().st_size > 0 for o in outs.values()):
        return outs
    cmd = ["weightmask", str(mef), "--config", str(cfg), "--flat_image", str(flat), "-o", str(outs["weight"]),
           "--output_mask", str(outs["mask"]), "--output_invvar", str(outs["invvar"]), "--output_sky", str(outs["sky"]),
           "--nproc", str(min(16, os.cpu_count() or 4))]
    t = time.perf_counter()
    subprocess.run(cmd, check=True)
    LOG.setdefault("timings", {}).setdefault("weightmask", {})[stem] = round(time.perf_counter() - t, 1)
    return outs


def ccd_observations(mef: Path, wm: dict, dev, zp_target=30.0):
    """Per CCD: pixels with torchfits; weightmask sky subtracted, weightmask inverse variance as weight with
    BAD/SAT/CR/NO_DATA pixels at zero weight; scaled to ZP=30 (ivar scaled accordingly)."""
    import torchfits
    from torchsky.map.observation import ImageObservation

    obs, meta = [], []
    # weightmask products name each HDU <PRODUCT>_<input EXTNAME>; match by name, not position
    idx = {}
    for key in ("sky", "invvar", "mask"):
        with fits.open(wm[key]) as h:
            idx[key] = {x.name.split("_", 1)[1]: i for i, x in enumerate(h) if "_" in x.name}
    with fits.open(mef) as h:  # products are named after CCDNAME (fallback EXTNAME)
        names = [str(x.header.get("CCDNAME", x.name)).strip().upper() for x in h]
    idx = {key: {n.upper(): i for n, i in v.items()} for key, v in idx.items()}
    nh = len(names)
    for k in range(1, nh):
        nm = names[k]
        if not all(nm in idx[key] for key in idx):
            continue
        d, hdr = torchfits.read(str(mef), hdu=k, return_header=True)
        if d is None or d.ndim != 2:
            continue
        d = d.to(dev, torch.float32)
        sky = torchfits.read(str(wm["sky"]), hdu=idx["sky"][nm]).to(dev, torch.float32)
        iv = torchfits.read(str(wm["invvar"]), hdu=idx["invvar"][nm]).to(dev, torch.float32)
        mk = torchfits.read(str(wm["mask"]), hdu=idx["mask"][nm]).to(dev).to(torch.int32)
        assert sky.shape == d.shape == iv.shape == mk.shape, (nm, d.shape, sky.shape)
        hd = {kk: hdr[kk] for kk in hdr.keys() if kk and kk not in ("COMMENT", "HISTORY", "")}
        for key in ("ZNAXIS", "ZNAXIS1", "ZNAXIS2", "ZIMAGE", "ZBITPIX", "ZCMPTYPE", "ZTILE1", "ZTILE2", "ZQUANTIZ", "XTENSION", "PCOUNT", "GCOUNT"):
            hd.pop(key, None)
        hd["NAXIS"], hd["NAXIS1"], hd["NAXIS2"] = 2, d.shape[1], d.shape[0]
        if any(k_.startswith("PV1_") for k_ in hd) and str(hd.get("CTYPE1", "")).endswith("TAN"):
            hd["CTYPE1"], hd["CTYPE2"] = "RA---TPV", "DEC--TPV"
        # The MegaCam CCD headers keep the Elixir-era RADECSYS = 'FK5' (EQUINOX 2000) while MegaPipe solved the
        # astrometry against Gaia (ASTCAT): the solution is ICRS. Taken literally, torchsky applies the FK5 -> ICRS
        # frame bias (~29 mas here) on the way to the ICRS tile grid; SWarp ignores frame keywords altogether.
        if "gaia" in str(hd.get("ASTCAT", "")).lower():
            hd.pop("RADECSYS", None)
            hd["RADESYS"] = "ICRS"
        good = ((mk & EXCLUDE) == 0) & torch.isfinite(d) & torch.isfinite(iv) & (iv > 0)
        # masked pixels (saturated cores, bad columns, cosmics) keep a tiny weight: they only fill places where
        # every exposure is masked (as SWarp keeps saturated cores), and never compete with a good pixel
        fill = (~good) & torch.isfinite(d) & ((mk & (1 << 6)) == 0)
        zp = float(hd.get("PHOTZP", zp_target))
        scale = 10 ** (-0.4 * (zp - zp_target))
        ivm = float(iv[good].median()) if good.any() else 1.0
        img = torch.where(good | fill, (d - sky) * scale, 0.0).float()
        w = torch.where(good, iv, torch.where(fill, torch.full_like(iv, FILL_EPS * ivm), 0.0)) / scale ** 2
        w = w.float()
        obs.append(ImageObservation(data=img.cpu(), header=hd, weight_map=w.cpu()))
        meta.append({"ext": k, "photzp": zp, "sky_median": float(sky.median()), "masked_fraction": float(1 - good.float().mean()),
                     "ivar_median_zp30": float(w[good].median()) if good.any() else 0.0})
    return obs, meta


def aperture(img, x, y, r):
    """Sum in a circular aperture (pixel-centre test) + local annulus background; x,y 0-based."""
    H, W = img.shape
    R = int(np.ceil(r + 6))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    fin, cx, cy = [], [], []
    for xi, yi in zip(x, y):
        x0, y0 = int(round(xi)), int(round(yi))
        if x0 - R < 0 or y0 - R < 0 or x0 + R >= W or y0 + R >= H:
            fin.append(np.nan); cx.append(np.nan); cy.append(np.nan)
            continue
        st = img[y0 - R:y0 + R + 1, x0 - R:x0 + R + 1]
        dx, dy = xx - (xi - x0), yy - (yi - y0)
        rr = np.hypot(dx, dy)
        ann = st[(rr > r + 2) & (rr < r + 6)]
        b = np.nanmedian(ann) if ann.size else 0.0
        m = rr <= r
        s = st - b
        fin.append(np.nansum(s[m]))
        wts = np.clip(s, 0, None) * (rr <= 2.5)
        tw = wts.sum()
        cx.append((wts * (xx)).sum() / tw + x0 if tw > 0 else np.nan)
        cy.append((wts * (yy)).sum() / tw + y0 if tw > 0 else np.nan)
    return np.array(fin), np.array(cx), np.array(cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="CFIS.000.229")
    ap.add_argument("--band", default="r")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", default="lanczos3")
    ap.add_argument("--stack", default="weighted_mean")
    ap.add_argument("--coadd-device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--gpu-tile", type=int, default=1250)
    ap.add_argument("--max-exposures", type=int, default=0)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    tb = f"{a.tile}.{a.band}"

    t = time.perf_counter()
    vcp(f"vos:cfis/tiles_DR6/{tb}.expcxy", work / f"{tb}.expcxy")
    vcp(f"vos:cfis/tiles_DR6/{tb}.fits", work / f"{tb}.fits")
    vcp(f"vos:cfis/tiles_DR6/{tb}.weight.fits.fz", work / f"{tb}.weight.fits.fz")
    vcp(f"vos:cfis/tiles_DR6/{tb}.cat", work / f"{tb}.cat")
    exps, counts = contributing_exposures(work / f"{tb}.expcxy")
    hist = [str(x) for x in fits.getheader(work / f"{tb}.fits").get("HISTORY", [])]
    hexps = [m.group(1) for h in hist for m in [re.search(r"input image (\d{7})p", h)] if m]
    if hexps:
        exps = hexps
    if a.max_exposures:
        exps = exps[: a.max_exposures]
    LOG.update(tile=tb, exposures=exps, source_counts=counts)
    tick(f"contributing exposures: {exps}")
    flats = {}
    for e in exps:
        vcp(f"vos:cfis/pitcairn/{e}p.fits.fz", work / f"{e}p.fits.fz")
        fl = str(fits.getheader(work / f"{e}p.fits.fz", 1)["FLAT"]).strip()
        vcp(f"vos:sgwyn/flats/{fl}", work / fl)
        flats[e] = work / fl
    LOG["flats"] = {e: f.name for e, f in flats.items()}
    LOG["timings"]["fetch"] = round(time.perf_counter() - t, 1)
    tick("fetched")

    import torchfits
    import torchsky
    from torchsky.map.image_plane_coadd import coadd_image_plane

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG["device"] = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    LOG["torchfits_version"] = torchfits.__version__
    LOG["torch_version"] = torch.__version__

    LOG["torchsky_version"] = torchsky.__version__
    t = time.perf_counter()
    obs, meta = [], []
    for e in exps:
        wm = run_weightmask(work / f"{e}p.fits.fz", flats[e], work)
        tick(f"weightmask {e} done")
        o, m = ccd_observations(work / f"{e}p.fits.fz", wm, dev)
        obs += o
        meta += [dict(mm, exposure=e) for mm in m]
    LOG["n_ccd_observations"] = len(obs)
    LOG["timings"]["prepare"] = round(time.perf_counter() - t, 1)
    tick(f"{len(obs)} CCD observations prepared")

    ref_hdr = fits.getheader(work / f"{tb}.fits", 0)
    thead = {k: ref_hdr[k] for k in ("NAXIS", "NAXIS1", "NAXIS2", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
                                      "CD1_1", "CD1_2", "CD2_1", "CD2_2") if k in ref_hdr}
    t = time.perf_counter()
    def run_coadd(device, tile):
        return coadd_image_plane(obs, target_header=thead, resampling_kernel=a.kernel, value_semantics="flux",
                                 stacking_method=a.stack, clip_sigma=4.0, clip_iterations=2, spatial_tile_hw=(tile, tile),
                                 device=str(device))
    try:
        cdev = torch.device(a.coadd_device) if a.coadd_device != "auto" else dev
        res = run_coadd(cdev, 2500 if cdev.type == "cpu" else a.gpu_tile)
        LOG["coadd_device"] = str(cdev) + (f" ({torch.cuda.get_device_name()})" if cdev.type == "cuda" else "")
    except Exception as exc:  # noqa: BLE001  (e.g. OOM on a MIG slice)
        tick(f"coadd on {dev} failed ({exc!r}); falling back to CPU")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        t = time.perf_counter()
        res = run_coadd("cpu", 2500)
        LOG["coadd_device"] = "cpu (fallback)"
    LOG["timings"]["coadd"] = round(time.perf_counter() - t, 1)
    tick(f"torchsky coadd done in {LOG['timings']['coadd']} s on {LOG['coadd_device']}")
    ts = res.data.float().cpu().numpy()
    tw = res.weight.float().cpu().numpy() if res.weight is not None else np.isfinite(ts).astype(np.float32)
    fits.writeto(out / f"{tb}.torchsky.fits", ts.astype(np.float32), fits.Header(thead), overwrite=True)
    fits.writeto(out / f"{tb}.torchsky.weight.fits", tw.astype(np.float32), fits.Header(thead), overwrite=True)
    del obs

    # ---------------- comparison with the official tile
    ref = fits.getdata(work / f"{tb}.fits").astype(np.float32)
    with fits.open(work / f"{tb}.weight.fits.fz") as h:
        rw = (h[1].data if len(h) > 1 and h[1].data is not None else h[0].data).astype(np.float32)
    common = np.isfinite(ts) & (tw > 0) & np.isfinite(ref) & (rw > 0)
    LOG["common_fraction"] = float(common.mean())
    # background-level difference (official tile is background subtracted with a mesh)
    diff = np.where(common, ts - ref, np.nan)
    LOG["pixel"] = {
        "median_diff": float(np.nanmedian(diff)),
        "robust_sigma_diff": float(robust_sigma(diff[::7, ::7])),
        "robust_sigma_torchsky": float(robust_sigma(np.where(common, ts, np.nan)[::7, ::7])),
        "robust_sigma_official": float(robust_sigma(np.where(common, ref, np.nan)[::7, ::7])),
    }
    # catalogue: SExtractor ASCII with header lines '#   n NAME'
    cols, rows = {}, []
    for line in open(work / f"{tb}.cat", errors="replace"):
        if line.startswith("#"):
            p = line.split()
            if len(p) >= 3 and p[1].isdigit():
                cols[p[2]] = int(p[1]) - 1
            continue
        rows.append(line.split())
    LOG["catalogue_columns"] = list(cols)[:40]
    cat = np.array([[float(v) for v in r[: max(cols.values()) + 1]] for r in rows if len(r) > max(cols.values())])
    def col(name, *alts):
        for nm in (name,) + alts:
            if nm in cols:
                return cat[:, cols[nm]]
        return None
    X = col("X_IMAGE") - 1
    Y = col("Y_IMAGE") - 1
    mag = col("MAG_AUTO")
    flags = col("FLAGS")
    cls = col("CLASS_STAR")
    fwhm = col("FWHM_IMAGE")
    sel = np.isfinite(X) & (mag < 23.5) & (mag > 16.5)
    if flags is not None:
        sel &= flags == 0
    X, Y, mag = X[sel], Y[sel], mag[sel]
    star = (cls[sel] > 0.95) if cls is not None else np.ones_like(mag, bool)
    rng = np.random.default_rng(0)
    if X.size > 20000:
        k = rng.choice(X.size, 20000, replace=False)
        X, Y, mag, star = X[k], Y[k], mag[k], star[k]
    r_ap = 2.0 / 0.1858  # 2" radius
    f_ts, cx_ts, cy_ts = aperture(ts, X, Y, r_ap)
    f_rf, cx_rf, cy_rf = aperture(ref, X, Y, r_ap)
    ok = np.isfinite(f_ts) & np.isfinite(f_rf) & (f_rf > 0) & (f_ts > 0)
    dm = -2.5 * np.log10(f_ts[ok] / f_rf[ok])
    m_ok = mag[ok]
    bins = [16.5, 18, 19, 20, 21, 22, 23.5]
    phot = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = (m_ok >= lo) & (m_ok < hi)
        if b.sum() > 10:
            phot.append({"mag_lo": lo, "mag_hi": hi, "n": int(b.sum()), "median_dmag": float(np.median(dm[b])),
                         "nmad_dmag": float(1.4826 * np.median(np.abs(dm[b] - np.median(dm[b]))))})
    LOG["photometry_2arcsec"] = phot
    bright = ok & (mag < 20) & star
    dx = (cx_ts - cx_rf)[bright] * 0.1858 * 1000
    dy = (cy_ts - cy_rf)[bright] * 0.1858 * 1000
    LOG["astrometry_mas"] = {"n": int(bright.sum()), "median_dx": float(np.nanmedian(dx)), "median_dy": float(np.nanmedian(dy)),
                             "nmad_dx": float(1.4826 * np.nanmedian(np.abs(dx - np.nanmedian(dx)))),
                             "nmad_dy": float(1.4826 * np.nanmedian(np.abs(dy - np.nanmedian(dy))))}
    np.savez_compressed(out / f"{tb}.compare.npz", mag=m_ok, dmag=dm, dx=dx, dy=dy)

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stretch(im, s):
        return np.arcsinh(im / s)

    s0 = LOG["pixel"]["robust_sigma_official"]
    cy0, cx0 = 5000, 5000
    # pick a crop centred on the brightest non-saturated galaxy-ish source near the centre
    half = 400
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="#0b1020")
    for ax, im, ttl in zip(axs, [ref, ts, ts - ref], ["official MegaPipe/SWarp tile", "torchsky coadd (same exposures)", "torchsky − official"]):
        c = im[cy0 - half:cy0 + half, cx0 - half:cx0 + half]
        if "−" in ttl:
            ax.imshow(c, origin="lower", cmap="RdBu_r", vmin=-5 * s0, vmax=5 * s0)
        else:
            ax.imshow(stretch(c, 3 * s0), origin="lower", cmap="gray", vmin=-1, vmax=6)
        ax.set_title(ttl, color="w")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"{tb}.compare_cutouts.png", dpi=110, facecolor="#0b1020")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="w")
    axs[0].scatter(m_ok, dm, s=2, alpha=0.3, color="k")
    axs[0].axhline(0, color="r")
    axs[0].set_ylim(-0.3, 0.3)
    axs[0].set_xlabel("MAG_AUTO (official catalogue)")
    axs[0].set_ylabel("m(torchsky) − m(official), 2″ aperture")
    axs[1].scatter(dx, dy, s=3, alpha=0.4)
    axs[1].set_xlabel("Δx [mas]")
    axs[1].set_ylabel("Δy [mas]")
    axs[1].set_xlim(-150, 150)
    axs[1].set_ylim(-150, 150)
    axs[1].set_title(f"centroid offsets, {bright.sum()} stars r<20")
    fig.tight_layout()
    fig.savefig(out / f"{tb}.compare_stats.png", dpi=110)
    json.dump(LOG, open(out / f"{tb}.compare.json", "w"), indent=1, default=float)
    json.dump(meta, open(out / f"{tb}.ccd_meta.json", "w"), indent=1, default=float)
    tick(json.dumps({k: LOG[k] for k in ("pixel", "astrometry_mas")}, default=float))
    tick("done")


if __name__ == "__main__":
    main()
