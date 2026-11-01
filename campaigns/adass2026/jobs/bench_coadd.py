"""Speed of torchsky coadds against the legacy codes on the same SMACS J0723 NIRCam exposures.

usage: bench_coadd.py --data-dir DIR --work /scratch/b --out OUT [--threads 8]

Per band (6 NIRCam bands, 123 exposures) onto two TAN grids: the slide grid (512 x 512 at 0.04") and the full
module-B field at 0.04". Inputs are first written once to local scratch as plain science / weight FITS (median sky
removed, DQ-masked inverse variance), so every code reads the same files.

  torchsky SWarp-like : coadd_image_plane(lanczos3, weighted_mean), CPU and GPU, files read with torchfits
  SWarp 2.41          : RESAMPLING_TYPE LANCZOS3, COMBINE_TYPE WEIGHTED, MAP_WEIGHT, no background, NTHREADS
  torchsky Drizzle-like: coadd_image_plane(drizzle_square, pixfrac 0.8), CPU and GPU
  drizzle (STScI)     : drizzle.resample.Drizzle square kernel, pixfrac 0.8, pixmap from astropy WCS (with SIP)

Timings are wall clock, best of --repeat, including reading the input files. SWarp has no SIP, so its inputs carry
the same distortion converted to TPV (sip_tpv); torchsky and drizzle read the SIP headers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
LOG: dict = {"timings": {}, "agreement": {}}
T0 = time.time()


def tick(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def tan_header(ra, dec, nx, ny, scale):
    s = scale / 3600.0
    return {"NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": ra, "CRVAL2": dec,
            "CRPIX1": (nx + 1) / 2, "CRPIX2": (ny + 1) / 2, "CD1_1": -s, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s,
            "RADESYS": "ICRS", "EQUINOX": 2000.0}


def prepare(data_dir, work):
    """Write sci_*.fits / wht_*.fits per exposure (median-subtracted science, DQ-masked inverse variance)."""
    from astropy.io import fits  # only to build the legacy inputs; torchsky reads them with torchfits
    from astropy.wcs import WCS
    work.mkdir(parents=True, exist_ok=True)
    items = []
    corners = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.fits"))):
        h0 = fits.getheader(f, 0)
        band = next((v for v in (str(h0.get("FILTER", "")).upper(), str(h0.get("PUPIL", "")).upper()) if v in BANDS), None)
        if band is None:
            continue
        with fits.open(f, memmap=False) as h:
            sh = h["SCI"].header
            sci = h["SCI"].data.astype(np.float32)
            err = h["ERR"].data.astype(np.float32)
            dq = h["DQ"].data
        ok = np.isfinite(sci) & np.isfinite(err) & (err > 0) & (dq == 0)
        med = float(np.median(sci[ok]))
        s = np.where(ok, sci - med, 0).astype(np.float32)
        w = np.where(ok, 1.0 / np.where(ok, err, 1) ** 2, 0).astype(np.float32)
        hd = fits.Header()
        for k in sh:
            if k and k not in ("COMMENT", "HISTORY", "") and not k.startswith(("XTENSION", "PCOUNT", "GCOUNT", "EXTNAME")):
                try:
                    hd[k] = sh[k]
                except Exception:  # noqa: BLE001
                    pass
        hd["EXPTIME"] = 1.0
        stem = Path(f).stem
        fs, fw = work / f"sci_{stem}.fits", work / f"wht_{stem}.fits"
        fits.writeto(fs, s, hd, overwrite=True)
        fits.writeto(fw, w, hd, overwrite=True)
        # SWarp has no SIP: give it the same images with the SIP distortion converted exactly to TPV (sip_tpv)
        hs = hd.copy()
        try:
            from sip_tpv import sip_to_pv
            sip_to_pv(hs)
            swarp_wcs = "TPV"
        except Exception as exc:  # noqa: BLE001
            for k in list(hs.keys()):
                if k.startswith(("A_", "B_", "AP_", "BP_")):
                    del hs[k]
            hs["CTYPE1"], hs["CTYPE2"] = "RA---TAN", "DEC--TAN"
            swarp_wcs = f"TAN only ({exc!r})"
        fss, fws = work / f"swsci_{stem}.fits", work / f"swwht_{stem}.fits"
        fits.writeto(fss, s, hs, overwrite=True)
        fits.writeto(fws, w, hs, overwrite=True)
        LOG["swarp_input_wcs"] = swarp_wcs
        ny, nx = s.shape
        corners.append(WCS(hd).all_pix2world([0, nx - 1, nx - 1, 0], [0, 0, ny - 1, ny - 1], 0))
        items.append({"band": band, "sci": str(fs), "wht": str(fw), "swsci": str(fss), "swwht": str(fws), "shape": s.shape})
    ra = np.concatenate([c[0] for c in corners])
    dec = np.concatenate([c[1] for c in corners])
    ra = np.where(ra > 180, ra - 360, ra) if ra.max() - ra.min() > 180 else ra
    return items, (float(ra.min()), float(ra.max()), float(dec.min()), float(dec.max()))


def read_obs(items, band):
    import torchfits
    from torchsky.map.observation import ImageObservation
    obs = []
    for it in items:
        if it["band"] != band:
            continue
        d, h = torchfits.read(it["sci"], hdu=0, return_header=True)
        w = torchfits.read(it["wht"], hdu=0)
        hd = {k: h[k] for k in h.keys() if k and k not in ("COMMENT", "HISTORY", "")}
        obs.append(ImageObservation(data=d.float(), header=hd, weight_map=w.float()))
    return obs


def run_torchsky(items, thead, kernel, pixfrac, device):
    from torchsky.map.image_plane_coadd import coadd_image_plane
    outs = []
    t_read = t_comp = 0.0
    for b in BANDS:
        t = time.perf_counter()
        obs = read_obs(items, b)
        t_read += time.perf_counter() - t
        t = time.perf_counter()
        res = coadd_image_plane(obs, target_header=thead, resampling_kernel=kernel, drizzle_pixfrac=pixfrac,
                                stacking_method="weighted_mean", device=device)
        img = res.data.float()
        if device == "cuda":
            torch.cuda.synchronize()
        t_comp += time.perf_counter() - t
        outs.append(img.cpu().numpy())
    return np.stack(outs), t_read, t_comp


def run_swarp(items, thead, work, threads, tag):
    from astropy.io import fits
    outs = []
    t_tot = 0.0
    d = work / f"swarp_{tag}"
    d.mkdir(exist_ok=True)
    for b in BANDS:
        lst = [it for it in items if it["band"] == b]
        (d / f"{b}.lst").write_text("\n".join(it["swsci"] for it in lst) + "\n")
        (d / f"{b}.wlst").write_text("\n".join(it["swwht"] for it in lst) + "\n")
        out = d / f"{b}.fits"
        hdr = fits.Header({k: v for k, v in thead.items() if k != "NAXIS"})
        (d / f"{b}.head").write_text(hdr.tostring(sep="\n", endcard=True, padding=False) + "\n")
        cmd = ["swarp", f"@{d / (b + '.lst')}", "-c", "/dev/null", "-IMAGEOUT_NAME", str(out), "-WEIGHTOUT_NAME", str(d / f"{b}.weight.fits"),
               "-WEIGHT_TYPE", "MAP_WEIGHT", "-WEIGHT_IMAGE", f"@{d / (b + '.wlst')}", "-COMBINE", "Y", "-COMBINE_TYPE", "WEIGHTED",
               "-RESAMPLE", "Y", "-RESAMPLING_TYPE", "LANCZOS3", "-SUBTRACT_BACK", "N", "-FSCALASTRO_TYPE", "NONE",
               "-CENTER_TYPE", "MANUAL", "-CENTER", f"{thead['CRVAL1']},{thead['CRVAL2']}",
               "-PIXELSCALE_TYPE", "MANUAL", "-PIXEL_SCALE", str(thead["CD2_2"] * 3600), "-IMAGE_SIZE", f"{thead['NAXIS1']},{thead['NAXIS2']}",
               "-PROJECTION_TYPE", "TAN", "-RESAMPLE_DIR", str(d), "-DELETE_TMPFILES", "Y", "-NTHREADS", str(threads),
               "-VERBOSE_TYPE", "QUIET", "-WRITE_XML", "N", "-GAIN_DEFAULT", "0"]
        t = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        t_tot += time.perf_counter() - t
        if r.returncode != 0:
            raise RuntimeError(f"swarp failed {b}: {r.stderr[-800:]}")
        outs.append(fits.getdata(out).astype(np.float32))
    return np.stack(outs), t_tot


def run_drizzle(items, thead, pixfrac):
    from astropy.io import fits
    from astropy.wcs import WCS
    from drizzle.resample import Drizzle
    from drizzle.utils import calc_pixmap
    owcs = WCS(fits.Header(thead))
    shape = (thead["NAXIS2"], thead["NAXIS1"])
    outs = []
    t_tot = 0.0
    for b in BANDS:
        t = time.perf_counter()
        dz = Drizzle(kernel="square", out_shape=shape, disable_ctx=True)
        for it in items:
            if it["band"] != b:
                continue
            with fits.open(it["sci"], memmap=False) as h:
                s = h[0].data.astype(np.float32)
                iw = WCS(h[0].header)
            w = fits.getdata(it["wht"]).astype(np.float32)
            pm = calc_pixmap(iw, owcs, shape=s.shape)
            dz.add_image(s, exptime=1.0, pixmap=pm, weight_map=w, pixfrac=pixfrac, in_units="cps")
        t_tot += time.perf_counter() - t
        outs.append(np.asarray(dz.out_img, np.float32))
    return np.stack(outs), t_tot


def agree(a, b):
    """Robust difference between two coadds, in units of the robust scatter of a (common finite, non-zero support)."""
    m = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if m.sum() < 100:
        return None
    sa = 1.4826 * np.median(np.abs(a[m] - np.median(a[m])))
    d = (a[m] - b[m]) / max(sa, 1e-30)
    return {"median": float(np.median(d)), "nmad": float(1.4826 * np.median(np.abs(d - np.median(d)))), "n": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/arc/projects/ots/torchsky-data/jwst_smacs0723/aligned")
    ap.add_argument("--work", default="/scratch/bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--grids", default="slide,full")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    import torchsky, torchfits
    LOG["versions"] = {"torch": torch.__version__, "torchsky": torchsky.__version__, "torchfits": torchfits.__version__,
                       "swarp": subprocess.run(["swarp", "-v"], capture_output=True, text=True).stdout.strip()}
    try:
        import drizzle
        LOG["versions"]["drizzle"] = drizzle.__version__
    except Exception:  # noqa: BLE001
        pass
    LOG["cpu_threads"] = a.threads
    LOG["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    t = time.perf_counter()
    items, (ra0, ra1, de0, de1) = prepare(a.data_dir, work)
    tick(f"prepared {len(items)} exposures in {time.perf_counter() - t:.0f} s")
    LOG["n_exposures"] = len(items)
    LOG["input_megapixels"] = float(sum(np.prod(it["shape"]) for it in items) / 1e6)
    grids = {}
    if "slide" in a.grids:
        grids["slide"] = tan_header(110.8268, -73.4544, 512, 512, 0.04)
    if "full" in a.grids:
        dec_c = 0.5 * (de0 + de1)
        ra_c = 0.5 * (ra0 + ra1)
        nx = int(np.ceil((ra1 - ra0) * np.cos(np.radians(dec_c)) * 3600 / 0.04)) + 20
        ny = int(np.ceil((de1 - de0) * 3600 / 0.04)) + 20
        grids["full"] = tan_header(ra_c % 360, dec_c, nx, ny, 0.04)
    for gname, th in grids.items():
        tick(f"grid {gname}: {th['NAXIS1']} x {th['NAXIS2']} at 0.04 arcsec")
        res = {}
        T = LOG["timings"].setdefault(gname, {"shape": [th["NAXIS2"], th["NAXIS1"]]})
        jobs = [("torchsky_swarp_cpu", lambda: run_torchsky(items, th, "lanczos3", 1.0, "cpu")),
                ("torchsky_drizzle_cpu", lambda: run_torchsky(items, th, "drizzle_square", 0.8, "cpu"))]
        if torch.cuda.is_available():
            jobs += [("torchsky_swarp_gpu", lambda: run_torchsky(items, th, "lanczos3", 1.0, "cuda")),
                     ("torchsky_drizzle_gpu", lambda: run_torchsky(items, th, "drizzle_square", 0.8, "cuda"))]
        jobs += [("swarp", lambda: run_swarp(items, th, work, a.threads, gname)),
                 ("drizzle_stsci", lambda: run_drizzle(items, th, 0.8))]
        for name, fn in jobs:
            best = None
            for rep in range(a.repeat):
                try:
                    r = fn()
                except Exception as exc:  # noqa: BLE001
                    tick(f"{gname} {name} failed: {exc!r}")
                    best = None
                    break
                img = r[0]
                tt = sum(r[1:])
                rec = {"total_s": tt}
                if len(r) == 3:
                    rec.update(read_s=r[1], compute_s=r[2])
                if best is None or tt < best["total_s"]:
                    best = rec
                res[name] = img
                tick(f"{gname} {name} run {rep}: {tt:.1f} s {rec}")
            if best is not None:
                T[name] = best
            json.dump(LOG, open(out / "bench.json", "w"), indent=1, default=float)
        for pa, pb in (("torchsky_swarp_cpu", "swarp"), ("torchsky_drizzle_cpu", "drizzle_stsci"),
                       ("torchsky_swarp_cpu", "torchsky_swarp_gpu"), ("torchsky_drizzle_cpu", "torchsky_drizzle_gpu")):
            if pa in res and pb in res:
                LOG["agreement"].setdefault(gname, {})[f"{pa}_vs_{pb}"] = {b: agree(res[pa][k], res[pb][k]) for k, b in enumerate(BANDS)}
        json.dump(LOG, open(out / "bench.json", "w"), indent=1, default=float)
    tick("done")


if __name__ == "__main__":
    main()
