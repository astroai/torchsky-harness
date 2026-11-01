"""Diagnose the UNIONS CPU coadd time across torchsky versions and tiling modes.

prep: fetch + weightmask N exposures with the v5 pipeline, save per-CCD (data, weight, header) to --work.
run:  rebuild ImageObservations, coadd onto the official tile grid (Lanczos-3, flux, weighted mean) with
      --tile T or --auto, under cProfile; write timing, tile plan and top functions to --out/<label>.json and the
      coadd to --work/<label>.npy for cross-checks.
cmp:  compare the saved coadds.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch


def prep(a):
    sys.path.insert(0, a.scripts)
    import unions_coadd_v5 as v5
    from astropy.io import fits

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tb = f"{a.tile}.{a.band}"
    v5.vcp(f"vos:cfis/tiles_DR6/{tb}.expcxy", work / f"{tb}.expcxy")
    v5.vcp(f"vos:cfis/tiles_DR6/{tb}.fits", work / f"{tb}.fits")
    hist = [str(x) for x in fits.getheader(work / f"{tb}.fits").get("HISTORY", [])]
    import re
    exps = [m.group(1) for h in hist for m in [re.search(r"input image (\d{7})p", h)] if m]
    if not exps:
        exps, _ = v5.contributing_exposures(work / f"{tb}.expcxy")
    exps = exps[: a.max_exposures]
    print("exposures", exps, flush=True)
    k = 0
    for e in exps:
        v5.vcp(f"vos:cfis/pitcairn/{e}p.fits.fz", work / f"{e}p.fits.fz")
        fl = str(fits.getheader(work / f"{e}p.fits.fz", 1)["FLAT"]).strip()
        v5.vcp(f"vos:sgwyn/flats/{fl}", work / fl)
        wm = v5.run_weightmask(work / f"{e}p.fits.fz", work / fl, work)
        obs, _ = v5.ccd_observations(work / f"{e}p.fits.fz", wm, torch.device("cpu"))
        for o in obs:
            torch.save({"data": o.data, "weight": o.weight_map, "header": dict(o.header)}, work / f"ccd_{k:04d}.pt")
            k += 1
        print(f"{e}: {len(obs)} CCDs", flush=True)
    ref = fits.getheader(work / f"{tb}.fits", 0)
    thead = {kk: ref[kk] for kk in ("NAXIS", "NAXIS1", "NAXIS2", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1",
                                    "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2") if kk in ref}
    (work / "thead.json").write_text(json.dumps(thead))
    print(f"prepared {k} CCDs", flush=True)


def run(a):
    import torchsky
    from torchsky.map.image_plane_coadd import coadd_image_plane
    from torchsky.map.observation import ImageObservation

    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    thead = json.loads((work / "thead.json").read_text())
    obs = []
    for f in sorted(work.glob("ccd_*.pt")):
        d = torch.load(f, weights_only=False)
        obs.append(ImageObservation(data=d["data"], header=d["header"], weight_map=d["weight"]))
    if a.n_ccd:
        obs = obs[: a.n_ccd]
    rec = {"label": a.label, "torchsky": torchsky.__version__, "torch": torch.__version__,
           "threads": torch.get_num_threads(), "n_ccd": len(obs), "tile": None if a.auto else a.tile_edge,
           "source_mpix": sum(o.data.numel() for o in obs) / 1e6}
    try:
        from torchsky.map.operators.execution import plan_spatial_tiles
        plan = plan_spatial_tiles((thead["NAXIS2"], thead["NAXIS1"]), resampling_kernel="lanczos3",
                                  value_semantics="flux", n_observations=len(obs),
                                  source_pixels=sum(o.data.numel() for o in obs), device="cpu")
        rec["auto_plan"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in plan.__dict__.items()}
    except Exception as exc:  # noqa: BLE001  (older torchsky has no planner)
        rec["auto_plan"] = repr(exc)
    kw = dict(target_header=thead, resampling_kernel="lanczos3", value_semantics="flux",
              stacking_method="weighted_mean", clip_sigma=4.0, clip_iterations=2, device="cpu")
    kw["spatial_tile_hw"] = None if a.auto else (a.tile_edge, a.tile_edge)
    if a.no_variance:
        kw["propagate_variance"] = False
    pr = cProfile.Profile()
    t = time.perf_counter()
    pr.enable()
    res = coadd_image_plane(obs, **kw)
    pr.disable()
    rec["coadd_s"] = time.perf_counter() - t
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(45)
    rec["profile_cumulative"] = s.getvalue().splitlines()[:80]
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
    rec["profile_tottime"] = s.getvalue().splitlines()[:60]
    np.save(work / f"{a.label}.npy", res.data.float().cpu().numpy())
    (out / f"{a.label}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"{a.label}: coadd {rec['coadd_s']:.1f} s, plan {rec['auto_plan']}", flush=True)


def cmp(a):
    work = Path(a.work)
    arrs = {p.stem: np.load(p, mmap_mode="r") for p in sorted(work.glob("*.npy"))}
    keys = list(arrs)
    rep = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = arrs[keys[i]][::3, ::3], arrs[keys[j]][::3, ::3]
            m = np.isfinite(x) & np.isfinite(y) & (x != 0)
            d = (x - y)[m]
            rep[f"{keys[i]}_vs_{keys[j]}"] = {"max_abs": float(np.abs(d).max()) if d.size else None,
                                              "rms": float(np.sqrt((d ** 2).mean())) if d.size else None,
                                              "ref_rms": float(np.sqrt((x[m] ** 2).mean())) if d.size else None,
                                              "support_diff": int((np.isfinite(x) & (x != 0)).sum() - (np.isfinite(y) & (y != 0)).sum())}
    Path(a.out, "compare.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "run", "cmp"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--scripts", default="/arc/projects/ots/adass2026/scripts")
    ap.add_argument("--tile", default="CFIS.000.229")
    ap.add_argument("--band", default="r")
    ap.add_argument("--max-exposures", type=int, default=2)
    ap.add_argument("--label", default="run")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--tile-edge", type=int, default=2500)
    ap.add_argument("--n-ccd", type=int, default=0)
    ap.add_argument("--no-variance", action="store_true")
    a = ap.parse_args()
    {"prep": prep, "run": run, "cmp": cmp}[a.mode](a)
