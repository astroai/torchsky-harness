"""Slide figures for the UNIONS tile comparison (dark theme). usage: make_unions_figs.py RUNDIR OFFICIAL_TILE TAG"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

run, offl, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
A = Path("/home/claude/adass/deck/assets")
tb = offl.name.replace(".fits", "")
L = json.load(open(run / f"{tb}.compare.json"))
Z = np.load(run / f"{tb}.compare.npz")
ts = fits.getdata(run / f"{tb}.torchsky.fits", memmap=True)
rf = fits.getdata(offl, memmap=True)
SAT = 16.5  # lower edge of the comparison (saturated cores are kept, as in the official stack)
BG, FG, MU = "#0B1020", "#F4F6FB", "#AEB6C8"
plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": MU, "ytick.color": MU, "axes.edgecolor": MU, "font.size": 13})
cy, cx, half = int(sys.argv[4]) if len(sys.argv) > 4 else 5000, int(sys.argv[5]) if len(sys.argv) > 5 else 5000, 450
a = np.asarray(rf[cy - half:cy + half, cx - half:cx + half], np.float32)
b = np.asarray(ts[cy - half:cy + half, cx - half:cx + half], np.float32)
s0 = 1.4826 * np.nanmedian(np.abs(a - np.nanmedian(a)))
fig, axs = plt.subplots(1, 3, figsize=(15, 5.3), facecolor=BG)
for ax, im, ttl in zip(axs, [a, b, b - a], ["official UNIONS tile (MegaPipe/SWarp)", "torchsky coadd, same exposures", "torchsky − official"]):
    if "−" in ttl:
        ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-4 * s0, vmax=4 * s0, interpolation="nearest")
    else:
        ax.imshow(np.arcsinh(im / (2 * s0)), origin="lower", cmap="gray", vmin=-1, vmax=6, interpolation="nearest")
    ax.set_title(ttl, color=FG, fontsize=15)
    ax.axis("off")
fig.tight_layout()
fig.savefig(A / f"unions_cutouts{tag}.png", dpi=110, facecolor=BG)
plt.close(fig)
m, dm, dx, dy = Z["mag"], Z["dmag"], Z["dx"], Z["dy"]
fig, axs = plt.subplots(2, 1, figsize=(6.2, 7.6), facecolor=BG, gridspec_kw=dict(height_ratios=[1.15, 1]))
ax = axs[0]
ax.set_facecolor(BG)
ax.scatter(m, dm * 1000, s=1.5, alpha=0.25, color=MU, rasterized=True)
bins = np.arange(16.5, 24.01, 0.5)
mid, med, lo, hi = [], [], [], []
for l, h in zip(bins[:-1], bins[1:]):
    k = (m >= l) & (m < h)
    if k.sum() > 10:
        q = np.percentile(dm[k] * 1000, [16, 50, 84])
        mid.append((l + h) / 2); lo.append(q[0]); med.append(q[1]); hi.append(q[2])
ax.fill_between(mid, lo, hi, color="#4CC9F0", alpha=0.25, lw=0)
ax.plot(mid, med, color="#F5A524", lw=2.5)
ax.axhline(0, color=FG, lw=0.8)
ax.set_xlim(16.3, 23.7)
ax.set_ylim(-60, 60)
ax.set_xlabel("r (official catalogue MAG_AUTO)")
ax.set_ylabel("Δm, 2″ aperture [mmag]")
ax = axs[1]
ax.set_facecolor(BG)
import os
GJ, GN = os.environ.get("GAIA_JSON"), os.environ.get("GAIA_NPZ")
gaia = None
if GJ and GN:
    G = json.load(open(GJ))["bands"][tb.split(".")[-1]]
    Z2 = np.load(GN)
    pxg, pyg, pix = Z2["px"], Z2["py"], G["pixel_mas"]
    for nm, col, lab in (("x_off", MU, "official tile"), ("x_ts", "#4CC9F0", "torchsky coadd")):
        de = -(Z2[nm] - pxg) * pix
        dn = (Z2[nm.replace("x_", "y_")] - pyg) * pix
        ax.scatter(de, dn, s=4, alpha=0.35, color=col, label=lab, rasterized=True)
    for nm, col in (("official", MU), ("torchsky", "#F5A524")):
        ax.plot([G[nm]["median_east_mas"]], [G[nm]["median_north_mas"]], marker="+", ms=16, mew=2.5, color=col)
    ax.axhline(0, color=MU, lw=0.6); ax.axvline(0, color=MU, lw=0.6)
    ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_aspect("equal")
    ax.set_xlabel("east [mas]"); ax.set_ylabel("north [mas]")
    ax.set_title(f"centroid − Gaia DR3, {G['torchsky']['n']} stars G 17–20.5", fontsize=13, color=FG)
    ax.legend(loc="lower left", fontsize=10, frameon=False, labelcolor=FG, markerscale=3)
    t, o, d = G["torchsky"], G["official"], G["torchsky_minus_official"]
    off = float(np.hypot(t["median_east_mas"], t["median_north_mas"]))
    gaia = dict(value=f"{off:.0f} mas" if off >= 1 else "< 1 mas",
                label=f"median offset from Gaia DR3, {t['n']} stars (official tile: {np.hypot(o['median_east_mas'], o['median_north_mas']):.0f} mas); {max(d['nmad_east_mas'], d['nmad_north_mas']):.1f} mas scatter vs official",
                torchsky=t, official=o, diff=d, epoch=G["epoch"])
else:
    ax.scatter(dx, dy, s=5, alpha=0.5, color="#4CC9F0")
    ax.axhline(0, color=MU, lw=0.6); ax.axvline(0, color=MU, lw=0.6)
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect("equal")
    ax.set_xlabel("Δx [mas]"); ax.set_ylabel("Δy [mas]")
    ax.set_title(f"centroids of {len(dx)} stars, r < 20", fontsize=13, color=FG)
fig.tight_layout()
fig.savefig(A / f"unions_stats{tag}.png", dpi=120, facecolor=BG)
plt.close(fig)
asm = L["astrometry_mas"]; px = L["pixel"]
ph = []
for lo_, hi_ in zip([SAT, 18, 19, 20, 21, 22], [18, 19, 20, 21, 22, 23.5]):
    k = (m >= lo_) & (m < hi_)
    if k.sum() > 10:
        md = float(np.median(dm[k]))
        ph.append({"mag_lo": lo_, "mag_hi": hi_, "n": int(k.sum()), "median_dmag": md, "nmad_dmag": float(1.4826 * np.median(np.abs(dm[k] - md)))})
bright = [p for p in ph if p["mag_hi"] <= 20]
out = dict(cut_aspect=5.3 / 15, stats_aspect=7.6 / 6.2,
           caption=(f"{len(L['exposures'])} individual MegaCam exposures → weightmask (master-flat bad pixels, cosmic rays, inverse variance, sky) → torchsky Lanczos-3 weighted coadd on the official TAN grid. "
                    f"Photometry, r = 16.5–23.5: |median Δm| ≤ {1000 * max(abs(p['median_dmag']) for p in ph):.1f} mmag in every bin, scatter {1000 * max(p['nmad_dmag'] for p in bright):.0f} mmag at r < 20. " +
                    (f"Astrometry against Gaia DR3 propagated to the epoch {gaia['epoch']:.1f}: torchsky ({gaia['torchsky']['median_east_mas']:.1f}, {gaia['torchsky']['median_north_mas']:.1f}) mas east, north; official ({gaia['official']['median_east_mas']:.1f}, {gaia['official']['median_north_mas']:.1f}) mas." if gaia else f"Astrometry: offset ({asm['median_dx']:.0f}, {asm['median_dy']:.0f}) mas, scatter {asm['nmad_dx']:.0f} mas.")),
           source=f"UNIONS {L['tile']} (tiles_DR6) vs torchsky {L.get('torchsky_version', '')} + torchfits {L.get('torchfits_version', '')} + weightmask 0.3.0; saturated cores kept as in the official stack. Outputs in vos:sfabbro/adass2026/" + os.environ.get("UNIONS_RUN", "unions_coadd_v3") + ".",
           notes=("~50 s. A real-survey check of the classical path. We took the seven MegaCam exposures that make up one UNIONS tile, the individual calibrated exposures from the archive. "
                  "weightmask, another of our packages, builds per-exposure masks from the master flat, flags saturation and cosmic rays, and computes an inverse variance and a sky. "
                  "torchsky then coadds them with Lanczos-3 onto the official tile grid, which is how the survey's own SWarp stack is made. " +
                  ("Against the official tile: aperture photometry agrees to a few millimagnitudes at every magnitude. Astrometry is checked against Gaia DR3, propagated to the epoch of the exposures, with windowed centroids on both images: "
                   f"our coadd sits {np.hypot(gaia['torchsky']['median_east_mas'], gaia['torchsky']['median_north_mas']):.1f} mas from Gaia, the official tile {np.hypot(gaia['official']['median_east_mas'], gaia['official']['median_north_mas']):.1f} mas. "
                   "A first run was off by 29 mas east and 9 north: exactly the FK5 to ICRS frame bias. The MegaCam CCD headers still declare the old FK5 frame although the astrometry was solved against Gaia, so torchsky faithfully applied the FK5 to ICRS rotation; SWarp ignores frame keywords, so it never sees this. torchsky now warns when it does that conversion. "
                   if gaia else f"Against the official tile: aperture photometry agrees to a few millimagnitudes at every magnitude; star centroids agree to {asm['nmad_dx']:.0f} mas scatter, with a {asm['median_dx']:.0f} mas offset in x that we are checking against Gaia DR3. ") +
                  "The difference image is noise plus residual background differences. This is the boring, necessary result: the classical coadd is reproduced before we go beyond it. "
                  f"(If asked: the thin dark line along one column, about -0.6 sigma, is not yet diagnosed; it is not resampling ringing, a synthetic masked-column test is flat, so it is most likely partially bad pixels next to the flat-based mask. The CPU coadd took {L['timings']['coadd'] / 60:.0f} min for 280 CCDs into a 10k x 10k tile; the GPU run ran out of memory on a 12 GB slice because of a full-size bookkeeping mask, now fixed in torchsky.)"),
           pixel=px, photometry=ph, astrometry=asm, gaia=gaia)
json.dump(out, open(A / f"unions_compare{tag}.json", "w"), indent=1)
print("ok", out["caption"])
