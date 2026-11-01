"""Slide figure for the continuous sky s(x, y, lambda): monochromatic renders of the arc + SEDs vs band data."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from astropy.io import fits

run, out = Path(sys.argv[1]), Path(sys.argv[2])
name = sys.argv[3] if len(sys.argv) > 3 else "implicit_chrom_3d"
M = json.load(open(run / "metrics.json"))
lam = np.array(M["spectral_cube_lam_um"])
cube = fits.getdata(run / "cubes" / f"{name}_spectral.fits").astype(np.float32)  # L H W (FITS rows: y up)
band = fits.getdata(run / "cubes" / f"{name}_sky.fits").astype(np.float32)  # model band averages
driz = fits.getdata(run / "cubes" / "drizzle.fits").astype(np.float32)
nodes = M["bandpass_nodes_um"]
curves = M["bandpass_curves"]
BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
H = cube.shape[-1]
BG, FG, MU = "#0B1020", "#F4F6FB", "#AEB6C8"
plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": MU, "ytick.color": MU, "axes.edgecolor": MU, "font.size": 13})

# display coords (north up) -> FITS rows
def fits_yx(yd, xd):
    return H - 1 - yd, xd

# arc box (display) as in the zoom slide
x0, yd0, s = 296, 0, 160
arc = cube[:, H - (yd0 + s):H - yd0, x0:x0 + s][:, ::-1]  # back to display orientation
pick = [0.9, 1.5, 2.2, 3.2, 4.4]
idx = [int(np.argmin(np.abs(lam - p))) for p in pick]
fig = plt.figure(figsize=(16, 7.2), facecolor=BG)
for j, k in enumerate(idx):
    ax = fig.add_axes([0.01 + j * 0.105, 0.52, 0.1, 0.44])
    im = arc[k]
    sig = 1.4826 * np.median(np.abs(im - np.median(im)))
    ax.imshow(np.arcsinh((im - np.median(im)) / (3 * sig)), cmap="magma", vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax.set_title(f"{lam[k]:.2f} µm", color=FG, fontsize=13)
    ax.axis("off")
# positions: arc knot, BCG core, bluest bright compact source
sky_sum = band.sum(0)
yk, xk = np.unravel_index(np.argmax(sky_sum[H - (yd0 + s):H - yd0, x0:x0 + s]), (s, s))
pos = {"lensed arc": (H - (yd0 + s) + yk, x0 + xk)}
c = H // 2
yb, xb = np.unravel_index(np.argmax(sky_sum[c - 60:c + 60, c - 60:c + 60]), (120, 120))
pos["BCG core"] = (c - 60 + yb, c - 60 + xb)
bright = sky_sum > np.percentile(sky_sum, 99.0)
ratio = np.where(bright, band[0] / np.maximum(band[5], 1e-6), -np.inf)
yr, xr = np.unravel_index(np.argmax(ratio), ratio.shape)
pos["compact source"] = (yr, xr)
cols = {"lensed arc": "#F5A524", "BCG core": "#E4572E", "compact source": "#4CC9F0"}
ax = fig.add_axes([0.58, 0.14, 0.4, 0.8], facecolor=BG)
for lab, (y, x) in pos.items():
    sl = (slice(y - 1, y + 2), slice(x - 1, x + 2))
    sed = cube[:, sl[0], sl[1]].mean((1, 2))
    ax.plot(lam, sed, color=cols[lab], lw=2.2, label=lab)
    bm = band[:, sl[0], sl[1]].mean((1, 2))
    bd = driz[:, sl[0], sl[1]].mean((1, 2))
    lc = [float(np.mean(nodes[b])) for b in BANDS]
    ax.scatter(lc, bm, color=cols[lab], s=40, zorder=3, edgecolor=BG)
    ax.scatter(lc, bd, facecolor="none", edgecolor=cols[lab], s=90, lw=1.3, zorder=3)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks([0.9, 1.5, 2, 3, 4.4])
ax.set_xticklabels(["0.9", "1.5", "2", "3", "4.4"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("wavelength [µm]")
ax.set_ylabel("surface brightness [MJy/sr]")
ylo, yhi = ax.get_ylim()
ax2 = ax.twinx()
for b in BANDS:
    cv = curves[b]
    ax2.fill_between(cv["lam_um"], 0, cv["T"], color=MU, alpha=0.12, lw=0)
ax2.set_ylim(0, 4)
ax2.set_yticks([])
leg = ax.legend(loc="upper left", frameon=False, fontsize=12)
ax.text(0.99, 0.02, "line: s(x, λ)   •: its band averages   ○: Drizzle (PSF-convolved)", transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color=MU)
axm = fig.add_axes([0.06, 0.05, 0.42, 0.4], facecolor=BG)
for b in BANDS:
    cv = curves[b]
    axm.fill_between(cv["lam_um"], 0, cv["T"], alpha=0.35, lw=0, color="#7D879C")
    axm.plot(nodes[b], np.interp(nodes[b], cv["lam_um"], cv["T"]), "o", ms=4, color="#F5A524")
    axm.text(np.mean(nodes[b]), max(cv["T"]) + 0.03, b, ha="center", fontsize=10, color=FG)
axm.set_xscale("log")
axm.set_xticks([0.9, 1.5, 2, 3, 4.4])
axm.set_xticklabels(["0.9", "1.5", "2", "3", "4.4"])
axm.xaxis.set_minor_formatter(NullFormatter())
axm.set_xlabel("NIRCam transmission (SVO) and quadrature nodes")
axm.set_yticks([])
fig.savefig(out, dpi=110, facecolor=BG)
print("wrote", out, {k: v for k, v in pos.items()})
