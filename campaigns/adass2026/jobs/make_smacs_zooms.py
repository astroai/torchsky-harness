"""SMACS J0723 zooms rendered with torchfits.transforms.rgb (one joint stretch per zoom across methods).
usage: make_smacs_zooms.py CUBEDIR OUTDIR [methods...]"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from PIL import Image
from torchfits.transforms import rgb

cd, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
methods = sys.argv[3:] or ["swarp", "drizzle", "explicit_ownpsf", "implicit_ownpsf"]
cubes = {m: torch.from_numpy(fits.getdata(cd / f"{m}.fits").astype(np.float32)).flip(-2) for m in methods}  # north up
H = next(iter(cubes.values())).shape[-1]
f = H / 512.0  # boxes defined on the 512 x 0.04" grid
ZOOMS = {"arc": (296, 0, 160), "star": (0, 296, 128), "group": (96, 352, 160), "core": (200, 220, 160), "right": (384, 224, 128), "lens": (90, 290, 160), "chain": (352, 236, 160), "top": (200, 0, 128), "arcend": (330, 90, 128), "ul": (60, 120, 128), "lowmid": (300, 360, 128), "faint": (288, 352, 160), "ll": (0, 352, 160), "full": (0, 0, 512)}
if os.environ.get("ZOOMS"):
    ZOOMS = {k: ZOOMS[k] for k in os.environ["ZOOMS"].split(",")}


def save(t, path, up):
    a = (t.numpy() * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a)
    if up != 1:
        im = im.resize((im.width * up, im.height * up), Image.NEAREST)
    im.save(path)


for z, (x0, y0, s) in ZOOMS.items():
    x0, y0, s = int(x0 * f), int(y0 * f), int(s * f)
    crops = [cubes[m][:, y0:y0 + s, x0:x0 + s] for m in methods]
    big = rgb(torch.cat(crops, -1), brightness=0.12, saturation=1.6, scene=os.environ.get("SCENE", "empty") if z != "full" else "auto", dtype=torch.float32)
    up = max(1, int(round(640 / s))) if z != "full" else 1
    for k, m in enumerate(methods):
        save(big[:, k * s:(k + 1) * s], out / f"smacs_{z}_{m}.png", up)
    save(big, out / f"smacs_{z}_strip.png", 1)
    print(z, s, up)
