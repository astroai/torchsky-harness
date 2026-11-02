#!/usr/bin/env python3
"""Torchsky vs comparator performance, one normalized JSON (record, never gate).

Areas (comma list via ``--areas``): ``wcs``, ``healpix``, ``catalogs``, ``moc``, ``sparse``,
``sht``, ``map``. For every (operation, size) the torchsky time and each available comparator's
time are recorded with the throughput (``size / torchsky_s``) and ``speedup``
(``comparator_s / torchsky_s``; above 1 means torchsky is faster). A comparator that is not
installed is recorded as ``unavailable`` with the reason, never silently dropped. No wall-time
threshold is applied (shared nodes); the file is the evidence.

    python benchmarks/bench_comparator_perf.py --areas wcs,healpix --device cpu \
        --sizes 100000,1000000 --json-out benchmark_results/comparator_perf_cpu.json

Comparators: PyAST runs out-of-process (``scripts/oracle/wcs_oracle.py``), STILTS is the
``stilts`` binary (wall time includes JVM start and table I/O, noted per row). Rows from the
parity gates (``tests/_perf_record.py``) land in ``comparator_perf.jsonl`` beside this file's
JSON and are merged by ``scripts/canfar/summarize.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

ROWS: list[dict[str, Any]] = []
DEADLINE: float | None = None  # per-area wall-time budget (monotonic); the bench records, never gates


class BudgetExceeded(Exception):
    """The area's time budget ran out: remaining cases are recorded as truncated."""


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def best_of(fn: Callable[[], Any], runs: int, dev: torch.device | None = None) -> float:
    if DEADLINE is not None and time.monotonic() > DEADLINE:
        raise BudgetExceeded
    fn()  # warm-up (JIT, caches, allocator)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if dev is not None:
            _sync(dev)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def add(area: str, op: str, comparator: str, size: int, t_ts: float, t_cmp: float | None, device: str, note: str = "") -> None:
    row = {
        "area": area, "operation": op, "comparator": comparator, "size": int(size), "device": device,
        "torchsky_s": t_ts, "comparator_s": t_cmp, "note": note,
        "throughput_per_s": size / t_ts if t_ts else None,
        "speedup": (t_cmp / t_ts) if (t_cmp is not None and t_ts) else None,
        "status": "ok" if t_cmp is not None else "unavailable",
    }  # fmt: skip
    ROWS.append(row)
    sp = "n/a" if row["speedup"] is None else f"{row['speedup']:.2f}x"
    print(f"[{area}] {op} n={size} vs {comparator}: torchsky {t_ts * 1e3:.2f} ms, speedup {sp} {note}", flush=True)


def unavailable(area: str, op: str, comparator: str, size: int, t_ts: float, device: str, why: str) -> None:
    add(area, op, comparator, size, t_ts, None, device, f"unavailable: {why}")


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["_"])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------- WCS
def bench_wcs(sizes, runs, dev) -> None:
    from astropy.wcs import WCS as AstroWCS

    import _oracle
    from test_wcs_pyast_parity import header_for
    from torchsky.wcs import WCS

    for case in ("TAN", "TAN_SIP"):
        header = header_for(case)
        # SIP needs a different grid scale for 1e6 points; sample uniformly in the image
        tw, aw = WCS(header), AstroWCS(header)
        rng = np.random.default_rng(0)
        for n in sizes:
            x = rng.uniform(8, 504, n)
            y = rng.uniform(8, 504, n)
            xt, yt = (torch.as_tensor(v, device=dev) for v in (x, y))
            t_p2w = best_of(lambda: tw.pixel_to_world(xt, yt, origin=1), runs, dev)
            ra, dec = (v.cpu().numpy() for v in tw.pixel_to_world(xt, yt, origin=1))
            rat, dect = (torch.as_tensor(v, device=dev) for v in (ra, dec))
            t_w2p = best_of(lambda: tw.world_to_pixel(rat, dect, origin=1), runs, dev)
            add("wcs", f"{case} pixel_to_world", "astropy.wcs", n, t_p2w, best_of(lambda: aw.all_pix2world(x, y, 1), runs), dev.type)
            add("wcs", f"{case} world_to_pixel", "astropy.wcs", n, t_w2p, best_of(lambda: aw.all_world2pix(ra, dec, 1), runs), dev.type)
            if _oracle.oracle_python("pyast") is None:
                unavailable("wcs", f"{case} pixel_to_world", "pyast", n, t_p2w, dev.type, "no PyAST oracle interpreter")
                unavailable("wcs", f"{case} world_to_pixel", "pyast", n, t_w2p, dev.type, "no PyAST oracle interpreter")
                continue
            try:
                ora = _oracle.run_wcs_oracle("pyast", header, x, y, ra, dec, repeat=max(3, runs))
            except BaseException as exc:  # noqa: BLE001 - pytest.skip raises Skipped (BaseException)
                why = f"oracle unavailable: {str(exc).splitlines()[0][:120]}"
                unavailable("wcs", f"{case} pixel_to_world", "pyast", n, t_p2w, dev.type, why)
                unavailable("wcs", f"{case} world_to_pixel", "pyast", n, t_w2p, dev.type, why)
                continue
            add("wcs", f"{case} pixel_to_world", "pyast", n, t_p2w, ora["t_forward_s"], dev.type, "PyAST time measured inside the oracle process")
            add("wcs", f"{case} world_to_pixel", "pyast", n, t_w2p, ora["t_inverse_s"], dev.type)
            if _oracle.oracle_python("kapteyn") is not None and case == "TAN":
                try:
                    kap = _oracle.run_wcs_oracle("kapteyn", header, x, y, ra, dec, repeat=max(3, runs))
                except BaseException as exc:  # noqa: BLE001
                    why = f"oracle unavailable: {str(exc).splitlines()[0][:120]}"
                    unavailable("wcs", f"{case} pixel_to_world", "kapteyn", n, t_p2w, dev.type, why)
                    unavailable("wcs", f"{case} world_to_pixel", "kapteyn", n, t_w2p, dev.type, why)
                else:
                    add("wcs", f"{case} pixel_to_world", "kapteyn", n, t_p2w, kap["t_forward_s"], dev.type)
                    add("wcs", f"{case} world_to_pixel", "kapteyn", n, t_w2p, kap["t_inverse_s"], dev.type)


# ------------------------------------------------------------------------ HEALPix
def bench_healpix(sizes, runs, dev, nside: int = 1024) -> None:
    from torchsky.sphere import ang2pix, neighbors, pix2ang

    healpy, hpgeom, ahp = _try_import("healpy"), _try_import("hpgeom"), _try_import("astropy_healpix")
    rng = np.random.default_rng(1)
    for n in sizes:
        lon = rng.uniform(0, 360, n)
        lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
        lt, bt = (torch.as_tensor(v, device=dev) for v in (lon, lat))
        pix = ang2pix(nside, lt, bt, nest=True, lonlat=True)
        pix_np = pix.cpu().numpy()
        ops = {
            "ang2pix": (best_of(lambda: ang2pix(nside, lt, bt, nest=True, lonlat=True), runs, dev), {
                "healpy": (healpy, lambda: healpy.ang2pix(nside, lon, lat, nest=True, lonlat=True)),
                "hpgeom": (hpgeom, lambda: hpgeom.angle_to_pixel(nside, lon, lat, nest=True)),
                "astropy-healpix": (ahp, lambda: ahp.HEALPix(nside=nside, order="nested").lonlat_to_healpix(lon * __import__("astropy.units").units.deg, lat * __import__("astropy.units").units.deg)),
            }),
            "pix2ang": (best_of(lambda: pix2ang(nside, pix, nest=True, lonlat=True), runs, dev), {
                "healpy": (healpy, lambda: healpy.pix2ang(nside, pix_np, nest=True, lonlat=True)),
                "hpgeom": (hpgeom, lambda: hpgeom.pixel_to_angle(nside, pix_np, nest=True)),
                "astropy-healpix": (ahp, lambda: ahp.HEALPix(nside=nside, order="nested").healpix_to_lonlat(pix_np)),
            }),
            "neighbours": (best_of(lambda: neighbors(nside, pix, nest=True), runs, dev), {
                "healpy": (healpy, lambda: healpy.get_all_neighbours(nside, pix_np, nest=True)),
                "hpgeom": (hpgeom, lambda: hpgeom.neighbors(nside, pix_np, nest=True)),
                "astropy-healpix": (ahp, lambda: ahp.HEALPix(nside=nside, order="nested").neighbours(pix_np)),
            }),
        }
        for op, (t_ts, comps) in ops.items():
            for name, (mod, fn) in comps.items():
                if mod is None:
                    unavailable("healpix", f"{op} nside={nside}", name, n, t_ts, dev.type, "not installed")
                    continue
                try:
                    add("healpix", f"{op} nside={nside}", name, n, t_ts, best_of(fn, runs), dev.type)
                except Exception as exc:  # noqa: BLE001
                    unavailable("healpix", f"{op} nside={nside}", name, n, t_ts, dev.type, repr(exc)[:120])


# ---------------------------------------------------------------------- catalogs
def _catalog(n: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    # right catalogue: counterparts with sub-arcsec scatter for 60 %, unrelated sources otherwise
    pick = rng.random(n) < 0.6
    ra2 = (ra[pick] + rng.normal(0, 0.4 / 3600, pick.sum()) / np.cos(np.radians(dec[pick]))) % 360
    dec2 = np.clip(dec[pick] + rng.normal(0, 0.4 / 3600, pick.sum()), -90, 90)
    ex = n // 2
    return ra, dec, np.concatenate([ra2, rng.uniform(0, 360, ex)]), np.concatenate([dec2, np.degrees(np.arcsin(rng.uniform(-1, 1, ex)))])


def bench_catalogs(sizes, runs, dev, radius: float = 1.5) -> None:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    import _stilts as st
    from torchsky.catalogs import crossmatch_sky

    for n in sizes:
        ra1, dec1, ra2, dec2 = _catalog(n)
        a = [torch.as_tensor(v, device=dev) for v in (ra1, dec1, ra2, dec2)]
        t_ts = best_of(lambda: crossmatch_sky(*a, radius_arcsec=radius, find="best"), max(1, runs // 2), dev)
        c1, c2 = SkyCoord(ra1 * u.deg, dec1 * u.deg), SkyCoord(ra2 * u.deg, dec2 * u.deg)
        add("catalogs", "crossmatch best", "astropy SkyCoord", n, t_ts, best_of(lambda: c1.match_to_catalog_sky(c2), 1), dev.type, "match_to_catalog_sky (nearest, no radius cut)")
        cmd = st.xt.stilts_command()
        if cmd is None:
            unavailable("catalogs", "crossmatch best", "stilts", n, t_ts, dev.type, "stilts not available")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            A = st.write_table(tmp / "a.fits", ra=ra1, dec=dec1, id=np.arange(n))
            B = st.write_table(tmp / "b.fits", ra=ra2, dec=dec2, id=np.arange(ra2.size))
            _, wall = st.tmatch2(A, B, tmp / "o.csv", radius=radius, find="best1")
            _, wall2 = st.tmatch2(A, B, tmp / "o.csv", radius=radius, find="best1", task="tskymatch2")
        add("catalogs", "crossmatch best", "stilts tmatch2", n, t_ts, wall, dev.type, "stilts wall includes JVM start and table I/O")
        add("catalogs", "crossmatch best", "stilts tskymatch2", n, t_ts, wall2, dev.type, "stilts wall includes JVM start and table I/O")


# ---------------------------------------------------------------- MOC / sparse
def bench_moc(sizes, runs, dev) -> None:
    from torchsky.sphere import MOC

    mocpy = _try_import("mocpy")
    order = 10
    t_cone = best_of(lambda: MOC.from_circle(20.0, -10.0, 5.0, max_order=order), runs)
    a = MOC.from_circle(0.0, 0.0, 2.0, max_order=order)
    b = MOC.from_circle(3.0, 0.0, 2.0, max_order=order)
    t_union = best_of(lambda: a.union(b), runs)
    t_inter = best_of(lambda: a.intersection(b), runs)
    rng = np.random.default_rng(3)
    fixed = (("from_circle", t_cone), ("union", t_union), ("intersection", t_inter))
    if mocpy is None:
        for op, t in fixed:
            unavailable("moc", f"{op} order={order}", "mocpy", 1, t, "cpu", "not installed")
    else:
        import astropy.units as u
        from astropy.coordinates import Angle, Latitude, Longitude

        MM = mocpy.MOC

        def cone(lo, la, r):
            return MM.from_cone(lon=Longitude(lo * u.deg), lat=Latitude(la * u.deg), radius=Angle(r, u.deg), max_depth=order)

        ma, mb = cone(0, 0, 2), cone(3, 0, 2)
        note = "size 1 = one call (throughput is calls/s)"
        add("moc", f"from_circle order={order}", "mocpy", 1, t_cone, best_of(lambda: cone(20, -10, 5), runs), "cpu", note)
        add("moc", f"union order={order}", "mocpy", 1, t_union, best_of(lambda: ma.union(mb), runs), "cpu", note)
        add("moc", f"intersection order={order}", "mocpy", 1, t_inter, best_of(lambda: ma.intersection(mb), runs), "cpu", note)
    for n in sizes:
        lon = rng.uniform(-3, 5, n)
        lat = rng.uniform(-3, 3, n)
        lt, bt = torch.as_tensor(lon), torch.as_tensor(lat)
        t_ct = best_of(lambda: a.contains(lt, bt), runs)
        if mocpy is None:
            unavailable("moc", f"contains order={order}", "mocpy", n, t_ct, "cpu", "not installed")
        else:
            add("moc", f"contains order={order}", "mocpy", n, t_ct, best_of(lambda: ma.contains_lonlat(Longitude(lon * u.deg), Latitude(lat * u.deg)), runs), "cpu")


def bench_sparse(sizes, runs, dev, nside: int = 4096) -> None:
    from torchsky.sphere import SparseHealpixMap

    hs = _try_import("healsparse")
    rng = np.random.default_rng(4)
    npix_cov = 20_000
    pix = np.unique(rng.integers(0, 12 * nside * nside, npix_cov)).astype(np.int64)
    vals = rng.normal(size=pix.size)
    sp = SparseHealpixMap.from_pixels(torch.as_tensor(pix), torch.as_tensor(vals), nside_sparse=nside, nest=True, nside_coverage=32)
    if hs is not None:
        hmap = hs.HealSparseMap.make_empty(32, nside, np.float64)
        hmap[pix] = vals
    for n in sizes:
        q = rng.choice(pix, size=n)
        qt = torch.as_tensor(q)
        t_get = best_of(lambda: sp.get_values(qt), runs)
        if hs is None:
            unavailable("sparse", f"get_values nside={nside}", "healsparse", n, t_get, "cpu", "not installed")
            continue
        add("sparse", f"get_values nside={nside}", "healsparse", n, t_get, best_of(lambda: hmap.get_values_pix(q), runs), "cpu")
    t_ud = best_of(lambda: sp.ud_grade(nside // 4), max(1, runs // 2))
    if hs is None:
        unavailable("sparse", f"ud_grade /4 nside={nside}", "healsparse", pix.size, t_ud, "cpu", "not installed")
    else:
        add("sparse", f"ud_grade /4 nside={nside}", "healsparse", pix.size, t_ud, best_of(lambda: hmap.degrade(nside // 4), max(1, runs // 2)), "cpu", "size = covered pixels")


# -------------------------------------------------------------------------- SHT
def _sht_peer(lib: str, nside: int, lmax: int, runs: int) -> dict[str, float]:
    """Time healpy / s2fft SHTs; run in a child process (separate OpenMP/JAX runtime from torch)."""
    m = np.random.default_rng(5).normal(size=12 * nside * nside)
    out: dict[str, float] = {}
    if lib == "healpy":
        import healpy

        alm = healpy.map2alm(m, lmax=lmax, iter=0)
        out["map2alm"] = best_of(lambda: healpy.map2alm(m, lmax=lmax, iter=0), runs)
        out["alm2map"] = best_of(lambda: healpy.alm2map(alm, nside, lmax=lmax), runs)
    else:
        import s2fft

        L = lmax + 1
        flm = s2fft.forward(m, L=L, nside=nside, sampling="healpix", method="jax", reality=True)
        out["map2alm"] = best_of(lambda: s2fft.forward(m, L=L, nside=nside, sampling="healpix", method="jax", reality=True), runs)
        out["alm2map"] = best_of(lambda: s2fft.inverse(flm, L=L, nside=nside, sampling="healpix", method="jax", reality=True), runs)
    return out


def _past_budget() -> None:
    if DEADLINE is not None and time.monotonic() > DEADLINE:
        raise BudgetExceeded


def _peer_times(lib: str, nside: int, lmax: int, runs: int) -> tuple[dict[str, float] | None, str]:
    _past_budget()
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--sht-peer", lib, "--nside", str(nside), "--lmax", str(lmax), "--runs", str(runs)],
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    if done.returncode != 0:
        return None, f"{lib} child exited {done.returncode}: {done.stderr.strip()[-120:]}"
    return json.loads(done.stdout.strip().splitlines()[-1]), ""


# lmax > 512 at nside >= 512 hits torchsky's dense-basis fallback (needs TBs): measured separately, see docs
def bench_sht(sizes, runs, dev, configs=((256, 512), (512, 512))) -> None:
    from torchsky.sphere import alm2map, map2alm

    for nside, lmax in configs:
        npix = 12 * nside * nside
        mt = torch.as_tensor(np.random.default_rng(5).normal(size=npix), device=dev)
        alm = map2alm(mt, nside=nside, lmax=lmax)
        r = max(1, runs // 2)
        t_a = best_of(lambda: map2alm(mt, nside=nside, lmax=lmax), r, dev)
        t_s = best_of(lambda: alm2map(alm, nside, lmax=lmax), r, dev)
        for lib, label in (("healpy", "healpy"), ("s2fft", "s2fft")):
            if _try_import("healpy" if lib == "healpy" else "s2fft") is None:
                times, why = None, "not installed"
            else:
                times, why = _peer_times(lib, nside, lmax, r)
            note = "s2fft jax on host; L=lmax+1" if lib == "s2fft" else ""
            for op, t_ts in (("map2alm", t_a), ("alm2map", t_s)):
                name = f"{op} nside={nside} lmax={lmax}"
                if times is None:
                    unavailable("sht", name, label, npix, t_ts, dev.type, why)
                else:
                    add("sht", name, label, npix, t_ts, times[op], dev.type, note)


# -------------------------------------------------------------------------- map
def _tan_header(n: int, scale: float, crval=(180.0, 0.0), rot=0.0) -> dict:
    import math

    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    return {
        "NAXIS": 2, "NAXIS1": n, "NAXIS2": n, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": (n + 1) / 2, "CRPIX2": (n + 1) / 2, "CRVAL1": crval[0], "CRVAL2": crval[1],
        "CD1_1": -scale * c, "CD1_2": scale * s, "CD2_1": scale * s, "CD2_2": scale * c,
    }  # fmt: skip


IMAGE_SIZES = (2048, 4096)


def bench_map(sizes, runs, dev) -> None:
    image_sizes = IMAGE_SIZES
    from astropy.wcs import WCS as AstroWCS

    from torchsky.map import resample
    from torchsky.scene import ImageObservation

    reproject = _try_import("reproject")
    dfr = _try_import("dfreproject")
    for n in image_sizes:
        src_h, dst_h = _tan_header(n, 1e-4), _tan_header(n, 1.05e-4, (180.002, 0.001), rot=7.0)
        rng = np.random.default_rng(6)
        img = rng.normal(size=(n, n)).astype(np.float32)
        obs = ImageObservation(data=torch.as_tensor(img, device=dev), header=src_h)
        for kernel, label in (("bilinear", "bilinear"), ("conservative_polygon", "exact")):
            try:
                t_ts = best_of(lambda: resample(obs, dst_h, kernel=kernel, device=dev), max(1, runs // 3), dev)
            except Exception as exc:  # noqa: BLE001
                print(f"torchsky {kernel} failed: {exc!r}")
                continue
            sw, dw = AstroWCS(src_h), AstroWCS(dst_h)
            if reproject is None:
                unavailable("map", f"{label} {n}x{n}", "reproject", n * n, t_ts, dev.type, "not installed")
            else:
                fn = reproject.reproject_interp if label == "bilinear" else reproject.reproject_exact
                kw = dict(order="bilinear") if label == "bilinear" else {}
                add("map", f"{label} {n}x{n}", "reproject", n * n, t_ts, best_of(lambda: fn((img, sw), dw, shape_out=(n, n), **kw), 1), dev.type)
            if label == "bilinear":
                if dfr is None:
                    unavailable("map", f"{label} {n}x{n}", "dfreproject", n * n, t_ts, dev.type, "not installed")
                else:
                    try:
                        add("map", f"{label} {n}x{n}", "dfreproject", n * n, t_ts, best_of(lambda: dfr.calculate_reprojection((img, sw), dw, shape_out=(n, n), order="bilinear", conserve_flux=False, compute_jacobian=False, show_chunk_progress=False, show_log=False), 1), dev.type)
                    except Exception as exc:  # noqa: BLE001
                        unavailable("map", f"{label} {n}x{n}", "dfreproject", n * n, t_ts, dev.type, repr(exc)[:120])
            # SWarp and Montage are binaries on the improc image
            for exe, key in (("swarp", "swarp"), ("mProject", "montage")):
                if shutil.which(exe) is None:
                    unavailable("map", f"{label} {n}x{n}", key, n * n, t_ts, dev.type, f"{exe} not on PATH")
                else:
                    t_bin = _time_binary(exe, img, src_h, dst_h, n, label)
                    if t_bin is None:
                        unavailable("map", f"{label} {n}x{n}", key, n * n, t_ts, dev.type, f"{exe} run failed")
                    else:
                        add("map", f"{label} {n}x{n}", key, n * n, t_ts, t_bin, dev.type, "wall time of the binary incl. file I/O")


def _time_binary(exe: str, img: np.ndarray, src_h: dict, dst_h: dict, n: int, label: str) -> float | None:
    """Wall time of SWarp / Montage ``mProject`` on files written with torchfits."""
    _past_budget()
    import torchfits

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        torchfits.write(str(tmp / "in.fits"), torch.as_tensor(img), header=src_h, overwrite=True)
        torchfits.write(str(tmp / "tpl.fits"), torch.zeros((n, n)), header=dst_h, overwrite=True)
        hdr = tmp / "tpl.hdr"
        hdr.write_text("".join(f"{k:<8}= {v!r}\n" if isinstance(v, str) else f"{k:<8}= {v}\n" for k, v in dst_h.items()) + "END\n")
        if exe == "mProject":
            cmd = [exe, str(tmp / "in.fits"), str(tmp / "out.fits"), str(hdr)]
        else:
            if label != "bilinear":
                return None
            cmd = [exe, str(tmp / "in.fits"), "-c", "/dev/null", "-IMAGEOUT_NAME", str(tmp / "out.fits"), "-WEIGHTOUT_NAME", str(tmp / "w.fits"),
                   "-RESAMPLING_TYPE", "BILINEAR", "-SUBTRACT_BACKGROUND", "N", "-WRITE_XML", "N", "-CENTER_TYPE", "MANUAL", "-IMAGE_SIZE", f"{n},{n}"]  # fmt: skip
        t0 = time.perf_counter()
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=tmp, check=False)
        return time.perf_counter() - t0 if done.returncode == 0 else None


AREAS = {
    "wcs": bench_wcs, "healpix": bench_healpix, "catalogs": bench_catalogs, "moc": bench_moc,
    "sparse": bench_sparse, "sht": bench_sht, "map": bench_map,
}  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--areas", default=",".join(AREAS))
    parser.add_argument("--sht-peer", choices=["healpy", "s2fft"], help=argparse.SUPPRESS)
    parser.add_argument("--nside", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--lmax", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--sizes", default="100000,1000000", help="point/row counts for wcs, healpix, catalogs, moc, sparse")
    parser.add_argument("--image-sizes", default="2048,4096", help="square image edge lengths for the map area")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--area-budget-s", type=float, default=45.0, help="per-area wall-time budget; later cases are recorded as truncated")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (areas without device support run on cpu)")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.sht_peer:
        print(json.dumps(_sht_peer(args.sht_peer, args.nside, args.lmax, args.runs)))
        return 0
    if args.json_out is None:
        parser.error("--json-out is required")
    global IMAGE_SIZES, DEADLINE
    IMAGE_SIZES = tuple(int(v) for v in args.image_sizes.split(","))
    dev = torch.device(args.device)
    sizes = [int(s) for s in args.sizes.split(",")]
    cpu_only = {"moc", "sparse"}
    failures = 0
    for area in args.areas.split(","):
        fn = AREAS[area]
        d = torch.device("cpu") if area in cpu_only else dev
        DEADLINE = time.monotonic() + args.area_budget_s
        try:
            fn(sizes, args.runs, d)
        except BudgetExceeded:
            ROWS.append({"area": area, "operation": "truncated", "comparator": "-", "size": 0, "device": d.type, "status": f"truncated: area budget {args.area_budget_s:g}s reached"})
            print(f"[{area}] truncated at the {args.area_budget_s:g}s area budget", file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{area}] FAILED: {exc!r}", file=sys.stderr, flush=True)
            ROWS.append({"area": area, "operation": "error", "comparator": "-", "size": 0, "device": d.type, "status": f"error: {exc!r}"[:300]})
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "comparator_perf/1", "device": args.device, "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(), "cpu_count": os.cpu_count(), "rows": ROWS,
    }  # fmt: skip
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out} ({len(ROWS)} rows, {failures} failed areas)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
