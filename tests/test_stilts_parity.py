"""Pair-set parity of ``torchsky.catalogs.crossmatch_sky`` against STILTS (Java, ``stilts`` binary).

Modes compared (matched pair sets must be identical, ties at exactly the radius excepted):

* ``tmatch2 matcher=sky find=best1`` <-> ``crossmatch_sky(find="best")`` (nearest right per left)
* ``tmatch2 find=best2``              <-> ``crossmatch_sky(right, left, find="best")``
* ``tmatch2 find=all``                <-> ``crossmatch_sky(find="all")``
* ``tmatch2 join=1not2 / 2not1``      <-> rows absent from the pair set
* ``tskymatch2`` (fast sky-only task) <-> same as ``tmatch2 matcher=sky``
* ``tmatch2 matcher=skyerr``          <-> per-left-row radius ``err1 + err2`` (right errors constant)
* ``tmatch1 matcher=sky`` groups      <-> connected components of the torchsky self-match pair graph

Not provided by torchsky 0.5 (see docs/coverage-matrix.md): ``find=best`` (symmetric one-to-one),
``tmatchn`` (multi-catalogue), ``sky3d``/``exact`` matchers, per-right-row errors, ``skyellipse``,
and group (friends-of-friends) construction itself.

Data: a synthetic pair of catalogues (1e5 rows, RA wrap band and a pole cap) and the real Gaia DR3
x 2MASS cone staged by ``scripts/fetch_real_catalogs.py`` (``CATALOG_REAL_DIR``). The real-data
cases are the release gate; they fail closed under ``REQUIRE_TOOLS``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torchfits

from _gate_helpers import device, f64, sep_arcsec
import _perf_record
import _stilts as st

from torchsky.catalogs import crossmatch_sky

RADIUS = 1.5  # arcsec; separations are continuous, so no pair sits at the radius except by tolerance


def _synthetic(n: int = 100_000, seed: int = 7) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)

    def sphere(m, dec_lo=-90.0, dec_hi=90.0):
        z = rng.uniform(np.sin(np.radians(dec_lo)), np.sin(np.radians(dec_hi)), m)
        return rng.uniform(0, 360, m), np.degrees(np.arcsin(z))

    parts = [sphere(n // 2, -60, 60)]
    # RA wrap band (dense: many ambiguous matches across ra = 0/360) and a pole cap
    ra = rng.uniform(-0.15, 0.15, n // 4) % 360.0
    parts.append((ra, rng.uniform(-3, 3, n // 4)))
    z = rng.uniform(np.sin(np.radians(89.9)), 1.0, n // 4)
    parts.append((rng.uniform(0, 360, n // 4), np.degrees(np.arcsin(z))))
    ra1 = np.concatenate([p[0] for p in parts])
    dec1 = np.concatenate([p[1] for p in parts])
    # counterparts with ~0.6" scatter in the tangent plane, plus unrelated sources
    pick = rng.random(ra1.size) < 0.7
    sig = 0.6 / 3600.0
    dec2 = np.clip(dec1[pick] + rng.normal(0, sig, pick.sum()), -90, 90)
    ra2 = (ra1[pick] + rng.normal(0, sig, pick.sum()) / np.maximum(np.cos(np.radians(dec2)), 1e-9)) % 360.0
    extra_ra, extra_dec = sphere(n // 5)
    return ra1, dec1, np.concatenate([ra2, extra_ra]), np.concatenate([dec2, extra_dec])


def _real() -> tuple[np.ndarray, ...]:
    root = os.environ.get("CATALOG_REAL_DIR")
    manifest = Path(root) / "manifest.json" if root else None
    if manifest is None or not manifest.is_file():
        pytest.skip("real catalogues not staged (CATALOG_REAL_DIR/manifest.json; scripts/fetch_real_catalogs.py)")
    json.loads(manifest.read_text())  # must parse
    out = []
    for name in ("gaia", "twomass"):
        table = torchfits.table.read(str(Path(root) / f"{name}.fits"))
        out += [np.asarray(table["ra"]), np.asarray(table["dec"])]
    return tuple(out)


def _match(l_ra, l_dec, r_ra, r_dec, radius, find):
    dev = device()
    t0 = time.perf_counter()
    res = crossmatch_sky(f64(l_ra, dev), f64(l_dec, dev), f64(r_ra, dev), f64(r_dec, dev), radius_arcsec=radius, find=find)
    li, ri = res.left_index.cpu().numpy(), res.right_index.cpu().numpy()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return li, ri, time.perf_counter() - t0


def _assert_same(ts, ss, l_ra, l_dec, r_ra, r_dec, radius, label, tol=1e-6):
    """Pair sets equal, except pairs whose separation is within ``tol`` arcsec of the radius."""
    diff = ts ^ ss
    if diff:
        idx = np.array(sorted(diff))
        sep = sep_arcsec(l_ra[idx[:, 0]], l_dec[idx[:, 0]], r_ra[idx[:, 1]], r_dec[idx[:, 1]])
        bad = np.abs(sep - radius) > tol
        assert not bad.any(), f"{label}: {int(bad.sum())} pair differences away from the radius (of {len(diff)})"
    assert len(diff) <= max(2, int(1e-5 * len(ts))), f"{label}: {len(diff)} boundary ties"


def _run_modes(tmp_path: Path, l_ra, l_dec, r_ra, r_dec, tag: str) -> None:
    A = st.write_table(tmp_path / "a.fits", ra=l_ra, dec=l_dec, id=np.arange(l_ra.size))
    B = st.write_table(tmp_path / "b.fits", ra=r_ra, dec=r_dec, id=np.arange(r_ra.size))
    out = tmp_path / "o.csv"
    n = l_ra.size
    dev = device().type

    def rec(op, comp_s, t_s, note="", size=n):
        _perf_record.record("catalogs", f"{tag}:{op}", "stilts", size, t_s, comp_s, dev, note + " stilts wall includes JVM start and table I/O")

    for find_st, find_t, swap in (("best1", "best", False), ("all", "all", False), ("best2", "best", True)):
        cols, wall = st.tmatch2(A, B, out, radius=RADIUS, find=find_st)
        if swap:
            ri, li, t_s = _match(r_ra, r_dec, l_ra, l_dec, RADIUS, find_t)
        else:
            li, ri, t_s = _match(l_ra, l_dec, r_ra, r_dec, RADIUS, find_t)
        _assert_same(set(zip(li.tolist(), ri.tolist())), st.pairs(cols), l_ra, l_dec, r_ra, r_dec, RADIUS, f"{tag} tmatch2 {find_st}")
        rec(f"tmatch2 find={find_st}", wall, t_s)
        if tag == "synthetic" and find_st == "all":
            wrap = (l_ra[li] < 0.2) | (l_ra[li] > 359.8)
            pole = l_dec[li] > 89.9
            assert wrap.sum() > 100 and pole.sum() > 100, "synthetic wrap/pole regions must be populated with matches"

    # join semantics: unmatched rows are the complement of the pair set
    li, ri, _ = _match(l_ra, l_dec, r_ra, r_dec, RADIUS, "best")
    cols, _ = st.tmatch2(A, B, out, radius=RADIUS, find="best1", join="1not2")
    assert abs(cols["id"].size - (n - np.unique(li).size)) <= 2, f"{tag} join=1not2"
    cols, _ = st.tmatch2(A, B, out, radius=RADIUS, find="best2", join="2not1")
    ri2, li2, _ = _match(r_ra, r_dec, l_ra, l_dec, RADIUS, "best")
    assert abs(cols["id"].size - (r_ra.size - np.unique(ri2).size)) <= 2, f"{tag} join=2not1"

    # tskymatch2: the fast sky-only task gives the same best1 set
    cols, wall = st.tmatch2(A, B, out, radius=RADIUS, find="best1", task="tskymatch2")
    li, ri, t_s = _match(l_ra, l_dec, r_ra, r_dec, RADIUS, "best")
    _assert_same(set(zip(li.tolist(), ri.tolist())), st.pairs(cols), l_ra, l_dec, r_ra, r_dec, RADIUS, f"{tag} tskymatch2")
    rec("tskymatch2 find=best1", wall, t_s)

    # skyerr with per-left-row errors and a constant right error: radius_i = e1_i + e2
    rng = np.random.default_rng(3)
    e1 = rng.uniform(0.3, 1.2, l_ra.size)
    e2 = 0.3
    A2 = st.write_table(tmp_path / "a2.fits", ra=l_ra, dec=l_dec, id=np.arange(n), err=e1)
    B2 = st.write_table(tmp_path / "b2.fits", ra=r_ra, dec=r_dec, id=np.arange(r_ra.size), err=np.full(r_ra.size, e2))
    cols, wall = st.tmatch2(
        A2, B2, out, radius=2.0, find="all", matcher="skyerr", values1="ra dec err", values2="ra dec err"
    )
    dev_t = device()
    t0 = time.perf_counter()
    res = crossmatch_sky(
        f64(l_ra, dev_t), f64(l_dec, dev_t), f64(r_ra, dev_t), f64(r_dec, dev_t),
        radius_arcsec=f64(e1 + e2, dev_t), find="all",
    )  # fmt: skip
    t_s = time.perf_counter() - t0
    ts = set(zip(res.left_index.cpu().tolist(), res.right_index.cpu().tolist()))
    diff = ts ^ st.pairs(cols)
    assert len(diff) <= max(2, int(1e-5 * len(ts))), f"{tag} skyerr: {len(diff)} pair differences"
    rec("tmatch2 matcher=skyerr find=all", wall, t_s)

    # tmatch1 sky groups vs connected components of the torchsky self-match pair graph
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    sub = min(n, 50_000)
    S = st.write_table(tmp_path / "s.fits", ra=l_ra[:sub], dec=l_dec[:sub], id=np.arange(sub))
    cols_g, wall = st.run_stilts(
        ["tmatch1", f"in={S}", "ifmt=fits", "matcher=sky", f"params={RADIUS}", "values=ra dec", "action=identify"], out
    )
    li, ri, t_s = _match(l_ra[:sub], l_dec[:sub], l_ra[:sub], l_dec[:sub], RADIUS, "all")
    keep = li != ri
    graph = coo_matrix((np.ones(keep.sum()), (li[keep], ri[keep])), shape=(sub, sub))
    ncomp, label = connected_components(graph, directed=False)
    sizes = np.bincount(label)
    grouped = sizes[label] > 1
    gid = cols_g["GroupID"]
    st_grouped = ~np.isnan(gid)
    assert np.array_equal(grouped, st_grouped), f"{tag} tmatch1: grouped-row sets differ"
    # same partition: pairing (torchsky label, STILTS GroupID) must be one-to-one on grouped rows
    pairs_ = set(zip(label[grouped].tolist(), gid[st_grouped].astype(np.int64).tolist()))
    assert len(pairs_) == len(set(p[0] for p in pairs_)) == len(set(p[1] for p in pairs_)), f"{tag} tmatch1 partitions differ"
    rec("tmatch1 sky identify (self-match pairs + components)", wall, t_s, "torchsky time is the self all-pairs match only;", size=sub)


def test_stilts_synthetic_wrap_and_pole(tmp_path: Path) -> None:
    st.need_stilts()
    _run_modes(tmp_path, *_synthetic(), tag="synthetic")


def test_stilts_real_gaia_2mass(tmp_path: Path) -> None:
    st.need_stilts()
    g_ra, g_dec, t_ra, t_dec = _real()
    _run_modes(tmp_path, g_ra, g_dec, t_ra, t_dec, tag="gaia-dr3 x 2mass")
