"""Parity of torchsky's STILTS-style crossmatch primitives against STILTS (Java, ``stilts`` binary).

Every comparison is on matched row tuples (0-based row indices, ``-1`` for an absent side), compared
as multisets: exact equality, with one documented exclusion. A pair whose great-circle separation is
within ``TIE_TOL`` (1e-6 arcsec, float rounding) of its threshold may be classified differently by
the two implementations; such pairs, and every output tuple that involves one of their rows, are
dropped from both sides, and their number is bounded (``<= max(2, 1e-5 n)``). Nothing else is excluded.

Mapping (STILTS <-> torchsky):

* ``tmatch2 matcher=sky find=best1|best2|all`` <-> ``crossmatch_sky(find="best1"|"best2"|"all")``
  (also ``tskymatch2``), each with every join ``1and2 1or2 all1 all2 1not2 2not1`` <-> ``join_rows``
* ``tmatch2 find=best`` <-> ``crossmatch_sky(find="mutual")`` (exactly the mutual-best pairs, every join).
  ``find="symmetric"`` is a greedy one-to-one assignment and is deliberately not STILTS
* ``tmatch2 matcher=skyerr`` (match if separation <= err1 + err2) <-> ``left_err_arcsec`` + ``right_err_arcsec``
  with per-row errors on both sides, ``find=all`` and all joins (STILTS ranks ``best`` by an
  error-normalised score; torchsky ranks by plain separation, so ``best*`` is not compared)
* ``tmatch2 matcher=exact`` <-> ``crossmatch_exact`` (``find=all`` and all joins by tuple; for
  ``best1/best2/best`` every pair of a key scores equally, STILTS' pick among equals is unspecified
  while torchsky takes the lowest rows, so only the matched row counts are compared)
* ``tmatch1 matcher=sky action=identify`` <-> ``crossmatch_self(group=True)`` (group partitions and sizes, not ids)
* ``tmatchn matcher=sky multimode=group`` <-> ``crossmatch_many(mode="group")`` (partitions and group
  sizes, since STILTS picks an arbitrary member of a catalogue within a group)
* ``tmatchn multimode=pairs``: STILTS gives each reference row its ``best1`` partner in every other
  catalogue (an entry may serve several reference rows) <-> ``crossmatch_many(mode="pairs")``, exact

Intentional differences, not tested against STILTS (docs/coverage-matrix.md): ``distance_tol`` is a
``|d1 - d2|`` cut (STILTS ``sky3d`` is Euclidean), no ``skyellipse``, error-normalised scores.

Data: three synthetic catalogues (1e5 rows, RA wrap band and a pole cap) and the real Gaia DR3 x 2MASS
x AllWISE cone staged by ``scripts/fetch_real_catalogs.py`` (``CATALOG_REAL_DIR``; sha256 checked
against the manifest). Per-row errors for ``skyerr`` are 10x the catalogues' positional errors (Gaia
``hypot(ra_error, dec_error)``, 2MASS ``err_maj``, AllWISE ``hypot(sigra, sigdec)``) so that matches
are plentiful in the crowded field. The real-data cases are the release gate; they fail closed under
``REQUIRE_TOOLS``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
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

from torchsky.catalogs import (
    crossmatch_exact,
    crossmatch_many,
    crossmatch_self,
    crossmatch_sky,
    join_rows,
)

RADIUS = 1.5  # arcsec
TIE_TOL = 1e-6  # arcsec; see the module docstring
JOINS = ("1and2", "1or2", "all1", "all2", "1not2", "2not1")
FINDS = (("best1", "best1"), ("best2", "best2"), ("all", "all"))  # (STILTS, torchsky)


@dataclass
class Cat:
    ra: np.ndarray
    dec: np.ndarray
    err: np.ndarray  # arcsec, per row, finite and >= 0 (skyerr)
    key: np.ndarray  # int64 cell key (exact matcher)

    @property
    def n(self) -> int:
        return self.ra.size

    def table(self, path: Path) -> Path:
        return st.write_table(
            path,
            ra=self.ra,
            dec=self.dec,
            id=np.arange(self.n),
            err=self.err,
            key=self.key,
        )


def _cat(ra, dec, err) -> Cat:
    ra, dec, err = (np.ascontiguousarray(v, dtype=np.float64) for v in (ra, dec, err))
    assert np.isfinite(err).all() and (err >= 0).all()
    # 10 arcsec cells: a real-data-derived exact key shared by catalogues of one field
    key = np.floor(ra * 360.0).astype(np.int64) * 1_000_000 + np.floor((dec + 90.0) * 360.0).astype(np.int64)
    return Cat(ra, dec, err, key)


def _synthetic(n: int = 100_000, seed: int = 7) -> list[Cat]:
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

    def derived(frac, sig_arcsec, extra):
        # counterparts with tangent-plane scatter, plus unrelated sources
        pick = rng.random(ra1.size) < frac
        sig = sig_arcsec / 3600.0
        dec = np.clip(dec1[pick] + rng.normal(0, sig, pick.sum()), -90, 90)
        ra = (ra1[pick] + rng.normal(0, sig, pick.sum()) / np.maximum(np.cos(np.radians(dec)), 1e-9)) % 360.0
        er, ed = sphere(extra)
        return np.concatenate([ra, er]), np.concatenate([dec, ed])

    ra2, dec2 = derived(0.7, 0.6, n // 5)
    ra3, dec3 = derived(0.4, 0.8, n // 10)
    out = []
    for ra_, dec_, lo, hi in (
        (ra1, dec1, 0.3, 1.2),
        (ra2, dec2, 0.2, 0.8),
        (ra3, dec3, 0.3, 1.0),
    ):
        out.append(_cat(ra_, dec_, rng.uniform(lo, hi, ra_.size)))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real() -> dict[str, Cat]:
    root = os.environ.get("CATALOG_REAL_DIR")
    manifest_path = Path(root) / "manifest.json" if root else None
    if manifest_path is None or not manifest_path.is_file():
        pytest.skip("real catalogues not staged (CATALOG_REAL_DIR/manifest.json; scripts/fetch_real_catalogs.py)")
    manifest = json.loads(manifest_path.read_text())
    errs = {
        "gaia": lambda t: 10.0 * np.hypot(t["ra_error"], t["dec_error"]) / 1000.0,  # mas -> arcsec
        "twomass": lambda t: 10.0 * t["err_maj"],
        "allwise": lambda t: 10.0 * np.hypot(t["sigra"], t["sigdec"]),
    }
    out = {}
    for name, err in errs.items():
        entry = manifest["tables"].get(name)
        assert entry is not None, f"manifest lacks {name}; rerun scripts/fetch_real_catalogs.py"
        path = Path(root) / entry["file"]
        assert _sha256(path) == entry["sha256"], f"{name}: sha256 differs from the manifest"
        table = torchfits.table.read(str(path))
        cols = {k: np.asarray(table[k]) for k in table.column_names}
        assert cols["ra"].size == entry["rows"], f"{name}: row count differs from the manifest"
        out[name] = _cat(cols["ra"], cols["dec"], err(cols))
    return out


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _sky(a: Cat, b: Cat, **kw):
    dev = device()
    t0 = time.perf_counter()
    res = crossmatch_sky(f64(a.ra, dev), f64(a.dec, dev), f64(b.ra, dev), f64(b.dec, dev), **kw)
    _sync(dev)
    return res, time.perf_counter() - t0


def _pairs(res) -> set[tuple[int, int]]:
    return set(zip(res.left_index.cpu().tolist(), res.right_index.cpu().tolist()))


def _rec(tag: str, op: str, comp_s: float, t_s: float, size: int, note: str = "") -> None:
    _perf_record.record(
        "catalogs", f"{tag}:{op}", "stilts", size, t_s, comp_s, device().type,
        (note + " stilts wall includes JVM start and table I/O").strip(),
    )  # fmt: skip


def _tie_rows(ts: set, ss: set, a: Cat, b: Cat, thr, label: str) -> tuple[set[int], set[int]]:
    """Pair-set difference: every differing pair must sit within TIE_TOL of its threshold; returns its rows."""
    diff = ts ^ ss
    if not diff:
        return set(), set()
    idx = np.array(sorted(diff))
    sep = sep_arcsec(a.ra[idx[:, 0]], a.dec[idx[:, 0]], b.ra[idx[:, 1]], b.dec[idx[:, 1]])
    bad = np.abs(sep - thr(idx[:, 0], idx[:, 1])) > TIE_TOL
    assert not bad.any(), f"{label}: {int(bad.sum())} pair differences away from the threshold (of {len(diff)})"
    assert len(diff) <= max(2, int(1e-5 * len(ts))), f"{label}: {len(diff)} boundary ties"
    return set(idx[:, 0].tolist()), set(idx[:, 1].tolist())


def _assert_rows(ts_rows, st_rows, tie_l: set, tie_r: set, label: str) -> None:
    def keep(rows):
        return Counter(t for t in rows if t[0] not in tie_l and t[1] not in tie_r)

    mine, theirs = keep(ts_rows), keep(st_rows)
    only_mine, only_theirs = (
        sum((mine - theirs).values()),
        sum((theirs - mine).values()),
    )
    # compare counts, not the Counters: pytest would render a multi-million-row diff on failure
    assert only_mine == 0 and only_theirs == 0, f"{label}: {only_mine} torchsky-only and {only_theirs} STILTS-only rows"


def _stilts_rows(cols, join: str):
    if join in ("1not2", "2not1"):
        return st.one_sided(cols, 1 if join == "1not2" else 2)
    return st.row_tuples(cols)


def _joined(res, a: Cat, b: Cat, join: str):
    left, right = join_rows(res, a.n, b.n, join)
    return list(zip(left.cpu().tolist(), right.cpu().tolist()))


def _sky_modes(tmp: Path, a: Cat, b: Cat, tag: str) -> None:
    """tmatch2 / tskymatch2 matcher=sky: every find x every join."""
    A, B, out = a.table(tmp / "a.fits"), b.table(tmp / "b.fits"), tmp / "o.csv"
    # boundary ties come from the find=all pair sets (the radius test is find-independent)
    cols, wall = st.tmatch2(A, B, out, radius=RADIUS, find="all")
    res_all, t_s = _sky(a, b, radius_arcsec=RADIUS, find="all")
    tie_l, tie_r = _tie_rows(_pairs(res_all), st.pairs(cols), a, b, lambda i, j: RADIUS, f"{tag} find=all")
    _rec(tag, "tmatch2 matcher=sky find=all", wall, t_s, a.n)
    if tag == "synthetic":
        wrap = (a.ra[res_all.left_index.cpu().numpy()] < 0.2) | (a.ra[res_all.left_index.cpu().numpy()] > 359.8)
        pole = a.dec[res_all.left_index.cpu().numpy()] > 89.9
        assert wrap.sum() > 100 and pole.sum() > 100, "synthetic wrap/pole regions must be populated with matches"
    for find_st, find_t in FINDS:
        res, t_s = _sky(a, b, radius_arcsec=RADIUS, find=find_t)
        for join in JOINS:
            cols, wall = st.tmatch2(A, B, out, radius=RADIUS, find=find_st, join=join)
            _assert_rows(
                _joined(res, a, b, join),
                _stilts_rows(cols, join),
                tie_l,
                tie_r,
                f"{tag} tmatch2 find={find_st} join={join}",
            )
            if join == "1and2" and find_st != "all":
                _rec(tag, f"tmatch2 matcher=sky find={find_st}", wall, t_s, a.n)
            if join == "1or2":
                _rec(
                    tag,
                    f"tmatch2 matcher=sky find={find_st} join=1or2",
                    wall,
                    t_s,
                    a.n,
                    "torchsky time is the match only;",
                )
    _mutual_modes(tmp, a, b, tag)
    # tskymatch2: the fast sky-only task gives the same best1 set
    cols, wall = st.tmatch2(A, B, out, radius=RADIUS, find="best1", task="tskymatch2")
    res, t_s = _sky(a, b, radius_arcsec=RADIUS, find="best1")
    _assert_rows(
        _joined(res, a, b, "1and2"),
        st.row_tuples(cols),
        tie_l,
        tie_r,
        f"{tag} tskymatch2",
    )
    _rec(tag, "tskymatch2 find=best1", wall, t_s, a.n)


def _mutual_modes(tmp: Path, a: Cat, b: Cat, tag: str) -> None:
    """STILTS find=best (every join) against torchsky find="mutual"."""
    A, B, out = a.table(tmp / "a.fits"), b.table(tmp / "b.fits"), tmp / "o.csv"
    cols, _ = st.tmatch2(A, B, out, radius=RADIUS, find="all")
    res_all, _ = _sky(a, b, radius_arcsec=RADIUS, find="all")
    tie_l, tie_r = _tie_rows(_pairs(res_all), st.pairs(cols), a, b, lambda i, j: RADIUS, f"{tag} find=all")
    res, t_s = _sky(a, b, radius_arcsec=RADIUS, find="mutual")
    for join in JOINS:
        cols, wall = st.tmatch2(A, B, out, radius=RADIUS, find="best", join=join)
        _assert_rows(
            _joined(res, a, b, join),
            _stilts_rows(cols, join),
            tie_l,
            tie_r,
            f"{tag} tmatch2 find=best join={join}",
        )


def _skyerr_modes(tmp: Path, a: Cat, b: Cat, tag: str) -> None:
    """tmatch2 matcher=skyerr with per-row errors on both sides (find=all, every join)."""
    A, B, out = a.table(tmp / "ea.fits"), b.table(tmp / "eb.fits"), tmp / "o.csv"
    scale = float(np.median(a.err) + np.median(b.err))
    el, er = a.err, b.err
    dev = device()
    kw = dict(left_err_arcsec=f64(el, dev), right_err_arcsec=f64(er, dev), find="all")
    res, t_s = _sky(a, b, **kw)
    cols, wall = st.tmatch2(
        A,
        B,
        out,
        radius=scale,
        find="all",
        matcher="skyerr",
        values1="ra dec err",
        values2="ra dec err",
    )
    assert len(res.left_index) > 1000, f"{tag} skyerr: too few matches to be a meaningful test"
    tie_l, tie_r = _tie_rows(_pairs(res), st.pairs(cols), a, b, lambda i, j: el[i] + er[j], f"{tag} skyerr")
    _rec(tag, "tmatch2 matcher=skyerr find=all", wall, t_s, a.n)
    for join in JOINS:
        cols, _ = st.tmatch2(
            A,
            B,
            out,
            radius=scale,
            find="all",
            matcher="skyerr",
            values1="ra dec err",
            values2="ra dec err",
            join=join,
        )
        _assert_rows(
            _joined(res, a, b, join),
            _stilts_rows(cols, join),
            tie_l,
            tie_r,
            f"{tag} skyerr join={join}",
        )


def _exact_modes(tmp: Path, a: Cat, b: Cat, tag: str) -> None:
    A, B, out = a.table(tmp / "xa.fits"), b.table(tmp / "xb.fits"), tmp / "o.csv"
    dev = device()
    ka, kb = torch.from_numpy(a.key).to(dev), torch.from_numpy(b.key).to(dev)

    def st_run(find, join="1and2"):
        args = [
            "tmatch2",
            f"in1={A}",
            f"in2={B}",
            "matcher=exact",
            "values1=key",
            "values2=key",
            f"find={find}",
            f"join={join}",
            "ifmt1=fits",
            "ifmt2=fits",
        ]
        return st.run_stilts(args, out)

    t0 = time.perf_counter()
    res = crossmatch_exact(ka, kb, find="all")
    t_s = time.perf_counter() - t0
    assert len(res.left_index) > 1000
    for join in JOINS:
        cols, wall = st_run("all", join)
        _assert_rows(
            _joined(res, a, b, join),
            _stilts_rows(cols, join),
            set(),
            set(),
            f"{tag} exact find=all join={join}",
        )
        if join == "1and2":
            _rec(tag, "tmatch2 matcher=exact find=all", wall, t_s, a.n)
    # equal scores: the pick among equals is unspecified in STILTS; the matched rows are not
    for find_st, find_t in FINDS[:3]:
        cols, _ = st_run(find_st)
        mine = crossmatch_exact(ka, kb, find=find_t)
        ts = st.row_tuples(cols)
        assert len(ts) == len(mine.left_index), f"{tag} exact find={find_st}: pair counts differ"
        if find_st == "best1":
            same = {t[0] for t in ts} == set(mine.left_index.cpu().tolist())
            assert same, f"{tag} exact best1: matched left rows differ"
        elif find_st == "best2":
            same = {t[1] for t in ts} == set(mine.right_index.cpu().tolist())
            assert same, f"{tag} exact best2: matched right rows differ"


def _identify(tmp: Path, a: Cat, tag: str) -> None:
    """tmatch1 action=identify groups vs crossmatch_self(group=True): same partition and sizes."""
    S, out = a.table(tmp / "s.fits"), tmp / "o.csv"
    cols, wall = st.tmatch1_identify(S, out, radius=RADIUS)
    dev = device()
    t0 = time.perf_counter()
    res = crossmatch_self(f64(a.ra, dev), f64(a.dec, dev), RADIUS, group=True)
    _sync(dev)
    t_s = time.perf_counter() - t0
    gid = res.group_id.cpu().numpy()
    gsize = res.group_size.cpu().numpy()
    grouped = gid >= 0
    st_gid = cols["GroupID"]
    st_grouped = ~np.isnan(st_gid)
    assert grouped.sum() > 1000, f"{tag}: too few grouped rows to be a meaningful test"
    diff = np.flatnonzero(grouped != st_grouped)
    assert diff.size == 0, f"{tag} tmatch1: {diff.size} rows grouped by only one side (first {diff[:5].tolist()})"
    pairs_ = set(zip(gid[grouped].tolist(), st_gid[st_grouped].astype(np.int64).tolist()))
    assert len(pairs_) == len({p[0] for p in pairs_}) == len({p[1] for p in pairs_}), f"{tag} tmatch1 partitions differ"
    assert len(pairs_) == res.n_groups
    np.testing.assert_array_equal(gsize[grouped], cols["GroupSize"][st_grouped].astype(np.int64))
    _rec(tag, "tmatch1 matcher=sky action=identify", wall, t_s, a.n)


def _components(cats: list[Cat]) -> tuple[np.ndarray, list[int]]:
    """Cross-catalogue friends-of-friends labels over stacked rows (torchsky pair search + scipy)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    offs = np.cumsum([0] + [c.n for c in cats])
    us, vs = [], []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            res, _ = _sky(cats[i], cats[j], radius_arcsec=RADIUS, find="all")
            us.append(res.left_index.cpu().numpy() + offs[i])
            vs.append(res.right_index.cpu().numpy() + offs[j])
    u, v = np.concatenate(us), np.concatenate(vs)
    total = int(offs[-1])
    _, label = connected_components(coo_matrix((np.ones(u.size), (u, v)), shape=(total, total)), directed=False)
    return label, [int(o) for o in offs]


def _many(tmp: Path, cats: list[Cat], tag: str) -> None:
    tables = [c.table(tmp / f"m{i}.fits") for i, c in enumerate(cats)]
    out = tmp / "o.csv"
    dev = device()
    inputs = [(f64(c.ra, dev), f64(c.dec, dev)) for c in cats]
    n_cat = len(cats)
    names = [f"id_{i + 1}" for i in range(n_cat)]

    def tuples(cols):
        return list(zip(*(np.nan_to_num(cols[k], nan=-1.0).astype(np.int64).tolist() for k in names)))

    # multimode=pairs against the first catalogue: each reference row takes its best1 partner in every
    # other catalogue (an entry of another catalogue may serve several reference rows); entries that
    # match no reference row stay alone.
    cols, wall = st.tmatchn(tables, out, radius=RADIUS, multimode="pairs", iref=1)
    t0 = time.perf_counter()
    expected = [tuple(r) for r in crossmatch_many(inputs, RADIUS, mode="pairs", reference=0).index.cpu().tolist()]
    _sync(dev)
    t_s = time.perf_counter() - t0
    assert sum(1 for r in expected if sum(i >= 0 for i in r) >= 2) > 1000, f"{tag}: too few multi-catalogue rows"
    _assert_rows(expected, tuples(cols), set(), set(), f"{tag} tmatchn pairs")
    _rec(
        tag,
        f"tmatchn multimode=pairs ({n_cat} catalogues)",
        wall,
        t_s,
        sum(c.n for c in cats),
        "torchsky time is crossmatch_many;",
    )

    # multimode=group: partitions and group sizes (STILTS keeps an arbitrary member per catalogue)
    cols, wall = st.tmatchn(tables, out, radius=RADIUS, multimode="group")
    t0 = time.perf_counter()
    res = crossmatch_many(inputs, RADIUS, mode="group")
    _sync(dev)
    t_s = time.perf_counter() - t0
    label, offs = _components(cats)
    n_groups = int(label.max()) + 1
    sizes = np.bincount(label)

    # STILTS (join=always): every input entry appears in exactly one row; rows holding two or more
    # entries are the groups (one arbitrary member per catalogue), single entries are either true
    # singletons or the other members of a group. Each multi-member component must give exactly one
    # multi-entry row, and no row may mix components.
    st_rows = tuples(cols)
    stacked = [offs[c] + i for row in st_rows for c, i in enumerate(row) if i >= 0]
    assert len(stacked) == offs[-1] and len(set(stacked)) == offs[-1], f"{tag} tmatchn group: entries not a partition of the inputs"
    multi_labels = []
    for row in st_rows:
        labs = {int(label[offs[c] + i]) for c, i in enumerate(row) if i >= 0}
        assert len(labs) == 1, f"{tag} tmatchn group (STILTS): a row mixes groups"
        if sum(i >= 0 for i in row) >= 2:
            multi_labels.append(labs.pop())
    n_multi = int((sizes >= 2).sum())
    assert len(multi_labels) == n_multi and len(set(multi_labels)) == n_multi, f"{tag} tmatchn group: STILTS groups differ from the components"
    assert (sizes[multi_labels] >= 2).all()

    # torchsky: one row per component (singletons included), group_size equals the component size
    ts_labels = []
    for row in res.index.cpu().tolist():
        labs = {int(label[offs[c] + i]) for c, i in enumerate(row) if i >= 0}
        assert len(labs) == 1, f"{tag} tmatchn group (torchsky): a row mixes groups"
        ts_labels.append(labs.pop())
    ts_labels = np.array(ts_labels)
    assert ts_labels.size == n_groups and np.unique(ts_labels).size == n_groups, f"{tag} tmatchn group (torchsky): partition differs"
    np.testing.assert_array_equal(res.group_size.cpu().numpy(), sizes[ts_labels])
    assert n_multi > 1000, f"{tag}: too few multi-member groups"
    _rec(
        tag,
        f"tmatchn multimode=group ({n_cat} catalogues)",
        wall,
        t_s,
        sum(c.n for c in cats),
    )


def test_stilts_synthetic_wrap_and_pole(tmp_path: Path) -> None:
    st.need_stilts()
    cats = _synthetic()
    _sky_modes(tmp_path, cats[0], cats[1], "synthetic")
    _skyerr_modes(tmp_path, cats[0], cats[1], "synthetic")
    _exact_modes(tmp_path, cats[0], cats[1], "synthetic")
    _identify(tmp_path, cats[0], "synthetic")
    _many(tmp_path, cats, "synthetic")


def test_stilts_real_gaia_2mass(tmp_path: Path) -> None:
    st.need_stilts()
    real = _real()
    gaia, tmass, wise = real["gaia"], real["twomass"], real["allwise"]
    tag = "gaia-dr3 x 2mass"
    _sky_modes(tmp_path, gaia, tmass, tag)
    _skyerr_modes(tmp_path, gaia, tmass, tag)
    _exact_modes(tmp_path, gaia, tmass, tag)
    _skyerr_modes(tmp_path, tmass, wise, "2mass x allwise")  # both-side errors of comparable size
    _identify(tmp_path, gaia, "gaia-dr3")
    _many(tmp_path, [gaia, tmass, wise], "gaia-dr3 x 2mass x allwise")
