import math

import numpy as np
import pytest
import torch

from torchsky._native import cpp as _native_cpp
from torchsky.sphere import healpix as hp
from torchsky.sphere import (
    alm2map,
    alm2cl,
    alm2map_der1,
    alm2map_spin,
    alm_index,
    alm_size,
    almxfl,
    beam2bl,
    bl2beam,
    gaussian_beam,
    map2alm,
    map2alm_lsq,
    map2alm_spin,
    pixwin,
    rotate_alm,
)






try:
    import healpy as _hp  # type: ignore
except Exception:  # pragma: no cover
    _hp = None


def _has_cpp_ring_fourier() -> bool:
    return (
        _native_cpp is not None
        and hasattr(_native_cpp, "_healpix_ring_fourier_modes_cpu")
        and hasattr(_native_cpp, "_healpix_ring_fourier_synthesis_cpu")
    )


def _has_cpp_spin_concat() -> bool:
    return (
        _native_cpp is not None
        and hasattr(_native_cpp, "_healpix_spin_interpolate_concat_cpu")
        and hasattr(_native_cpp, "_healpix_spin_integrate_concat_cpu")
    )


def _has_cpp_spin_ring_fused() -> bool:
    return (
        _native_cpp is not None
        and hasattr(_native_cpp, "_healpix_spin_map2alm_ring_concat_cpu")
        and hasattr(_native_cpp, "_healpix_spin_ring_finalize_cpu")
    )


def _has_cpp_spin_recurrence() -> bool:
    return (
        _native_cpp is not None
        and hasattr(_native_cpp, "_healpix_spin_interpolate_recurrence_cpu")
        and hasattr(_native_cpp, "_healpix_spin_integrate_recurrence_cpu")
    )


def _random_alm(lmax: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = np.zeros(alm_size(lmax), dtype=np.complex128)
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            scale = 1.0 / ((ell + 1) ** 2)
            if m == 0:
                val = scale * rng.normal()
            else:
                val = scale * (rng.normal() + 1j * rng.normal())
            arr[alm_index(ell, m, lmax)] = val
    return torch.from_numpy(arr)


def _random_spin_alms(
    lmax: int, spin: int = 2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    a_e = np.zeros(alm_size(lmax), dtype=np.complex128)
    a_b = np.zeros(alm_size(lmax), dtype=np.complex128)
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            if ell < spin:
                continue
            scale = 1.0 / ((ell + 1) ** 2)
            idx = alm_index(ell, m, lmax)
            if m == 0:
                a_e[idx] = scale * rng.normal()
                a_b[idx] = scale * rng.normal()
            else:
                a_e[idx] = scale * (rng.normal() + 1j * rng.normal())
                a_b[idx] = scale * (rng.normal() + 1j * rng.normal())
    return torch.from_numpy(a_e), torch.from_numpy(a_b)


















def test_map2alm_gal_cut_matches_healpy_when_installed() -> None:
    healpy = pytest.importorskip("healpy")
    nside = 8
    rng = np.random.default_rng(0)
    m = rng.normal(size=hp.nside2npix(nside))
    cut = 15.0
    got = map2alm(
        torch.as_tensor(m, dtype=torch.float64), nside=nside, lmax=8, gal_cut=cut
    )
    ref = healpy.map2alm(m, lmax=8, gal_cut=cut, use_weights=False, iter=0)
    np.testing.assert_allclose(got.numpy(), ref, atol=0.0, rtol=1e-6)














@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_almxfl_matches_healpy_short_fl() -> None:
    lmax = 8
    alm = _random_alm(lmax, seed=23)
    fl = torch.linspace(1.0, 2.0, steps=5, dtype=torch.float64)
    got = almxfl(alm, fl)
    exp = torch.from_numpy(_hp.almxfl(alm.numpy(), fl.numpy())).to(torch.complex128)  # type: ignore
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_alm2cl_matches_healpy_scalar_and_pol() -> None:
    lmax = 8
    a_t = _random_alm(lmax, seed=31)
    a_e = _random_alm(lmax, seed=32)
    a_b = _random_alm(lmax, seed=33)

    got_s = alm2cl(a_t, lmax=lmax, mmax=lmax)
    exp_s = torch.from_numpy(_hp.alm2cl(a_t.numpy(), lmax=lmax, mmax=lmax)).to(  # type: ignore
        torch.float64
    )
    torch.testing.assert_close(got_s, exp_s, atol=1e-12, rtol=0.0)

    alms = torch.stack([a_t, a_e, a_b], dim=0)
    got_p = alm2cl(alms, lmax=lmax, mmax=lmax)
    exp_p = torch.from_numpy(_hp.alm2cl(alms.numpy(), lmax=lmax, mmax=lmax)).to(  # type: ignore
        torch.float64
    )
    torch.testing.assert_close(got_p, exp_p, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_pixwin_matches_healpy() -> None:
    nside = 16
    lmax = 12
    got = pixwin(nside, pol=False, lmax=lmax)
    exp = torch.from_numpy(
        _hp.pixwin(nside, pol=False, lmax=lmax).astype("float64").copy()  # type: ignore
    ).to(torch.float64)
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)

    got_t, got_p = pixwin(nside, pol=True, lmax=lmax)
    exp_t, exp_p = _hp.pixwin(nside, pol=True, lmax=lmax)  # type: ignore
    torch.testing.assert_close(
        got_t,
        torch.from_numpy(exp_t.astype(float)).to(torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        got_p,
        torch.from_numpy(exp_p.astype(float)).to(torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_bl2beam_and_beam2bl_close_to_healpy() -> None:
    lmax = 24
    theta = torch.linspace(0.0, 0.2, steps=256, dtype=torch.float64)
    bl = gaussian_beam(math.radians(0.8), lmax=lmax)

    beam_tf = bl2beam(bl, theta)
    beam_hp = torch.from_numpy(_hp.bl2beam(bl.numpy(), theta.numpy())).to(torch.float64)  # type: ignore
    torch.testing.assert_close(beam_tf, beam_hp, atol=2e-6, rtol=2e-6)

    bl_tf = beam2bl(beam_tf, theta, lmax=lmax)
    bl_hp = torch.from_numpy(_hp.beam2bl(beam_hp.numpy(), theta.numpy(), lmax=lmax)).to(  # type: ignore
        torch.float64
    )
    torch.testing.assert_close(bl_tf, bl_hp, atol=1e-4, rtol=2e-4)
















@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_map2alm_lsq_torch_residual_close_to_healpy_on_random_map() -> None:
    nside = 16
    lmax = 8
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(13)
    m = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)

    _, rel_tf, _ = map2alm_lsq(
        m,
        lmax=lmax,
        mmax=lmax,
        nside=nside,
        pol=False,
        tol=1e-12,
        maxiter=3,
    )
    _, rel_hp, _ = _hp.map2alm_lsq(  # type: ignore
        m.numpy(), lmax=lmax, mmax=lmax, pol=False, tol=1e-12, maxiter=3
    )
    assert rel_tf == pytest.approx(float(rel_hp), rel=0.0, abs=1e-8)


@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_healpy_reference_alm2map_map2alm_parity_small_case() -> None:
    nside = 16
    lmax = 8
    alm = _random_alm(lmax, seed=5)

    map_tf = alm2map(alm, nside=nside, lmax=lmax)
    map_hp = torch.from_numpy(
        _hp.alm2map(alm.numpy(), nside=nside, lmax=lmax, mmax=lmax, pol=False)  # type: ignore
    ).to(torch.float64)
    diff = torch.linalg.norm(map_tf - map_hp) / torch.linalg.norm(map_hp).clamp_min(
        1e-15
    )
    assert float(diff.item()) < 1.5e-1

    rec_tf = map2alm(map_hp, nside=nside, lmax=lmax)
    rec_hp = torch.from_numpy(
        _hp.map2alm(map_hp.numpy(), lmax=lmax, mmax=lmax, iter=0, pol=False)  # type: ignore
    ).to(torch.complex128)
    adiff = torch.linalg.norm(rec_tf - rec_hp) / torch.linalg.norm(rec_hp).clamp_min(
        1e-15
    )
    assert float(adiff.item()) < 1.5e-1
























@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_gaussian_beam_pol_matches_healpy() -> None:
    fwhm = math.radians(1.0)
    lmax = 24
    got = gaussian_beam(fwhm, lmax=lmax, pol=True)
    exp = torch.from_numpy(_hp.gauss_beam(fwhm, lmax=lmax, pol=True)).to(torch.float64)  # type: ignore
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)












@pytest.mark.parametrize("spin", [1, 2, 3])
@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_spin_map2alm_alm2map_parity(spin: int) -> None:
    nside = 16
    lmax = 10
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(7)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)

    a1, a2 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    qu_rec = alm2map_spin((a1, a2), nside=nside, spin=spin, lmax=lmax)
    assert getattr(qu_rec, "shape", None) == getattr(qu, "shape", None)

    hp_a1, hp_a2 = _hp.map2alm_spin(qu.numpy(), spin=spin, lmax=lmax)  # type: ignore
    torch.testing.assert_close(
        a1, torch.from_numpy(hp_a1).to(torch.complex128), atol=3e-10, rtol=0.0
    )
    torch.testing.assert_close(
        a2, torch.from_numpy(hp_a2).to(torch.complex128), atol=3e-10, rtol=0.0
    )

    hp_qu = _hp.alm2map_spin([hp_a1, hp_a2], nside=nside, spin=spin, lmax=lmax)  # type: ignore
    torch.testing.assert_close(
        qu_rec,
        torch.from_numpy(np.asarray(hp_qu)).to(torch.float64),
        atol=5e-10,
        rtol=0.0,
    )


@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_spin_high_l_parity_spotcheck() -> None:
    nside = 64
    lmax = 96
    spin = 2
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(2026)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)

    ae, ab = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    hp_ae, hp_ab = _hp.map2alm_spin(qu.numpy(), spin=spin, lmax=lmax)  # type: ignore
    hp_ae_t = torch.from_numpy(hp_ae).to(torch.complex128)
    hp_ab_t = torch.from_numpy(hp_ab).to(torch.complex128)
    rel_ae = torch.linalg.norm(torch.as_tensor(ae) - hp_ae_t) / torch.linalg.norm(
        hp_ae_t
    ).clamp_min(1e-15)
    rel_ab = torch.linalg.norm(torch.as_tensor(ab) - hp_ab_t) / torch.linalg.norm(
        hp_ab_t
    ).clamp_min(1e-15)
    assert float(rel_ae) < 5e-10
    assert float(rel_ab) < 5e-10

    rec = alm2map_spin((ae, ab), nside=nside, spin=spin, lmax=lmax)
    hp_rec = _hp.alm2map_spin(  # type: ignore
        [ae.numpy(), ab.numpy()], nside=nside, spin=spin, lmax=lmax
    )
    hp_rec_t = torch.from_numpy(np.asarray(hp_rec)).to(torch.float64)
    rel_rec = torch.linalg.norm(rec - hp_rec_t) / torch.linalg.norm(hp_rec_t).clamp_min(
        1e-15
    )
    assert float(rel_rec) < 5e-10














































def _force_spin_ring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP", "1")
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RECURRENCE_CPP", "1")








@pytest.mark.skipif(_hp is None, reason="healpy not available")
@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("ishioka", ["0", "1"])
def test_spin_sht_matches_healpy_ishioka_toggle(
    spin: int, ishioka: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    nside = 16
    lmax = 3 * nside - 1
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(431)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)
    nalm = alm_size(lmax)
    alm_e = (rng.normal(size=nalm) + 1j * rng.normal(size=nalm)).astype(np.complex128)
    alm_b = (rng.normal(size=nalm) + 1j * rng.normal(size=nalm)).astype(np.complex128)
    for ell in range(spin):
        for m in range(ell + 1):
            alm_e[alm_index(ell, m, lmax)] = 0.0
            alm_b[alm_index(ell, m, lmax)] = 0.0
    _force_spin_ring(monkeypatch)
    monkeypatch.setenv("TORCHSKY_SPIN_ISHIOKA", ishioka)

    ae, ab = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    hp_ae, hp_ab = _hp.map2alm_spin(qu.numpy(), spin=spin, lmax=lmax)  # type: ignore
    torch.testing.assert_close(
        ae, torch.from_numpy(hp_ae).to(torch.complex128), atol=5e-10, rtol=0.0
    )
    torch.testing.assert_close(
        ab, torch.from_numpy(hp_ab).to(torch.complex128), atol=5e-10, rtol=0.0
    )

    qu_ts = alm2map_spin(
        (torch.from_numpy(alm_e), torch.from_numpy(alm_b)),
        nside=nside,
        spin=spin,
        lmax=lmax,
    )
    hp_qu = _hp.alm2map_spin((alm_e, alm_b), nside, spin, lmax)  # type: ignore
    torch.testing.assert_close(
        qu_ts,
        torch.from_numpy(np.asarray(hp_qu)).to(torch.float64),
        atol=5e-10,
        rtol=0.0,
    )


@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_spin_sht_mmax_lt_lmax_matches_healpy(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 20
    mmax = 8
    spin = 2
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(441)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)
    _force_spin_ring(monkeypatch)
    monkeypatch.setenv("TORCHSKY_SPIN_ISHIOKA", "1")
    ae, ab = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax, mmax=mmax)
    hp_ae, hp_ab = _hp.map2alm_spin(  # type: ignore
        qu.numpy(), spin=spin, lmax=lmax, mmax=mmax
    )
    torch.testing.assert_close(
        ae, torch.from_numpy(hp_ae).to(torch.complex128), atol=5e-10, rtol=0.0
    )
    torch.testing.assert_close(
        ab, torch.from_numpy(hp_ab).to(torch.complex128), atol=5e-10, rtol=0.0
    )






@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_rotate_alm_matches_healpy() -> None:
    lmax = 8
    rng = np.random.default_rng(7)
    nalm = alm_size(lmax)
    alm_np = np.asarray(
        rng.normal(size=nalm) + 1j * rng.normal(size=nalm), dtype=np.complex128
    )
    psi, theta, phi = 0.3, 0.5, -0.2
    hp_alm = alm_np.copy()
    _hp.rotate_alm(hp_alm, psi, theta, phi)
    ts = rotate_alm(torch.from_numpy(alm_np.copy()), psi, theta, phi, lmax=lmax)
    torch.testing.assert_close(ts, torch.from_numpy(hp_alm), atol=1e-8, rtol=0.0)




@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_alm2map_der1_matches_healpy() -> None:
    nside = 8
    lmax = 6
    rng = np.random.default_rng(11)
    nalm = alm_size(lmax)
    alm_np = np.asarray(
        rng.normal(size=nalm) + 1j * rng.normal(size=nalm), dtype=np.complex128
    )
    for ell in range(lmax + 1):
        idx = _hp.Alm.getidx(lmax, ell, 0)
        alm_np[idx] = alm_np[idx].real
    hp_m, hp_dt, hp_dp = _hp.alm2map_der1(alm_np, nside, lmax=lmax)
    m, dt, dp = alm2map_der1(torch.from_numpy(alm_np), nside, lmax=lmax)
    torch.testing.assert_close(
        m, torch.from_numpy(np.asarray(hp_m)), atol=1e-8, rtol=0.0
    )
    torch.testing.assert_close(
        dt, torch.from_numpy(np.asarray(hp_dt)), atol=1e-8, rtol=0.0
    )
    torch.testing.assert_close(
        dp, torch.from_numpy(np.asarray(hp_dp)), atol=1e-8, rtol=0.0
    )
