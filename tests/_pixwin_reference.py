"""Independent HEALPix temperature pixel window, NumPy only (no healpy, no torchsky).

The window is the pixel average of each pixel's harmonic power,

    w_l^2 = (1/N_pix) sum_p (4 pi / (2l+1)) sum_m |b_lm(p)|^2 / Omega^2,
    b_lm(p) = int_p Y_lm^* dOmega,

which is the addition-theorem form of the pair integral torchsky evaluates
(``(1/Omega^2) iint_{p x p} P_l(n . n')``). The method here is different from
torchsky's (pair moments, a hypergeometric series in the pair separation and
2-d Gauss-Legendre rules on the pixel): every ``b_lm`` is integrated exactly in
azimuth, ``int e^{-im phi} dphi`` over the pixel's azimuth interval at fixed
``z``, and with Gauss-Legendre in the ring coordinate ``s = x + y`` of the face
(``z`` is constant on lines of constant ``s``). The azimuth limits are affine
in the pixel's face coordinates, ``z(s)`` is a polynomial on either side of the
polar-cap boundary, and the pixel is split at its centre line (where the
limits have their kink, and where the cap boundary falls for the straddling
ring), so the 1-d integrand is smooth on every piece.

Pixel classes: polar-cap pixels of face 0 (``ix + iy >= nside - 1``) with the
mirror ``ix >= iy``, times 4 faces and 2 hemispheres (cap pixels of one ring
are not congruent, so each class is integrated); one pixel per equatorial
ring (equatorial pixels of a ring are congruent under rotation about the polar
axis). The geometry follows the continuous HEALPix projection (Gorski et al.
2005; ``xyf2loc`` in HEALPix C++).

The cost is ``O(L^2 x nodes x nside^2 / 4)`` (every cap pixel needs all
``(l, m)``), so callers bound ``lmax`` at large nside.
"""

from __future__ import annotations

import math

import numpy as np

_JRLL = (2, 3)  # face 0 (north), face 4 (equatorial)
_JPLL = (1, 0)


def _ring_geometry(
    nside: int, face_row: int, ix: np.ndarray, iy: np.ndarray, order: int
):
    """Nodes and azimuth limits for pixels that share ``ix + iy`` (one ring).

    Returns ``z`` (2*order,), ``wz`` (2*order,) = Gauss weight * dz/ds, and
    ``phi1``, ``phi2`` (J, 2*order).
    """

    h = 1.0 / nside
    ring_sum = int(ix[0] + iy[0])
    if not np.all(ix + iy == ring_sum):
        raise ValueError("pixels of one group must share ix + iy")
    x_gl, w_gl = np.polynomial.legendre.leggauss(order)
    lo = ring_sum * h
    # Two pieces: [lo, lo + h] and [lo + h, lo + 2h].
    s = np.concatenate((lo + 0.5 * h * (x_gl + 1.0), lo + h + 0.5 * h * (x_gl + 1.0)))
    ws = np.concatenate((0.5 * h * w_gl, 0.5 * h * w_gl))
    jr = _JRLL[face_row] - s
    cap = jr < 1.0
    nr = np.where(cap, jr, 1.0)
    z = np.where(cap, 1.0 - jr * jr / 3.0, (2.0 - jr) * (2.0 / 3.0))
    dz_ds = np.where(cap, 2.0 * jr / 3.0, 2.0 / 3.0)
    a = ix[:, None] * h
    b = iy[:, None] * h
    x_lo = np.maximum(a, s[None, :] - b - h)
    x_hi = np.minimum(a + h, s[None, :] - b)
    d_lo = 2.0 * x_lo - s[None, :]
    d_hi = 2.0 * x_hi - s[None, :]
    scale = (math.pi / 4.0) / nr[None, :]
    phi1 = scale * (_JPLL[face_row] * nr[None, :] + d_lo)
    phi2 = scale * (_JPLL[face_row] * nr[None, :] + d_hi)
    return z, ws * dz_ds, phi1, phi2


def _groups(nside: int, order: int):
    """(z, wz, phi1, phi2, pixel weights) per ring group; weights tile 12 nside^2."""

    groups = []
    # Polar cap and the straddling ring: face 0, ix + iy in [nside-1, 2 nside-2].
    for ring_sum in range(nside - 1, 2 * nside - 1):
        ix = np.arange(max(0, ring_sum - (nside - 1)), nside)
        iy = ring_sum - ix
        keep = (iy >= 0) & (iy < nside) & (ix >= iy)
        ix, iy = ix[keep], iy[keep]
        weight = 8.0 * np.where(ix == iy, 1.0, 2.0)
        groups.append((*_ring_geometry(nside, 0, ix, iy, order), weight))
    # Equatorial rings nside+1 .. 2 nside: one face-4 pixel each, ring i has ix+iy = 3 nside - 1 - i.
    for ring in range(nside + 1, 2 * nside + 1):
        ring_sum = 3 * nside - 1 - ring
        ix0 = min(ring_sum, nside - 1)
        ix = np.array([ix0])
        iy = np.array([ring_sum - ix0])
        weight = np.array([4.0 * nside * (2.0 if ring < 2 * nside else 1.0)])
        groups.append((*_ring_geometry(nside, 1, ix, iy, order), weight))
    return groups


def _lambda_m(m: int, lmax: int, z: np.ndarray) -> np.ndarray:
    """Orthonormal associated Legendre ``lambda_lm(z)`` for ``l = m..lmax`` (sign dropped)."""

    out = np.zeros((lmax - m + 1, z.size))
    log_norm = 0.5 * math.log((2 * m + 1) / (4.0 * math.pi)) + 0.5 * sum(
        math.log((2 * k - 1) / (2 * k)) for k in range(1, m + 1)
    )
    sin2 = np.clip((1.0 - z) * (1.0 + z), 0.0, None)
    with np.errstate(divide="ignore"):
        out[0] = np.exp(log_norm + 0.5 * m * np.log(sin2)) if m else math.exp(log_norm)
    if lmax == m:
        return out
    out[1] = math.sqrt(2 * m + 3) * z * out[0]
    a_prev = math.sqrt(2 * m + 3)
    for idx in range(2, lmax - m + 1):
        ell = m + idx
        a = math.sqrt((4.0 * ell * ell - 1.0) / (ell * ell - m * m))
        out[idx] = a * (z * out[idx - 1] - out[idx - 2] / a_prev)
        a_prev = a
    return out


def reference_pixwin(nside: int, lmax: int, order: int = 12) -> np.ndarray:
    """Temperature pixel window ``w_l`` for ``0 <= l <= lmax`` (float64)."""

    groups = _groups(nside, order)
    npix = 12 * nside * nside
    total_weight = sum(float(g[4].sum()) for g in groups)
    if abs(total_weight - npix) > 1e-9 * npix:
        raise RuntimeError("pixel classes do not tile the sphere")
    area = sum(float(g[4] @ ((g[3] - g[2]) @ g[1])) for g in groups)
    if abs(area - 4.0 * math.pi) > 1e-12:
        raise RuntimeError(f"pixel areas sum to {area}, not 4 pi")

    z_all = np.concatenate([g[0] for g in groups])
    nodes = groups[0][0].size
    acc = np.zeros(lmax + 1)
    e1 = [np.exp(-1j * g[2]) for g in groups]
    e2 = [np.exp(-1j * g[3]) for g in groups]
    p1 = [np.ones_like(e) for e in e1]
    p2 = [np.ones_like(e) for e in e2]
    for m in range(lmax + 1):
        lam = _lambda_m(m, lmax, z_all)
        for gi, (_z, wz, phi1, phi2, weight) in enumerate(groups):
            if m == 0:
                f = (phi2 - phi1).astype(np.complex128)
            else:
                p1[gi] *= e1[gi]
                p2[gi] *= e2[gi]
                f = (p1[gi] - p2[gi]) / (1j * m)
            g = f * wz[None, :]
            block = lam[:, gi * nodes : (gi + 1) * nodes] @ g.T  # (l, pixels)
            power = (block.real**2 + block.imag**2) @ weight
            acc[m:] += power if m == 0 else 2.0 * power
    ell = np.arange(lmax + 1)
    return np.sqrt(acc * npix / (4.0 * math.pi * (2 * ell + 1)))
