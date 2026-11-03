# Draft upstream issue: HEALPix pixel-window tables for nside > 128

Status: **draft, not filed.** Target: `healpy/healpy` (tables come from `healpy-data`, generated
by HEALPix; may be forwarded to the HEALPix maintainers). Evidence: `tests/test_pixwin_parity.py`
(gate) and `tests/_pixwin_reference.py` (independent reference); see the "pixwin" section of
[coverage-matrix.md](coverage-matrix.md).

Background checked (2026-11): HEALPix documents that the windows are exact only for
`nside <= 128`; above that the temperature window is extrapolated from `nside = 128` with a
top-hat scaling in `l`, and the polarization window is the temperature one times a low-`l` factor
([HEALPix: Pixel window functions](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm)).
No healpy issue reports the size of this approximation (searched `pixwin` / `pixel window` issues;
closest: #536, #653, #1133, #1138, none about values).

---

**Title:** `pixwin` tables for nside >= 256 are extrapolated and off by up to 1.3e-5 (exact values are now cheap)

`hp.pixwin(nside)` for `nside >= 256` reads tables that HEALPix documents as extrapolated from
nside 128. Against an exact computation the error is:

| nside | max abs error T, `l <= 4 nside` | at `l` | error at `l = 1` |
|---|---|---|---|
| <= 128 | < 1.3e-13 | - | < 3e-15 |
| 256 | 7.3e-6 | 641 | 4.0e-7 |
| 512 | 1.06e-5 | 1282 | 1.35e-7 |
| 1024 | 1.21e-5 | 2564 | 3.6e-8 |
| 2048 | 1.28e-5 | 5128 | 9.2e-9 |

The tables are too high, with a smooth part growing like `(l/nside)^2` (relative error about
`4e-6 (l/nside)^2` at nside 2048) plus irregular low-`l` errors (at nside 256: +4.0e-7, +3.1e-7,
-7.6e-7, -1.1e-6 for `l = 1..4`). The polarization tables carry the same error (max 7.0e-6 to
1.28e-5). This is below most CMB error budgets but is a systematic `~1e-5` bias in
pixel-window-corrected spectra at high `l`, and the table data do not mention it.

Minimal reproducer (healpy only, `l = 1`): `w_1^2` is the pixel average of
`|(1/Omega) int_p n dOmega|^2`, and `int_p n dOmega = (1/2) oint r x dr` along the pixel boundary.

```python
import numpy as np, healpy as hp

def w1(nside, step, chunk=20000):
    npix = hp.nside2npix(nside); omega = 4 * np.pi / npix; acc = 0.0
    for start in range(0, npix, chunk):
        r = np.moveaxis(hp.boundaries(nside, np.arange(start, min(start + chunk, npix)), step=step), 1, 2)
        c = 0.5 * np.cross(r, np.roll(r, -1, axis=1)).sum(1) / omega   # pixel-mean direction
        acc += (c * c).sum()
    return np.sqrt(acc / npix)

for nside in (128, 256, 512):
    a, b = w1(nside, 32), w1(nside, 64)
    exact = b + (b - a) / 3          # Richardson: the boundary sum converges as step^-2
    print(nside, f"{hp.pixwin(nside)[1] - exact:+.1e}")
# 128 +7.7e-14 (exact table), 256 +4.0e-07, 512 +1.4e-07
```

The full exact window is no longer expensive: writing `P_l(1 - 2t)` as the terminating series
`2F1(-l, l+1; 1; t)` with `t = |n - n'|^2 / 4` separates `l` from pixel pair moments `<t^k>`,
which take `O(nside^2)` quadrature once per nside (about 1 min at nside 2048 on a laptop CPU,
so roughly 15 min at 8192) and give `w_l` for all `l <= 4 nside` to about 1e-13. An independent
check that integrates `int_p Y_lm^*` exactly in azimuth per pixel class agrees to 3e-14 (nside
256, `l <= 767`; nside 512-2048, lower `l`). We are happy to contribute regenerated
`pixel_window_n0256..n8192.fits` tables or the generator.

Questions: would regenerated tables be accepted into healpy-data (and HEALPix)? Should the
polarization windows also be recomputed exactly instead of scaled from temperature?
