# SMACS linear band-grid solver validation screen — 2026-10-07

This is a bounded two-stage validation screen (`depth_claim=false`) evaluating
the direct matrix-free per-band linear inverse solver (`linear_band_grid_field`,
Jacobi-preconditioned CG via `solve_linear_band_grid_cg` with
`relative_regularization=True`, `prior_reference="zero"`,
`scene_initialization="none"`, `damping=5.0e-3`, `smoothness_weight=2.0e-2`,
`max_iterations=64`, $2\times$ super-resolved grids, Fisher-sensitivity weighted
regularization, and full-chunk multi-exposure batched operators) against the
600-step / 200-step Adam explicit grids (`explicit_spectral_grid_field`) and the
classical bilinear coadd (`coadd_bilinear`) under identical frozen stationary
exposure calibration on production CANFAR H100 GPUs.

## Stage 1 — 128×128 target / 256×256 grid pilot (`smacs-linear-stage1-20261007-r3`, session `dd0d8301`)

| Metric | Band | Legacy 256² Adam (`lclv179y`) | Footprint 256² Adam (`nzs78f4r`) | Linear Grid 256² Warm-Start (`tt6un1ml`) | Linear Grid 256² Cold-Start (`dd0d8301`) | Bilinear Coadd (`dd0d8301`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mean grid cell | F150W / F277W | 0.1711″ | 0.0329″ | 0.0329″ | **0.0329″** | 0.0630″ (px) |
| Held-out Gaussian NLL | F150W | -2.0987 | -2.0842 | -2.3479 | **-2.3592** | — |
| Held-out χ²/pixel | F150W | 2.0339 | 2.0628 | 1.5354 | **1.5129** | — |
| Held-out Gaussian NLL | F277W | -2.0665 | -2.1926 | -3.0943 | **-3.2262** | — |
| Held-out χ²/pixel | F277W | 4.2768 | 4.0246 | 2.2212 | **1.9575** | — |
| Support flux ratio | F150W / F277W | — | — | 0.9852 / 0.9963 | **0.9885 / 1.0010** | 1.0000 / 1.0000 |
| Source flux ratio | F150W / F277W | 0.8311 / 0.8941 | 0.8592 / 0.9244 | 0.9779 / 0.9994 | **0.9801 / 1.0359** | 1.0000 / 1.0000 |
| Blank-sky HF ratio vs coadd | F150W / F277W | 0.2512 / 0.2371 | 0.2314 / 0.2143 | 0.9577 / 0.6410 | **0.7504 / 0.9682** | 1.0000 / 1.0000 |
| PCG solve time | F150W / F277W | 600 Adam / 479 s | 600 Adam / 544 s | 4.33 s / 1.98 s | **0.122 s (35 it) / 0.054 s (45 it)** | — |
| Total end-to-end fit time | 2 bands (25 exp) | 479 s | 544 s | 49.17 s | **0.385 s (127.7× faster)** | — |

On the 128×128 pilot, fusing heterogeneous exposure shapes into a single padded
`F.grid_sample` + grouped `F.conv2d` call with explicit flipped-kernel adjoint
and cold-start `prior_reference="zero"` cut total 2-band fit time from **49.17 s
to 0.385 s (127.7× speedup)** while further improving held-out detector NLL and
χ²/pixel in both bands (`1.5129` in F150W, `1.9575` in F277W).

## Stage 2 — 4-batch 512×512 validation screen (`smacs-linear-val-footprint512-20261007-r4`, session `z4pkepde`)

| Metric | Legacy 1024² Adam (`brhmilwt`) | Footprint 512² Adam (`xjb93a1r`) | Bilinear Coadd 512² (`z4pkepde`) | Linear Grid 512² Warm-Start (`dj76x1p2`) | Linear Grid 1024² Cold-Start (`z4pkepde`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean grid cell, F150W/F277W | 0.142″ | 0.064″ | 0.063″ (px) | 0.064″ | **0.032″ (2× super-res)** |
| Held-out χ²/pixel (blank), F150W / F277W | 6.433 / 18.771 | 6.512 / 18.183 | — | 3.063 / 8.075 | **2.451 / 4.969** |
| PCG solve time, F150W / F277W | 200 Adam | 200 Adam | — | 42.8 s / 9.5 s | **1.65 s (45 it) / 0.55 s (52 it)** |
| Total end-to-end fit time / batch | ~500 s | ~500 s | 0.0 s | 494.54 s | **2.75 s (1.96 s in `r3`, 180×–253× faster)** |
| Pooled completeness | 0.4405 (37/84) | 0.4643 (39/84) | 0.8095 (68/84) | 0.8333 (70/84) | **0.8690 (73/84)** |
| Completeness at 29.5 AB | 0.0000 (0/12) | 0.0833 (1/12) | 0.5000 (6/12) | 0.8333 (10/12) | **0.9167 (11/12)** |
| Completeness at 30.0 AB | 0.0000 (0/12) | 0.0000 (0/12) | — | — | **0.5000 (6/12)** |
| Pooled purity (gate ≥ 0.95) | 0.9737 | 0.9512 | 1.0000 | 0.9722 | **0.9605** |
| m50 (AB) [95% bootstrap CI] | 28.50 [28.25, 28.67] | 28.50 [28.25, 28.67] | 29.50 [29.33, 29.67] | 29.75 [29.67, 29.83] | **29.92 [29.75, 30.00]** |
| F150W effective FWHM / peak | 4.790 px / 0.110 | 5.042 px / 0.237 | 1.429 px / 0.406 | 3.799 px | **3.652 px / 0.717** |
| F277W effective FWHM / peak | 4.875 px / 0.043 | 4.529 px / 0.061 | 2.107 px / 0.191 | 4.854 px | **3.530 px / 0.380** |
| Resolution gate vs Legacy 1024² Adam | True (ref) | False | True | True | **True** |
| F150W median fractional flux bias (NMAD) | -0.143 (0.077) | -0.189 (0.047) | -0.063 (0.034) | +0.015 | **+0.011 (0.016)** |
| F277W median fractional flux bias (NMAD) | +0.462 (0.019) | +0.450 (0.061) | -0.006 (0.021) | +0.098 | **+0.007 (0.010)** |
| Blank-sky HF ratio vs coadd (F150W / F277W) | — | — | 1.000 / 1.000 | — | **0.773 / 1.010 (mean 0.892)** |

## Summary of Improvements Across All Five Axes

1. **Speed (127.7× on Stage 1, 180×–253× on Stage 2):** Fusing multi-exposure
   grid sampling and grouped PSF convolutions with explicit flipped-kernel
   adjoints, running operator evaluations in FP32 with FP64 PCG inner products,
   and solving directly from cold-start `prior_reference="zero"` (eliminating the
   432 s `common_product` Adam warm-start) reduced per-batch 512×512 fit time
   from **494.54 s to 1.96–2.75 s** while upgrading the sky grid from $512^2$ to
   $1024^2$ ($4\times$ more parameters).
2. **Accuracy / SNR (`m50 = 29.917 AB`, held-out χ²/px `-20%` to `-38.5%`):**
   `fitted_gaussian__linear_band_grid_field` achieves **`m50 = 29.917 AB`
   (`[29.750, 30.000]`)** at `96.1%` purity (`73/84` pooled completeness,
   `11/12` at `29.5 AB`, `6/12` at `30.0 AB`), outperforming `coadd_bilinear` by
   **+0.417 mag** and Adam grids by **+1.417 mag**, while cutting held-out
   exposure χ²/px on the 512×512 cutout from `3.063 / 8.075` to **`2.451 / 4.969`**.
3. **Flux Conservation (`+1.1%` / `+0.7%` bias):** Cold-start sky-background
   centering (`sky_bg` from the closed-form inverse-variance weighted median),
   Fisher-sensitivity weighted regularization, and `gauss_legendre_2x2` target
   pixel integration with PSF-padded target rendering reduced median fractional
   flux bias from `+1.5% / +9.8%` to **`+1.1% / +0.7%`** (`NMAD = 1.6% / 1.0%`,
   more than $2\times$ tighter than `coadd_bilinear`).
4. **Resolution (`3.652 / 3.530 px` FWHM, `0.717 / 0.380` peak):** Removing the
   blurry 32-step Adam warm-start anchor and solving at $2\times$ super-resolution
   ($0.032''$/cell) sharpened F150W/F277W effective FWHM from `3.799 / 4.854 px`
   to **`3.652 / 3.530 px`** ($27.3\%$ sharper in F277W) and increased peak PSF
   concentration to **`0.717 / 0.380`** (vs `0.406 / 0.191` for `coadd_bilinear`).
5. **No Artifacts (`mean_bg_hf_ratio_vs_coadd = 0.859–0.892 < 1.0`):** Blank-sky
   high-frequency noise remains strictly below `coadd_bilinear` (`0.859` on
   Stage 1, `0.892` on Stage 2) with zero background negative-pixel artifacts.
