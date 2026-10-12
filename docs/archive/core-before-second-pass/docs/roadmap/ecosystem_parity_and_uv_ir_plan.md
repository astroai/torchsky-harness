# Ecosystem Parity Matrix & UV-to-Far-IR Completion Plan

> **Living Document**: Update the status column (`[x] Shipped`, `[~] Partial`, `[ ] Planned`) and entry-point links in the tables below whenever a capability lands or gains an upstream parity/benchmark gate. See also [`../compatibility.md`](../compatibility.md) and [`status.md`](status.md).

---

## Part I — Ecosystem Equivalence Matrix (`sphere`, `wcs`, `map`)

### 1. `torchsky.sphere` vs. Spherical & HEALPix Libraries

| Peer Library | Peer Capability | `torchsky` Equivalent | Status | Notes / Remaining Delta |
| :--- | :--- | :--- | :---: | :--- |
| **`healpy`** | Pixel indexing & conversions (`ang2pix`, `pix2ang`, `vec2pix`, `pix2vec`, `ang2vec`, `vec2ang`, `ring2nest`, `nest2ring`, `reorder`) | `torchsky.sphere.healpix` & `torchsky.sphere.compat` (`torchsky.sphere.healpy`) | `[x] Shipped` | CPU C++ SIMD + GPU/MPS ATen |
| **`healpy`** | Grid metadata (`nside2npix`, `npix2nside`, `nside2order`, `order2nside`, `nside2pixarea`, `nside2resol`, `max_pixrad`, `isnsideok`, `isnpixok`) | `torchsky.sphere.healpix` | `[x] Shipped` | Exact parity |
| **`healpy`** | Bilinear interpolation & 8-neighbors (`get_interp_weights`, `get_interp_val`, `get_all_neighbours`, `boundaries`, `ud_grade`) | `torchsky.sphere.healpix`, `torchsky.sphere.sample_healpix_map` | `[x] Shipped` | Differentiable map sampling (`sample_healpix_map`, `sample_multiband_healpix`, `sample_multiwavelength_healpix`) |
| **`healpy`** | Spherical Harmonics & CMB spectra (`map2alm`, `map2alm_spin`, `map2alm_lsq`, `alm2map`, `alm2map_spin`, `alm2map_der1`, `anafast`, `synalm`, `synfast`, `alm2cl`, `almxfl`, `pixwin`, `gaussian_beam`, `beam2bl`, `bl2beam`, `smoothing`, `rotate_alm`, `Rotator`, `Alm`, `fit_dipole`, `remove_dipole`, `fit_monopole`, `remove_monopole`) | `torchsky.sphere.spectral`, `torchsky.sphere.compat` | `[x] Shipped` | Fused Ishioka/Wigner C++ on CPU; ATen ring FFT on CUDA/MPS; RING & pixel weights |
| **`hpgeom`** | Vectorized sky region queries (`query_circle`, `query_disc`, `query_box`, `query_ellipse`, `query_polygon`, `query_strip`, inclusive `fact=4` & arbitrary `fact`, `return_pixel_ranges`) | `torchsky.sphere.healpix`, `torchsky.sphere.compat` | `[x] Shipped` | Exact `hpgeom` query semantics |
| **`astropy-healpix`** | `lonlat_to_healpix`, `healpix_to_lonlat`, `bilinear_interpolation_weights`, `neighbours`, `cone_search_lonlat` | `torchsky.sphere.healpix`, `torchsky.wcs.WCS` (`HPX`) | `[x] Shipped` | Validated against `astropy-healpix` oracle |
| **`cdshealpix` (Python & Rust)** | Morton bit-interleaving (`spread_bits`, `compact_bits`), `uniq` (`nest2uniq`, `uniq2nest`), hierarchical range algebra (`pixels_to_pixel_ranges`, `upgrade_pixel_ranges`, `degrade_pixel_ranges`), cone/polygon/box/ellipse MOC construction | `torchsky.sphere.healpix`, `torchsky.sphere.MOC`, `torchsky.sphere.STMOC` | `[x] Shipped` | Optional: explicit `BMOC` per-cell boundary-vs-interior flag array |
| **`healsparse`** | Two-level coverage + sparse block maps (`HealSparseMap`), `WideBitMask`, `ud_grade`, `get_values_pix`/`pos`, `valid_pixels`, `coverage_map`, `SkyMaskPipe`, `sum_union`, `sum_intersection`, `.hs` FITS format | `torchsky.sphere.sparse` (`HealSparseMap`, `SparseMap`, `WideBitMask`, `SkyMaskPipe`) | `[x] Shipped` | Cross-opens `.hs` files with upstream `healsparse` |
| **`mocpy`** | IVOA Spatial `MOC` & Space-Time `STMOC` (`from_cone`, `from_polygon`, `contains`, `union`, `intersection`, `difference`, `complement`, FITS/JSON/ASCII serialization) | `torchsky.sphere.MOC`, `torchsky.sphere.STMOC` | `[x] Shipped` | Native 61-bit µs time × NUNIQ spatial intervals |
| **`spherical_geometry`** | Great-circle polygon topology (`SphericalPolygon`, `SphericalMultiPolygon`, `contains`, `intersects`, Girard `area`, `union`, `intersection`, `difference`) | `torchsky.sphere.geom` (`SphericalPolygon`, `NativeExactSphericalRegion`, `to_exact_region`) | `[x] Shipped` | $\le 10^{-12}\text{ sr}$ exact area parity vs `spherical_geometry`, plus GPU ATen vectorized `contains`/`intersects` |

---

### 2. `torchsky.wcs` vs. WCS & Astrometric Coordinate Libraries

| Peer Library | Peer Capability | `torchsky` Equivalent | Status | Notes / Remaining Delta |
| :--- | :--- | :--- | :---: | :--- |
| **`wcslib` / `astropy.wcs`** | **WCS Paper II** celestial projections (all 28 standard projections + legacy aliases): `AZP`, `SZP`, `TAN`, `STG`, `SIN`, `NCP`, `ARC`, `ZPN`, `ZEA`, `AIR`, `CYP`, `CEA`, `CAR`, `MER`, `SFL`, `GLS`, `PAR`, `MOL`, `AIT`, `COP`, `COE`, `COD`, `COO`, `BON`, `PCO`, `TSC`, `CSC`, `QSC`, `HPX`, `XPH` | `torchsky.wcs.WCS` (`pixel_to_world`, `world_to_pixel`) | `[x] Shipped` | 30/30 codes forward & inverse on CPU/CUDA/MPS, differentiable |
| **`wcslib` / `astropy.wcs`** | **WCS Paper III** spectral axes (`FREQ`, `WAVN`, `VRAD`, `FELO`, `VOPT`, `ZOPT`, `AWAV`, `WAVE`, `BETA`, `ENER`, `-LOG`, `-GRI`, `-GRA`, `-TAB`, air↔vacuum, rest-frequency) | `torchsky.wcs.spectral`, 3D/ND `torchsky.wcs.WCS` | `[x] Shipped` | Full spectral coordinate conversions |
| **`wcslib` / `astropy.wcs`** | **WCS Paper IV** & non-standard polynomial distortions: `SIP` (`A_p_q`, `B_p_q`, `AP_p_q`, `BP_p_q`), `TPV`, `TNX`, `ZPX`, `DSS`, and lookup tables (`DET2IM1/2`, `CPDIS1/2`) | `torchsky.wcs.WCS`, `torchsky.wcs.DistortionLookupTable` | `[x] Shipped` | Full sequential pipeline `DET2IM -> SIP -> CPDIS -> CD/PC` |
| **`wcslib` / `astropy.wcs`** | **Tabular (`-TAB`) WCS** (1D separable and 2D coupled lookup tables) | `torchsky.wcs.TabularLookupWCS`, `torchsky.wcs.CoupledTabularLookupWCS` | `[x] Shipped` | Fast GPU/CPU inverse ($5 \times 10^{-11}\text{ px}$ accuracy) |
| **`gwcs`** | Generalized WCS pipeline (JWST, Roman, DKIST): detector $\to$ distortion $\to$ tangent/celestial/spectral frame | `torchsky.wcs.sample_gwcs_detector_to_tangent`, `FrozenDetectorEpochTables`, `DetectorToSkyTransform`, `IFUSliceWCS`, `torchsky.wcs.gwcs_live` | `[x] Shipped` | 2D imaging, multi-epoch frozen tables, & 3D IFU slice-by-slice disjoint `IFUSliceWCS` shipped |
| **`astropy.coordinates` / `ERFA`** | Celestial & observed coordinate frames (`ICRS`, `FK5`, `FK4`, `Galactic`, `Ecliptic`, `EclipticOfDate`, `Galactocentric`, `GCRS`, `CIRS`, `TETE`, `ITRS`, `AltAz`, IAU 2006/2000A precession-nutation, aberration, refraction, DCR, ephemeris tracking, LSR RV, 6D space motion + Jacobians) | `torchsky.wcs.frames`, `torchsky.wcs.observed`, `torchsky.wcs.iau2006` (`dcr_offset`, `apply_dcr`, `apply_ephemeris_offset`), `torchsky.wcs.CatalogState` | `[x] Shipped` | Tensor-native frame conversions, DCR, moving-target offsets, & epoch propagation |
| **`pyast` (Starlink AST)** | FrameSet chaining, compound mappings, cross-frame reprojection, auto-bounding-box planning | `torchsky.wcs.WCS`, `torchsky.wcs.convert_celestial`, `torchsky.map.plan` | `[~] Partial` | Reprojection & 28-projection `plan` shipped; curvilinear Matplotlib grid plotter (`ast.Plot`) not replicated |

---

### 3. `torchsky.map` vs. Mapmaking & Reprojection Software

| Peer Software | Peer Capability & Options | `torchsky` Equivalent | Status | Notes / Remaining Delta |
| :--- | :--- | :--- | :---: | :--- |
| **`AstroDrizzle` (`drizzlepac`)** | `pixfrac` ($0 < p \le 1$), kernels (`square`, `point`/`turbo`, `gaussian`, `lanczos2`, `lanczos3`), `wt_scl` (`exptime`, `expsq`), DQ bitmask filtering (`bits`), `blot` back-projection, derivative cosmic-ray rejection (`driz_cr_snr`, `driz_cr_scale`, `driz_cr_grow`), `context` bitplane cube (`ctx`) | `ResampleConfig`, `Observation(good_bits=...)`, `blot`, `detect_outliers`, `OutlierConfig`, `MapResult.context` | `[x] Shipped` | Full 2D drizzle + blot + outlier + context bitplane parity |
| **`stcal` (JWST & Roman)** | `stcal.resample` (`var_poisson`, `var_rnoise`, `var_flat` separate variance propagation; `gwcs`), `stcal.skymatch` (`local`, `global`, `match`, `global+match` overlap solver), `stcal.outlier_detection`, 3D IFU `cube_build` (drizzle & `EMSM`) | `Observation(var_poisson=..., var_rnoise=..., var_flat=..., ifu_wcs=..., wavelength_map=...)`, `MapResult`, `match_sky`, `detect_outliers`, `coadd_cube` | `[x] Shipped` | Full 2D `stcal` + 3D IFU `cube_build` (`drizzle` & `emsm`) parity |
| **`SWarp` (+ `SCAMP`)** | Resampling kernels (`NEAREST`, `BILINEAR`, `LANCZOS2`, `LANCZOS3`, `LANCZOS4`), combine types (`MEDIAN`, `AVERAGE`, `MIN`, `MAX`, `WEIGHTED`, `CHI2`, `CLIPPED`, `SUM`), `FSCALASTRO_TYPE` (`NONE`, `FIXED`, `VARIABLE` Jacobian conservation), `subtract_back` bicubic mesh (`BACK_SIZE`, `BACK_FILTERSIZE`), `.head` sidecars, all 28 WCS projections | `ResampleConfig`, `StackConfig`, `SkyConfig(mode="mesh")`, `Observation(use_sidecar_head=True)`, `plan(projection=...)` | `[x] Shipped` | Full SWarp parity + bounded-memory tiled mosaic execution (`tile_size`) |
| **`Montage`** | Exact spherical quadrilateral pixel-overlap area (`mProject`/`mProjectPP`), pairwise overlap constant & planar gradient solver (`mDiffExec` + `mFitplane` + `mBgModel`), cross-frame reprojection (`ICRS` $\leftrightarrow$ `Galactic` $\leftrightarrow$ `Ecliptic`) | `ResampleConfig(kernel="exact_spherical")`, `SkyConfig(mode="overlap"\|"overlap_planar")`, `match_sky` | `[x] Shipped` | Full Montage parity |
| **`CORNISH` / `reproject_exact` / `drizzlib`** | Exact Sutherland–Hodgman convex polygon clipping (`kernel="exact"`) & exact spherical arc-polygon area (`kernel="exact_spherical"`) with exact flux/surface-brightness conservation | `ResampleConfig(kernel="exact" \| "exact_spherical")` | `[x] Shipped` | Includes fused C++ CPU fast path for `exact` |
| **Native `torchsky.map` Extensions** | Exact pixel-to-pixel covariance propagation into arbitrary apertures (`aperture_uncertainty`), joint multi-band/cube coaddition (`coadd_multiband`, `coadd_cube`), direct-to-HEALPix coaddition & cutout resampling (`coadd_healpix`, `resample_healpix`), differentiable observation operators ($P, P^\top$), continuous neural/basis sky fields (`torchsky.map.field`) | `aperture_uncertainty`, `coadd_multiband`, `coadd_cube`, `coadd_healpix`, `resample_healpix`, `ObservationOperatorBundle`, `torchsky.map.field` | `[x] Shipped` | Unique to `torchsky` |

---

## Part II — Multi-Wavelength Ecosystem Parity (X-Ray, $\gamma$-Ray, CMB, Radio / mm)

> Tracks both **already-shipped** `torchsky` foundations and **planned** capabilities across high-energy, CMB/sub-mm, and radio astronomy (excluding FITS/MS/HDF5 file I/O, which belongs to `torchfits` or format adapters).

### 1. X-Ray & $\gamma$-Ray Astronomy (`CIAO`, `SAS`, `HEASoft`, `eSASS`, `Fermitools`, `Gammapy`, `XSPEC`, `Sherpa`, `3ML`)

| Peer Software | Peer Capability | `torchsky` Equivalent | Status | Notes / Remaining Delta |
| :--- | :--- | :--- | :---: | :--- |
| **`wcslib` / `HEASoft` / `Gammapy`** | High-energy spectral axes (`ENER`, `WAVN`, `FREQ`, `-LOG`, `-TAB` energy edges) & tangent/AIT/CAR/HPX sky projections | `torchsky.wcs.WCS`, `torchsky.wcs.spectral`, `TabularLookupWCS` | `[x] Shipped` | Full `ENER`/`FREQ`/`-LOG`/`-TAB` support on CPU/CUDA/MPS |
| **`mocpy` / `Fermitools` / `Gammapy`** | Joint Space-Time Good Time Intervals (`GTI` $\times$ sky region / FoV mask) | `torchsky.sphere.STMOC`, `torchsky.sphere.MOC`, `torchsky.sphere.BMOC` | `[x] Shipped` | Native 61-bit µs time intervals $\times$ NUNIQ spatial cells (`STMOC.from_times_and_positions`, `from_time_ranges_and_sky_moc`) |
| **`CIAO` `dmcopy` / `eSASS` `evtool` / `Fermitools` `gtbin` / `Gammapy`** | Event-list photon binning & sub-pixel splatting $(\alpha_i, \delta_i, E_i, t_i) \to$ 2D WCS image, 3D $(x, y, E)$ cube, or `HEALPix` / `HealSparseMap` | `torchsky.map.splat` (`splat_bilinear`, `splat_to_native`), `torchsky.sphere.ang2pix` | `[~] Partial` | 2D differentiable bilinear/native splatting shipped; need dedicated `bin_events` / `bin_events_healpix` with `GTI` + energy-bin edges + deadtime weighting |
| **`CIAO` `mkexpmap` / `Fermitools` `gtexpcube2` / `Gammapy`** | Attitude-convolved exposure maps & off-axis vignetting ($\text{ARF}(E) \times V(E, \theta, \phi) \times \Delta t_{\text{live}}$) | `torchsky.map.operators` (`ObservationOperatorBundle.exposure_map`) | `[~] Partial` | Static `exposure_map` operator slot shipped; need pointing-history $(t, \text{RA}_{\text{pnt}}, \text{Dec}_{\text{pnt}}, \text{roll}) \ast V(E, \theta, \phi)$ integrator |
| **`XSPEC` / `Sherpa` / `Gammapy` / `3ML`** | Non-diagonal Energy Dispersion Matrix (`RMF` / `EDISP` $R(E_{\text{reco}} \mid E_{\text{true}})$) + off-axis King / double-King PSF forward folding | `torchsky.map.operators`, `torchsky.map.field` | `[ ] Planned` | Need sparse/batched `EnergyDispersionOperator` and radially varying King-profile PSF kernel |
| **`Sherpa` / `XSPEC` / `Gammapy`** | Low-count Poisson likelihoods: Cash statistic ($C = 2 \sum (M_i - D_i \ln M_i)$) and $W$-stat (profiled Poisson background) | `torchsky.map.field` | `[ ] Planned` | Currently Gaussian/Charbonnier/Student-t; need exact `cash_loss` and `wstat_loss` |
| **`CIAO` `wavdetect` / `csmooth` / `Cappellari` Voronoi** | Adaptive Poisson Voronoi tessellation & multiscale à-trous wavelet source/diffuse reconstruction | `torchsky.map`, `torchsky.catalogs` | `[ ] Planned` | Planned alongside innovative multiscale background & segmentation phase |

---

### 2. CMB & Millimeter Astronomy (`healpy`, `ducc0`, `pixell`/`enmap`, `TOAST`, `sotodlib`, `Madam`, `NaMaster`, `lenspyx`, `fgbuster`)

| Peer Software | Peer Capability | `torchsky` Equivalent | Status | Notes / Remaining Delta |
| :--- | :--- | :--- | :---: | :--- |
| **`healpy` / `ducc0.sht`** | Spin-0 & Spin-2 $(T, Q, U \leftrightarrow a_{\ell m}^T, a_{\ell m}^E, a_{\ell m}^B)$ Spherical Harmonic Transforms (`map2alm`, `alm2map`, `map2alm_spin`, `alm2map_spin`, `alm2map_der1`, `map2alm_lsq`, `anafast`, `synfast`, `synalm`, `smoothalm`, `pixwin`, `gaussian_beam`, `rotate_alm`) | `torchsky.sphere.spectral`, `torchsky.sphere.harmonics`, `torchsky.sphere.compat` | `[x] Shipped` | Fused Ishioka/Wigner C++ SIMD on CPU + ATen ring FFT on CUDA/MPS; RING & pixel weights |
| **`healpy` / `pixell`** | Direct harmonic-to-WCS tangent/image evaluation & adjoint accumulation ($a_{\ell m} \leftrightarrow$ 2D WCS cutout, spin-0 & spin-2) | `torchsky.map.healpix_spectral` (`sample_alm_on_image_header_spectral`, `sample_spin_alm_on_image_header_spectral`, `accumulate_image_on_alm_adjoint`, `accumulate_image_on_spin_alm_adjoint`) | `[x] Shipped` | Exact direct & adjoint evaluation on arbitrary WCS headers |
| **`Wiener` / Bayesian CMB Solvers** | Harmonic $C_\ell$ and coupled $(T, E, B)$ covariance/precision priors + matrix-free PCG map-space & harmonic-space solvers + Hutchinson posterior variance | `torchsky.map.solvers` (`build_harmonic_cl_prior_operator`, `build_teb_cls_prior_operator`, `build_spherical_spin_harmonic_solver_hooks`, `SurveyInstrumentComponentMixer`) | `[x] Shipped` | Full spin-0 and spin-2 $(T, E, B)$ prior + multi-instrument component-mixing PCG solver substrate |
| **`pixell` (`enmap`) / `ducc0.sht`** | Rectangular `CAR` (Clenshaw–Curtis / Fejer1 equidistant cylindrical) native fast SHT (ACT, Simons Observatory, SPT) | `torchsky.wcs.WCS` (`CAR`), `torchsky.sphere.spectral` | `[~] Partial` | `CAR` WCS projection & `coadd` shipped; fast 2D FFT+Legendre SHT directly on `CAR` rasters without intermediate HEALPix `[ ] Planned` |
| **`pixell.reproject` / `TOAST`** | Spin-2 $(Q, U)$ local meridian polarization angle rotation ($\cos 2\psi, \sin 2\psi$, IAU vs COSMO sign convention) during WCS reprojection & frame transforms (`ICRS` $\leftrightarrow$ `Galactic`) | `torchsky.wcs`, `torchsky.map` | `[ ] Planned` | Scalar reprojection & spin-2 harmonic sampling shipped; need explicit 2×2 Mueller/parallactic rotation in `resample`/`coadd` & `convert_celestial` |
| **`TOAST` / `sotodlib` / `Madam` / `Sanepic` / `Scanamorphos`** | Time-Ordered Data (TOD) pointing matrix $P_{tp}$ (with half-wave plate $4\omega t$ modulation), correlated $1/f$ noise filter $N^{-1}$, and destriping/PCG mapmaker | `torchsky.map.solvers`, `torchsky.map.operators` | `[ ] Planned` | PCG normal-equation solver exists; need streaming TOD pointing + baseline/Fourier $1/f$ noise operator |
| **`NaMaster` (`pymaster`) / `PolSpice`** | Cut-sky pseudo-$C_\ell$ mode-coupling matrix ($M_{\ell \ell'}$ via Wigner-$3j$) and pure-$E/B$ Smith-Zaldarriaga estimator | `torchsky.sphere.spectral` | `[ ] Planned` | Full-sky `anafast` shipped; need mask mode-coupling matrix & pure-$E/B$ |
| **`lenspyx` / `symlens`** | Spherical geodesic lensing deflection ($\tilde{T}(\hat{n}) = T(\hat{n} + \nabla \phi)$) & quadratic estimator | `torchsky.sphere` | `[ ] Planned` | Need tangent-plane geodesic displacement interpolator on the sphere |

---

### 3. Radio & Sub-Millimeter Astronomy (`CASA`, `WSClean`, `ducc0.wgridder`, `cygrid`, `GILDAS`, `spectral-cube`, `radio-beam`, `RM-Tools`)

| Peer Software | Peer Capability | `torchsky` Equivalent | Status | Notes / Remaining Delta |
| :--- | :--- | :--- | :---: | :--- |
| **`wcslib` / `CASA` / `spectral-cube`** | Radio spectral & velocity WCS (`FREQ`, `VRAD`, `VOPT`, `ZOPT`, `FELO`, `BETA`, `RESTFRQ`/`RESTWAV`, air↔vacuum) + `SIN`/`NCP` (WSRT/LOFAR East-West array) projections | `torchsky.wcs.WCS`, `torchsky.wcs.spectral` | `[x] Shipped` | Exact `NCP` (slant orthographic) and radio velocity/frequency conversions |
| **`astropy.coordinates` / `CASA`** | Kinematic Local Standard of Rest (`LSRK` / `LSRD`) line-of-sight velocity projection & 3D spectral cube coaddition | `torchsky.wcs.frames` (`heliocentric_rv_to_lsr`, `lsr_line_of_sight_kms`), `torchsky.map.coadd_cube` | `[x] Shipped` | LSR velocity projection & 3D cube stacking shipped |
| **`CASA` `linmos` / `WSClean`** | Primary-beam attenuation $A(\theta, \nu)$ and primary-beam-weighted linear mosaicking ($\sum A_i w_i I_i / \sum A_i^2 w_i$) | `torchsky.map.operators` (`ObservationOperatorBundle.primary_beam_map`) | `[~] Partial` | 2D `primary_beam_map` operator slot & exact adjoint shipped; need analytic/parametric frequency-dependent 3D primary beam models (`AiryDisk`, `GaussianBeam`, `Cos3Beam`) wired into `coadd_cube` |
| **`CASA` / `GILDAS`** | Time-dependent Earth-orbital Doppler tracking (`TOPO` $\to$ `LSRK` / `BARY` per-exposure frequency shift) during `coadd_cube` | `torchsky.wcs.spectral`, `torchsky.map.coadd_cube` | `[ ] Planned` | Need automatic per-exposure `mjd_obs` + `Site` $\to$ spectral axis shift inside `coadd_cube` |
| **`cygrid` / `CASA` `tsdimaging` / `GILDAS`** | Single-dish On-The-Fly (OTF) irregular dump gridding onto 3D $(x, y, \nu)$ cubes with Gauss–Bessel / prolate-spheroidal kernels | `torchsky.map` | `[ ] Planned` | `coadd_cube` IFU mode handles 2D detector frames; need 1D/irregular OTF spectra dump gridding |
| **`WSClean` / `ducc0.wgridder` / `CASA` `tclean` / `finufft`** | Interferometric $(u, v, w)$ visibility gridding & degridding (2D/3D NUFFT + $w$-stacking / $w$-towers + direction-dependent $A$-projection) | `torchsky.map.operators` | `[ ] Planned` | High-priority differentiable visibility operator $V(u,v,w) \leftrightarrow I(l,m,\nu)$ for continuous field & clean/forward modeling |
| **`radio-beam` / `spectral-cube`** | Elliptical Gaussian restoring beam algebra & common-beam cube homogenization ($\Sigma_{\text{kern}}(\nu) = \Sigma_{\text{target}} - \Sigma_{\text{chan}}(\nu)$) | `torchsky.map` (`Observation(beam_major_arcsec=..., beam_minor_arcsec=..., beam_pa_deg=...)`) | `[~] Partial` | Per-observation beam metadata & header serialization (`BMAJ`, `BMIN`, `BPA`) shipped; need analytical common-beam solver & per-channel convolution |
| **`RM-Tools` (`cirada`)** | Broadband Stokes $(Q(\lambda^2), U(\lambda^2))$ Faraday Rotation Measure synthesis ($\tilde{F}(\phi) = \int P(\lambda^2) e^{-2i\phi(\lambda^2 - \lambda_0^2)} d\lambda^2$) & RM-CLEAN | `torchsky.spectral`, `torchsky.map` | `[ ] Planned` | Differentiable 1D/3D Faraday dispersion function transform |


---

## Part III — Drill-Down Execution Plan: Completing the UV-to-Far-IR Stack

The goal of this phase is to make `torchsky` a **self-contained, survey-grade UV-to-Far-IR stack** covering GALEX, HST, JWST, Euclid, Roman, Rubin/LSST, CFHT/Subaru/DECam, Spitzer, WISE, and Herschel—eliminating any remaining need for external `PSFEx`, `photutils`, `SExtractor`/`SEP`, `SCAMP`/`tweakreg`, `jwst.cube_build`, or `Scanamorphos`.

### Pillar 1: `torchsky.wcs` (UV-to-IR Astrometric & IFU WCS Completion) — `[x] Shipped`

| ID | Deliverable | Entry Points | Status |
| :--- | :--- | :--- | :---: |
| **WCS-1** | **WCS Header & `.head` Sidecar Serialization (`to_header`, `to_head_string`)** | `WCS.to_header`, `WCS.to_head_string` (`src/torchsky/wcs/core.py`) | `[x] Shipped` |
| **WCS-2** | **First-Class WCS Geometry Helpers (`pixel_scales`, `pixel_area`, `position_angle`, `footprint`, `pixel_area_map`)** | `WCS.pixel_scales`, `WCS.pixel_area`, `WCS.position_angle`, `WCS.footprint`, `WCS.pixel_area_map` (`src/torchsky/wcs/core.py`) | `[x] Shipped` |
| **WCS-3** | **3D IFU & Slitless Disperse-and-Project WCS (`IFUSliceWCS`)** | `IFUSliceWCS.from_slices`, `IFUSliceWCS.from_callable` (`src/torchsky/wcs/gwcs_sampling.py`) | `[x] Shipped` |
| **WCS-4** | **Differential Chromatic Refraction (DCR) & Moving-Target Ephemeris WCS Offsets** | `dcr_offset`, `apply_dcr`, `apply_ephemeris_offset` (`src/torchsky/wcs/iau2006.py`) | `[x] Shipped` |

---

### Pillar 2: `torchsky.sphere` (UV-to-IR Survey Footprint & Multi-Order Operations) — `[x] Shipped`

| ID | Deliverable | Entry Points | Status |
| :--- | :--- | :--- | :---: |
| **SPH-1** | **`BMOC` & WCS-to-MOC / HealSparse Footprint Compiler** | `BMOC`, `MOC.from_wcs`, `MOC.from_observations`, `BMOC.from_wcs` (`src/torchsky/sphere/moc.py`) | `[x] Shipped` |
| **SPH-2** | **Rejection-Free Spherical Randoms & Survey Window Sampling (`sample_randoms`)** | `sample_randoms`, `MOC.sample_randoms`, `BMOC.sample_randoms`, `HealSparseMap.sample_randoms` (`src/torchsky/sphere/moc.py`) | `[x] Shipped` |
| **SPH-3** | **Multi-Column Survey Property `HealSparseMap` (`from_columns`, `from_wcs`, `from_observations`)** | `HealSparseMap.from_columns`, `HealSparseMap.from_wcs`, `HealSparseMap.from_observations`, `hsp["col"]` (`src/torchsky/sphere/sparse.py`) | `[x] Shipped` |

---

### Pillar 3: `torchsky.map` (HEALPix Coaddition, 3D IFU Drizzle/EMSM, & Tiled Mosaicking) — `[x] Shipped`

| ID | Deliverable | Entry Points | Status |
| :--- | :--- | :--- | :---: |
| **MAP-1** | **Direct-to-HEALPix Coaddition & Cutout Resampling (`coadd_healpix`, `resample_healpix`)** | `coadd_healpix`, `resample_healpix`, `HealpixResult` (`src/torchsky/map/image_plane.py`) | `[x] Shipped` |
| **MAP-2** | **3D Spatial-Spectral Voxel Drizzle & EMSM (`coadd_cube` IFU mode) + DCR/Moving-Target Coadds** | `coadd_cube`, `ResampleConfig(spectral_kernel="drizzle"\|"emsm")`, `Observation(wavelength_map=..., ifu_wcs=..., tracking_rate=..., dcr_zenith_deg=...)` (`src/torchsky/map/image_plane.py`) | `[x] Shipped` |
| **MAP-3** | **Bounded-Memory Spatial Tile Streaming (`coadd(..., tile_size=...)`)** | `coadd(..., tile_size=...)`, `StackConfig(tile_size=...)` (`src/torchsky/map/image_plane.py`) | `[x] Shipped` |

---

### Pillar 4: Self-Supervised & Calibration-Driven Defect / Weightmask Segmentation (`torchsky.quality` & `torchsky.map`)
*Replacing `MaxiMask`, `WeightWatcher`, `AstroDrizzle` CR-reject, and manual polygon masks.*

| ID | Deliverable | Irreplaceable Physics / Math | What Is Learned from Data / Calibration | Entry Points & Oracle |
| :--- | :--- | :--- | :--- | :--- |
| **CLN-1** | **Calibration-Stack Detector Priors (`learn_detector_priors`)** | Detector-coordinate invariance $(u, v)$ of static/telegraph defects, optical ghosts, and exponential charge-trap persistence decay $\sum_k a_k e^{-\Delta t / \tau_k}$ across exposures | Empirical Bayes multi-exposure pixel-history mixture (finding dead, hot, non-linear, and H2RG/H4RG RTS/telegraph pixels) + low-rank NMF wisp/claw/ghost templates + persistence decay amplitudes | `learn_detector_priors` (`torchsky.map`); validated on multi-epoch darks/flats/science stacks |
| **CLN-2** | **Multi-Exposure Counterfactual & Morphological Defect Decomposition (`clean`)** | Dither counterfactual invariance (celestial sky co-moves with $\mathcal{G}_n(\alpha, \delta)$; cosmic rays, snowballs, and streaks do not), Laplacian sharp-edge bounds, and 2D Radon line-integral geometry | Leave-one-out blot innovation $z_n = (y_n - \hat{y}_{n \mid -n}) / \sigma_n$ + differentiable multi-scale Radon streak detector + bright-star diffraction/saturation ray geometry + optional in-situ self-supervised head outputting calibrated probabilities into `WeightProduct` | `clean` (`torchsky.map`), `WeightProduct` (`torchsky.quality`); outperforms `MaxiMask` & `AstroDrizzle` `detect_outliers` on CRs, NIRCam snowballs, and $S/N \sim 1.5$ satellite trails |

---

### Pillar 5: Hybrid Wavefront-Physics + Neural Field PSF & Cross-Wavelength Homogenization (`torchsky.map.psf`)
*Replacing `PSFEx`, `photutils.psf.EPSFBuilder`, `WebbPSF` empirical tweaks, `Alard–Lupton`, and `T-PHOT` kernels.*

| ID | Deliverable | Irreplaceable Physics / Math | What Is Learned from Data / Calibration | Entry Points & Oracle |
| :--- | :--- | :--- | :--- | :--- |
| **PSF-1** | **Continuous Spatio-Spectral-Flux Effective PSF (`estimate_psf`)** | Exact positivity ($P \ge 0$), unit flux ($\iint P = 1$), zero first-moment centroid gauge ($\iint \boldsymbol{u} P = \boldsymbol{0}$), sub-pixel phase shift across dithers, and broadband SED integral $\int T(\lambda)\lambda F_\lambda P(\lambda) d\lambda$ | Continuous coordinate-conditioned field $P_\psi(\Delta u, \Delta v \mid x_{\text{det}}, y_{\text{det}}, \lambda, t_n, \text{flux})$ combining an analytical Moffat/Airy/wavefront-Zernike core with `LowRankChromaticPSF` + flux-dependent brighter-fatter `GatedResidualOperator` fit jointly across dithered star stamps | `estimate_psf` (`torchsky.map`); sub-pixel astrometric/photometric fidelity vs `PSFEx` / `EPSFBuilder` on undersampled HST/JWST/Euclid frames |
| **PSF-2** | **Regularized Positive-Definite PSF Homogenization (`match_psf`) & Native Forward Fusion** | Exact flux conservation, zero dipole shift, and exact correlated variance propagation $W_{\text{out}}$ | Multi-scale Gaussian-Laguerre / shapelet basis solver + SURE/GCV auto-regularized Fourier Wiener kernel $K * P_{\text{src}} = P_{\text{target}}$, plus native zero-degradation forward mode in `torchsky.field` | `match_psf`, `coadd_multiband(..., target_psf=...)`; $\le 1\%$ encircled-energy match across UV ($0.03''$) to Far-IR ($6''\text{–}35''$) |

---

### Pillar 6: Physics-Disentangled Multi-Exposure Background & $1/f$ Destriping (`torchsky.map.sky`)
*Replacing `SWarp` `BACK_SIZE`, `photutils.Background2D`, `stcal.skymatch`, `jwst.clean_flicker_noise`, and `Scanamorphos`.*

| ID | Deliverable | Irreplaceable Physics / Math | What Is Learned from Data / Calibration | Entry Points & Oracle |
| :--- | :--- | :--- | :--- | :--- |
| **BKG-1** | **Joint Multi-Exposure Sky + Cirrus/LSB + $1/f$ Decomposition (`match_sky` / `estimate_sky`)** | Symmetry split: Galactic cirrus & Intracluster Light (ICL) are stationary in sky coordinates $(\alpha, \delta)$ across dithers/rolls, whereas thermal drift is smooth in $(u, v, t_n)$ and H2RG/H4RG $1/f$ noise is rank-1 along readout rows/columns/amplifiers | Matrix-free PCG / alternating solver decomposing overlapping exposures into stationary sky $I_{\text{LSB}}(\alpha, \delta)$ + per-exposure smooth `ConditionedBackgroundField` $B_n(u, v, t_n)$ + masked Huber row/column/amplifier $1/f$ destriper (`destripe=True`) | `estimate_sky`, `match_sky`; **0% attenuation** of injected extended LSB halos / Galactic cirrus + complete removal of JWST NIRCam/MIRI $1/f$ striping |

---

### Pillar 7: Global Differentiable Astrometric Bundle & Photometric Solver (`torchsky.wcs` & `torchsky.catalogs`)
*Replacing `SCAMP`, `drizzlepac.tweakreg`, and `jwst.tweakreg`.*

| ID | Deliverable | Irreplaceable Physics / Math | What Is Learned from Data / Calibration | Entry Points & Oracle |
| :--- | :--- | :--- | :--- | :--- |
| **AST-1** | **Two-Stage Global Astrometric Bundle & Direct Pixel Solver (`solve_astrometry`)** | Exact gnomonic tangent-plane chart, shared camera focal-plane distortion (`TPV`/`SIP`) across exposures vs per-exposure pointing/atmospheric shear, 6D Gaia DR3 epoch propagation $(\boldsymbol{\mu}, \varpi)$, and chromatic DCR | **Stage A:** Sparse GPU Gauss–Newton/Cholesky solver jointly fitting per-CCD `CRVAL`/`CD`, shared focal-plane `TPV`/`SIP` (+ smooth residual field), and source proper motions $\boldsymbol{\mu}_i$ across epochs. **Stage B:** Direct differentiable image-gradient alignment (`ExposurePose`) for sparse/deep/IFU fields | `solve_astrometry` (`torchsky.wcs`); sub-milliarcsecond internal RMS and Gaia DR3 tie vs `SCAMP` & `tweakreg` |

---

### Pillar 8: Spatio-Spectral Source Detection, Deblending & Forward Cataloguing (`torchsky.catalogs` & `torchsky.scene`)
*Replacing `SExtractor`/`SEP`, `photutils`, `SCARLET`, `Tractor`, and `T-PHOT`.*

| ID | Deliverable | Irreplaceable Physics / Math | What Is Learned from Data / Calibration | Entry Points & Oracle |
| :--- | :--- | :--- | :--- | :--- |
| **SEG-1** | **Multi-Band SED-Matched & $\chi^2$ Source Detection (`detect_sources`)** | Exact GLS matched-filter sufficient statistics $P_b^\top W_b y_b$ and Szalay multi-band $\chi^2$ under correlated noise | Optimal SED-dictionary matched filtering + GPU connected-component footprint labeling and multi-scale saddle-point peak finding | `detect_sources` (`torchsky.catalogs`); completeness/purity parity & gain vs `SEP` |
| **CAT-1** | **Vectorized Aperture & Cross-Wavelength Forward/Deblended Photometry (`measure_catalog`)** | Exact sub-pixel aperture geometry, Kron/Petrosian moments, `aperture_uncertainty` correlated variance, and linear flux Cramer–Rao inverse Hessian $\Sigma_{\text{flux}}$ | **Fast mode:** GPU aperture/Kron/moment extraction. **Forward/deblend mode (`T-PHOT`/`SCARLET` replacement):** factorized non-negative morphology $\times$ spectrum (or `SpectralPowerFoam` components) projected through each band's native `WCS` + `PSFModel` to deblend low-res Mid/Far-IR from high-res UV/Opt/NIR | `measure_catalog`, `deblend_sources` (`torchsky.catalogs`) |
| **CAT-2** | **Bayesian Multi-Resolution & Astrometric Catalog Association (`associate`)** | Exact 5D astrometric covariance propagation ($\chi^2_{\text{Mahalanobis}}$) + Sutherland–Saunders Poisson spatial likelihood ratio | Learned empirical magnitude/color/morphology prior density $q(m, c)$ self-calibrated from the field vs background density $n(m, c)$ | `associate` (`torchsky.catalogs`) |

---

### Pillar 9: Remaining UV-to-IR Forward Physics (Slitless Grism `WFSS` & Mid-IR/IGM Extinction)
*Replacing `grizli`/`hstaxe` core trace projection and completing UV-to-Mid-IR dust/IGM laws.*

| ID | Deliverable | Irreplaceable Physics / Math | What Is Learned from Data / Calibration | Entry Points & Oracle |
| :--- | :--- | :--- | :--- | :--- |
| **WFS-1** | **Differentiable Wide-Field Slitless Spectroscopy (`WFSS` / Grism) Operator (`GrismWCS`, `GrismDispersionOp`)** | Multi-order ($m=-1, 0, +1, +2$) trace polynomials $(\Delta u_m(\lambda; u_0, v_0), \Delta v_m(\lambda; u_0, v_0))$ and sensitivity $S_m(\lambda)$ with exact linear adjoint $\mathcal{D}^\top$ | Joint multi-roll-angle (`PA_1`, `PA_2`) + direct-image trace decontamination and 1D/3D spectral extraction via PCG or `torchsky.field` | `GrismWCS` (`torchsky.wcs`), `GrismDispersionOp` (`torchsky.map.operators`); HST WFC3 + JWST NIRISS/NIRCam + Euclid/Roman WFSS |
| **EXT-1** | **Gordon+23 ($0.09\text{–}32\,\mu\text{m}$) Silicate Dust & Inoue+14 IGM $\text{Ly}\alpha$ Attenuation** | Drude profile $9.7\,\mu\text{m}$ & $18\,\mu\text{m}$ silicate absorption (`g23`) and redshift-dependent Lyman-series + Lyman-continuum optical depth $\tau_{\text{IGM}}(\lambda_{\text{obs}}, z)$ | Differentiable $R_V$ and $E(B-V)$ / IGM damping-wing joint inference inside `torchsky.spectrophotometry` and `torchsky.field` | `extinction_curve(..., law="g23")`, `igm_transmission` (`torchsky.extinction`) |

---

### Recommended Execution Sequence

1. **Phase A — Defect & Weightmask Segmentation (`CLN-1`, `CLN-2`) + $1/f$ & LSB-Safe Sky (`BKG-1`)**: establishes clean, uncorrupted per-exposure weights (`WeightProduct`), removes H2RG/H4RG $1/f$ stripes, and protects extended LSB/cirrus before any source or PSF fitting.
2. **Phase B — Multi-Band Source Detection (`SEG-1`) + Global Astrometric Solver (`AST-1`)**: detects high-$S/N$ anchors and solves the global multi-exposure focal-plane (`TPV`/`SIP`) + Gaia DR3 proper-motion bundle (`solve_astrometry`).
3. **Phase C — Continuous Spatio-Spectral-Flux PSF & Homogenization (`PSF-1`, `PSF-2`)**: learns the sub-pixel dither-aware `PSFModel` (with chromaticity and brighter-fatter residuals) on astrometrically locked exposures and enables `match_psf` + PSF-matched `coadd_multiband`.
4. **Phase D — Deblended & Cross-Wavelength Forward Cataloguing (`CAT-1`, `CAT-2`) + Grism/Extinction Physics (`WFS-1`, `EXT-1`)**: completes vectorized aperture/Kron/forward photometry (`T-PHOT`/`SCARLET`/`Tractor` replacement), Bayesian catalog association, and slitless grism + Mid-IR/IGM forward physics.

---

## Part IV — Clean API Surface Summary (`[x] Shipped`)

| Module | Canonical Verbs & Nouns | Notes |
| :--- | :--- | :--- |
| **`torchsky.wcs`** | `WCS`, `IFUSliceWCS`, `wcs.pix2world`, `wcs.world2pix`, `wcs.jacobian`, `wcs.with_affine`, `wcs.to_header`, `wcs.to_head_string`, `wcs.pixel_scales`, `wcs.pixel_area`, `wcs.position_angle`, `wcs.footprint`, `wcs.pixel_area_map`, `from_tabular`, `from_coupled_tabular`, `sample_gwcs`, `sample_gwcs_epochs`, `fit_sky_affine`, `dcr_offset`, `apply_dcr`, `apply_ephemeris_offset` | Short verb methods on `WCS` (`pix2world`, `world2pix`, `jacobian`, `with_affine`) alongside functional API. |
| **`torchsky.sphere`** | `Polygon`, `MultiPolygon`, `Circle`, `Cone`, `Cap`, `Box`, `Ellipse`, `MOC`, `BMOC`, `STMOC`, `HealSparseMap`, `query_cone`, `query_circle`, `query_box`, `query_polygon`, `query_ellipse`, `query_strip`, `sample_map`, `sample_randoms`, `ang2pix`, `pix2ang`, `ang2vec`, `vec2ang`, `vec2pix`, `pix2vec`, `ring2nest`, `nest2ring`, `reorder`, `ud_grade`, `neighbors`, `boundaries`, `get_interp_weights`, `get_interp_val`, `alm2map`, `map2alm`, `anafast`, `synfast`, `smoothing` | Short region nouns (`Polygon`, `Cone`, `Cap`, `Circle`, `Box`, `Ellipse`), unified `sample_map(values, lon_deg, lat_deg)`, and top-level HEALPix core exports. |
| **`torchsky.map`** | `Observation`, `TargetGrid`, `CubeTarget`, `MapResult`, `CubeResult`, `MultibandResult`, `HealpixResult`, `ProductSpec`, `BandpassSpec`, `ProductResult`, `UncertaintyProduct`, `ApertureUncertainty`, `plan`, `resample`, `coadd`, `coadd_multiband`, `plan_cube`, `coadd_cube`, `resample_healpix`, `coadd_healpix`, `estimate_sky`, `match_sky`, `aperture_uncertainty`, `plan_product`, `stack_product`, `evaluate_product` | Concise `Verb` + `NounResult` pattern across 2D, 3D cube, multiband, HEALPix, and survey products. |
| **`torchsky.catalogs`** | `Catalog`, `Catalog.state`, `CatalogState`, `CrossmatchResult`, `crossmatch`, `crossmatch_sky`, `cone_search`, `SpatialIndex`, `HealpixIndex` | Unified `crossmatch(left, right, ...)` handling `Catalog`, `(ra, dec)` tensors, and Arrow/streamed tables. |
| **`torchsky.coords`** | `SkyCoord`, `Frame`, `ObserverLocation`, `transform_coords`, `angular_separation`, `position_angle`, `offset_by`, `dcr_offset`, `apply_dcr`, `apply_ephemeris_offset`, `apply_refraction`, `apply_stellar_aberration` | Chromatic refraction, aberration, and moving-target ephemeris offsets available directly in `torchsky.coords`. |
| **`torchsky.scene`** | `coadd`, `coadd_cube`, `coadd_healpix`, `coadd_multiband`, `plan`, `plan_cube`, `resample`, `resample_healpix`, `fit`, `model`, `stack_product` | Unified high-level scene composition facade mirroring `torchsky.map` and `torchsky.field`. |

