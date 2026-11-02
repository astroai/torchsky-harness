# Subject API gaps

Harness code that still imports torchsky names outside `docs/api-manifest.json` in the subject
(`torchsky` 0.4.x). Every name that has a public equivalent with the *identical object* was
already ported (for example `torchsky.field.fit` for `fit_field_model`, `torchsky.scene` for
`FieldObservation`). The names below have **no public equivalent**; the harness keeps importing
them from their implementation modules until torchsky either exports them or the evaluation is
retired. Semantics were not changed.

155 names in 33 modules.

- `torchsky`: `_cpp`
- `torchsky._native`: `cpp`
- `torchsky.calibration_evidence`: `calibration_exposure_id`, `load_operator_calibration_split`
- `torchsky.device`: `default_compute_device`
- `torchsky.domain`: `DomainSpec`
- `torchsky.instrument`: `NIRCAM_METADATA_SOURCES`, `NIRCAM_UPDEC_BANDS`, `jwst_exposure_timing`, `nircam_imaging_band`
- `torchsky.map`: `calibration`, `image_plane_coadd`, `image_plane_projection`, `windowing`
- `torchsky.map.field`: `IRLSConfig`, `LearnedSpectralBasisField`, `LinearBandGridField`, `SineImplicitField`, `SkyEncoderBandpassField`, `SkyEncoderMLPField`, `SkyEncoderResMLPField`, `SpectralBasisField`, `SpectralPowerFoam`, `SpectralPowerFoamField`, `adapt_stationary_calibrated_band_observation`, `batched_bounded_power_pixel_operator`, `bind_linear_band_grid_observations_to_operator_set`, `bounded_power_ownership`, `bounded_power_pixel_operator`, `cech_adjacency`, `field_quality_scorecard`, `prepare_linear_band_grid_observation_data`, `pretrain_strainer_trunk_synthetic`, `render_field_cube`, `render_field_observation`, `render_field_observation_distribution`, `render_power_spectral_samples`, `render_spectral_power_foam_observation_exact`, `save_strainer_trunk_checkpoint`, `solve_joint_multiband_grid_cg`, `solve_linear_band_grid_cg`, `solve_linear_band_grid_lsqr`, `solve_power_pixel_affine_least_squares`, `solve_power_pixel_least_squares`, `solve_spectral_cell_coefficients`, `solve_spectral_power_pixel_affine_least_squares`, `sparse_bounded_power_pixel_operator`, `sparse_cech_graph`, `spectral_basis_whitening_transform`, `tile_disk_candidate_csr`
- `torchsky.map.field.bandpass`: `bandpass_quadrature`
- `torchsky.map.field.core`: `ExplicitSpectralGridField`
- `torchsky.map.field.forwarding`: `type_method_accepts_kwarg`
- `torchsky.map.field.frequency_adaptive`: `BaconField`, `SASNetField`
- `torchsky.map.field.gauss_newton`: `gauss_newton_pcg_step`
- `torchsky.map.field.hashgrid`: `sample_feature_plane`
- `torchsky.map.field.observation_ops`: `crop_field_observation`, `observation_support_mask`
- `torchsky.map.field.rendering`: `iter_field_observation_tiles`
- `torchsky.map.field.training`: `_compute_full_observation_losses`, `compute_field_fit_diagnostics`
- `torchsky.map.operators`: `AdaptiveSpatioSpectralTemporalEncoder`, `BackgroundCalibrationTerm`, `ConditionedBackgroundField`, `ConditionedKernelPSF`, `ExtendedResidualKernelPSF`, `IndexedResponsePlan`, `LearnedBackgroundOp`, `LearnedPSFOp`, `LowRankChromaticPSF`, `MapValueSemantics`, `PSFCalibrationTerm`, `SpatioSpectralTemporalEncoder`, `StructuredDetectorNoise`, `StructuredNoiseOp`, `VAR_FLAT`, `VAR_POISSON`, `VAR_RNOISE`, `apply_learned_psf_spec_adjoint`, `apply_operator_calibration_pruning`, `apply_psf_spec`, `apply_response_products`, `bind_image_plane_target_operator`, `field_observation_operator_bundle`, `fit_background_plane`, `fit_operator_calibration_terms`, `headers_equivalent_for_direct_copy`, `make_minimal_learned_operator_scaffold`, `normalized_header_for_shape`, `operator_calibration_identifiability_report`, `operator_calibration_score_report`, `operator_calibration_validation_pruning_report`, `pixel_area_sr_at`, `pixel_area_sr_map`, `profile_amplifier_row_offsets`, `registered_image_stamps`, `spatial_fill_distance_bounds`, `spherical_spectral_operator_bundle`, `swap_bundle_operator_families`, `target_shape_from_header`, `validate_image_plane_2d`
- `torchsky.map.operators.background`: `fit_background_plane`
- `torchsky.map.operators.execution`: `exact_median_scratch_bytes`, `overlap_background_scratch_bytes`, `sigma_clipped_scratch_bytes`, `spilled_exact_median_stack`, `spilled_sigma_clipped_stack`
- `torchsky.map.operators.noise`: `ExplicitNoiseOp`
- `torchsky.map.operators.psf`: `gaussian_kernel1d`, `gaussian_support_radius`, `psf_support_radius`
- `torchsky.map.product`: `estimate_map_inverse_variance`, `render_field_product`, `render_scene_product`
- `torchsky.map.sky`: `CrossExposureSkyCorrector`, `MeshSkyEstimator`, `PolynomialSkyModel`
- `torchsky.map.solvers`: `BoundObservationOperator`, `BoundObservationOperatorSet`, `build_solver_hooks`, `build_spherical_harmonic_solver_hooks`, `build_spherical_spin_harmonic_solver_hooks`, `build_state_kronecker_operator`, `build_state_subspace_operator`, `build_teb_cls_prior_operator`, `compose_bound_operators`, `conjugate_gradient`, `harmonic_cl_diagonal`, `make_diagonal_preconditioner`
- `torchsky.sphere`: `spectral`
- `torchsky.sphere.geom`: `_numpy_sg_contains_lonlat_bulk`, `_region_to_sg`
- `torchsky.sphere.healpix`: `ang2pix_nested`, `ang2pix_ring`, `ang2vec`, `get_all_neighbours`, `get_map_size`, `get_nside`, `isnpixok`, `isnsideok`, `max_pixel_radius`, `nest_children`, `pix2ang_nested`, `pix2ang_ring`, `vec2ang`
- `torchsky.viz.renderers.image`: `zscale_limits`
- `torchsky.wcs.calibration`: `_PhotometrySolver`
- `torchsky.wcs.frames`: `ARCSEC_TO_RAD`
- `torchsky.wcs.iau2006`: `cirs_matrix`, `npb_matrix`
- `torchsky.wcs.temporal`: `SECONDS_PER_DAY`, `TAI_MINUS_GPS_SECONDS`, `TemporalWCS`, `convert_time_scale`

## Update for torchsky e65a08f2 (0.5 API cleanup)

Now public: `torchsky.map.coadd_image_plane`, `torchsky.map.ImageObservation`,
`torchsky.wcs.TemporalWCS`, `torchsky.wcs.DomainSpec`; field rendering through
`torchsky.field.render(..., target=...)`. Made private: `map.field.gauss_newton`
(now `_gauss_newton`, white-box import in the SMACS joint-fit benchmark),
`sphere.chord_radius` (the catalog-scale contract computes the chord directly).
The frozen `campaigns/adass2026/jobs` keep their original imports.
