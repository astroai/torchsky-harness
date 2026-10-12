"""Software checks only; these fixtures provide no scientific evidence."""

import json
import sys
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch








def test_neural_pcg_cli_rejects_a_grid_only_request() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_map_smacs0723_field_fit_joint.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--models",
            "coadd_bilinear",
            "--inr-pcg-refine-steps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires a supported neural model" in result.stderr


def test_native_training_view_preserves_detector_background_weight_and_mask(tmp_path: Path) -> None:
    from dataclasses import replace

    import torchsky
    from torchsky.map import FieldObservation, ForwardContext, ImageObservation, render_field_observation
    from torchsky.map.operators import LearnedBackgroundOp, field_observation_operator_bundle

    from torchsky_harness.benchmarks.bench_map_smacs0723_field_fit_joint import _ZeroScene, _export_native_training_view

    class FixedBackground(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.7), requires_grad=False)

        def forward(self, x: torch.Tensor, y: torch.Tensor, wavelength: torch.Tensor,
                    time: torch.Tensor) -> torch.Tensor:
            return self.bias + 0.01 * x + 0.02 * y

    header = {
        "NAXIS1": 4, "NAXIS2": 3, "BUNIT": "MJy/sr",
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": 110., "CRVAL2": -25., "CRPIX1": 2.2, "CRPIX2": 1.8,
        "CD1_1": -0.0003, "CD1_2": 0.00004, "CD2_1": 0.00006, "CD2_2": 0.0003,
        "SUBSTRT1": 6, "SUBSTRT2": 8,
    }
    err = torch.linspace(0.3, 0.6, 12).reshape(3, 4)
    weight = err.reciprocal().square()
    mask = torch.ones_like(err)
    mask[0, 0] = 0.
    observation = FieldObservation(
        image=torch.zeros_like(err), header=header, err=err, weight=weight, mask=mask,
        wavelength=27700., detector_x0_pix=5., detector_y0_pix=7.,
        contamination_prior=torch.full_like(err, 0.9),
    )
    source_path = tmp_path / "source.fits"
    context = ForwardContext(
        survey="F277W", detector="NRCBLONG", exposure_id="fixed-software-fixture",
        extra={"split": "train", "source_path": str(source_path)},
    )
    background = FixedBackground().eval()
    bundle = field_observation_operator_bundle(observation).with_updates(
        background=LearnedBackgroundOp(background, coordinate_frame="detector")
    )
    settings = {"num_spectral_samples": 1, "spatial_sampling": "center"}
    fixed = render_field_observation(
        _ZeroScene(torch.device("cpu")), observation, forward_context=context,
        operator_bundle=bundle, **settings,
    )
    signal = torch.arange(12, dtype=err.dtype).reshape(3, 4) / 10.
    observation.image.copy_(fixed + signal)
    observation.image[0, 0] = float("nan")
    scale = 3.
    torchsky.write(str(source_path), observation.image * scale, header=header)
    source_bytes = source_path.read_bytes()
    before = {name: getattr(observation, name).clone() for name in ("image", "err", "weight", "mask")}
    report = _export_native_training_view(
        tmp_path / "view", [observation], [context], [bundle],
        intensity_scale=scale, fit_kwargs=settings,
        calibration_metadata={"software_fixture": True},
    )
    saved = json.loads(Path(report["manifest_path"]).read_text())
    descriptor = saved["rows"][0]["image_observation"]
    image_observation = ImageObservation(**descriptor)
    science, exported_header = torchsky.read(image_observation.path, return_header=True)
    saved_weight = torchsky.read(image_observation.weight_path)
    saved_mask = torchsky.read(image_observation.mask_path)
    valid = mask > 0
    torch.testing.assert_close(science[valid], (signal * scale)[valid])
    torch.testing.assert_close(saved_weight[valid], weight[valid] / scale ** 2)
    torch.testing.assert_close(saved_mask, mask)
    assert science[0, 0] == saved_weight[0, 0] == 0
    assert saved["resampled"] is False and saved["rows"][0]["valid_pixels"] == 11
    assert exported_header["CRPIX1"] == header["CRPIX1"]
    assert exported_header["CD1_2"] == header["CD1_2"]
    assert exported_header["SUBSTRT1"] == 6
    assert source_path.read_bytes() == source_bytes
    for name, tensor in before.items():
        torch.testing.assert_close(getattr(observation, name), tensor, equal_nan=True)
    assert background.bias.item() == pytest.approx(0.7)
    with pytest.raises(ValueError, match="TRAIN"):
        _export_native_training_view(
            tmp_path / "heldout", [observation], [replace(context, extra={"split": "validation"})],
            [bundle], intensity_scale=scale, fit_kwargs=settings, calibration_metadata={},
        )
    with pytest.raises(FileExistsError):
        _export_native_training_view(
            tmp_path / "view", [observation], [context], [bundle],
            intensity_scale=scale, fit_kwargs=settings, calibration_metadata={},
        )


@pytest.mark.parametrize(
    "model_kind,calibrated",
    [
        ("sine_implicit", False),
        ("wire_implicit", False),
        ("fourier_mlp", False),
        ("sky_encoder_mlp", False),
        ("sky_encoder_resmlp", False),
        ("hashgrid_mlp", False),
        ("hashgrid_resmlp", False),
        ("sine_implicit", True),
    ],
)
def test_neural_pcg_refines_native_detector_likelihood(
    model_kind: str, calibrated: bool
) -> None:
    from torchsky.map import (
        FieldDomain,
        FieldObservation,
        ForwardContext,
        fit_field_model,
        make_field_model,
        render_field_observation,
    )
    from torchsky.map.operators import field_observation_operator_bundle

    from torchsky_harness.benchmarks.bench_map_smacs0723_field_fit_joint import (
        _refine_neural_scene_pcg,
        _save_joint_field_artifact,
    )

    torch.manual_seed(101)
    domain = FieldDomain(150.0, 2.0, 0.01, 5000.0, 500.0)
    kwargs = {"hidden_dim": 4}
    if model_kind == "hashgrid_resmlp":
        kwargs["num_res_blocks"] = 1
    elif model_kind == "sky_encoder_resmlp":
        kwargs["num_blocks"] = 1
    else:
        kwargs["hidden_layers"] = 1
    if model_kind.startswith("hashgrid"):
        kwargs.update(
            num_levels=2,
            features_per_level=2,
            base_resolution=2,
            finest_resolution=4,
            log2_hashmap_size=4,
        )
    model = make_field_model(model_kind, domain, **kwargs)
    header = {
        "NAXIS": 2,
        "NAXIS1": 5,
        "NAXIS2": 4,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2.2,
        "CRPIX2": 1.8,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CD1_1": -0.0003,
        "CD1_2": 0.00004,
        "CD2_1": 0.00006,
        "CD2_2": 0.0003,
    }
    if calibrated:
        header.update(
            CTYPE1="RA---TPV",
            CTYPE2="DEC--TPV",
            PV1_1=1.0,
            PV2_1=1.0,
            PV1_4=0.02,
            PV2_6=-0.03,
        )
    elif model_kind == "wire_implicit":
        header.update(
            CTYPE1="RA---TAN-SIP",
            CTYPE2="DEC--TAN-SIP",
            A_ORDER=2,
            B_ORDER=2,
            A_2_0=0.001,
            B_0_2=-0.001,
        )
    mask = torch.ones(4, 5)
    mask[0, 0] = 0.0
    observation = FieldObservation(
        image=torch.zeros(4, 5),
        header=header,
        wavelength=5000.0,
        weight=torch.linspace(0.5, 2.0, 20).reshape(4, 5),
        mask=mask,
        psf_sigma_pix=0.6,
        background_offset=0.2,
    )
    context = ForwardContext(extra={"split": "train"})
    bundle = field_observation_operator_bundle(observation)
    if calibrated:
        from torchsky.map.operators import (
            ConditionedBackgroundField,
            ConditionedKernelPSF,
            LearnedBackgroundOp,
            LearnedPSFOp,
            SpatioSpectralTemporalEncoder,
        )

        encoder = SpatioSpectralTemporalEncoder(
            spatial_center=(1.0, 1.0),
            spatial_scale=(2.0, 2.0),
            wavelength_center=5000.0,
            spectral_scale=500.0,
            spatial_frequencies=0,
            spectral_frequencies=0,
            temporal_frequencies=0,
        )
        psf = ConditionedKernelPSF(
            encoder, n_basis=1, kernel_size=3, init_fwhm_pix=1.5, hidden_dim=4
        ).eval()
        background = ConditionedBackgroundField(
            encoder, hidden_dim=4, hidden_layers=1
        ).eval()
        psf.requires_grad_(False)
        background.requires_grad_(False)
        bundle = bundle.with_updates(
            psf=LearnedPSFOp(
                psf,
                frozen=True,
                pixel_response_included=False,
                pixel_response_provenance="software_fixture",
            ),
            background=LearnedBackgroundOp(background, coordinate_frame="detector"),
        )
        calibration_before = [
            parameter.clone()
            for operator in (psf, background)
            for parameter in operator.parameters()
        ]
    with torch.no_grad():
        model.output.bias.add_(0.5)
        observation.image.copy_(
            render_field_observation(
                model,
                observation,
                num_spectral_samples=1,
                spatial_sampling="gauss_legendre_2x2",
                forward_context=context,
                operator_bundle=bundle,
            )
        )
        model.output.bias.sub_(0.5)
    # Unsupported pixels cannot influence the detector objective.
    observation.image[0, 0] = float("nan")
    fit_kwargs = {
        "loss_kind": "mse",
        "loss_reduction": "global",
        "num_spectral_samples": 1,
        "spatial_sampling": "gauss_legendre_2x2",
        "observation_forward_contexts": (context,),
        "observation_operator_bundles": (bundle,),
    }
    fitted = fit_field_model([observation], model, steps=1, lr=1.0e-3, **fit_kwargs)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    refined = _refine_neural_scene_pcg(
        fitted,
        [observation],
        fit_kwargs=fit_kwargs,
        steps=1,
        cg_iterations=4,
    )
    assert refined.final_loss < fitted.final_loss
    assert refined.optimizer_state is None and refined.amp_scaler_state is None
    assert any(
        not torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )
    prediction = render_field_observation(
        model,
        observation,
        num_spectral_samples=1,
        spatial_sampling="gauss_legendre_2x2",
        forward_context=context,
        operator_bundle=bundle,
    )
    support = mask > 0
    exact = (
        (prediction[support] - observation.image[support]).square()
        * observation.weight[support]
    ).sum() / observation.weight[support].sum()
    assert refined.final_loss == pytest.approx(float(exact.detach()), rel=1.0e-5)
    report = refined.diagnostics["inr_pcg_refinement"]
    assert report["steps"][0]["accepted_step_scale"] > 0
    assert report["pixels"] == 19
    assert report["steps"][0]["final_loss"] == pytest.approx(
        refined.final_loss, rel=1.0e-5
    )
    invalid_kwargs = {
        **fit_kwargs,
        "observation_forward_contexts": (
            ForwardContext(extra={"split": "validation"}),
        ),
    }
    with pytest.raises(ValueError, match="training observations only"):
        _refine_neural_scene_pcg(
            fitted, [observation], fit_kwargs=invalid_kwargs, steps=1
        )
    if calibrated:
        calibration_after = [
            parameter
            for operator in (psf, background)
            for parameter in operator.parameters()
        ]
        for before_parameter, after_parameter in zip(
            calibration_before, calibration_after, strict=True
        ):
            torch.testing.assert_close(
                before_parameter, after_parameter, rtol=0, atol=0
            )
        psf.requires_grad_(True)
        with pytest.raises(ValueError, match="frozen observation operators"):
            _refine_neural_scene_pcg(
                fitted, [observation], fit_kwargs=fit_kwargs, steps=1
            )
        psf.requires_grad_(False)
    elif model_kind == "sine_implicit":
        with tempfile.TemporaryDirectory() as artifact_root:
            path = Path(artifact_root) / "scene"
            artifact = _save_joint_field_artifact(
                path,
                result=refined,
                model_kind=model_kind,
                model_kwargs=kwargs,
                fit_kwargs=fit_kwargs,
                run_config={},
                contexts=[context],
                input_identities=[],
                calibration_audit=None,
                intensity_scale=1.0,
                normalization_training_count=1,
                exact_refined=True,
            )
            assert artifact["training_resume_available"] is False
            assert "neural PCG" in artifact["optimizer_status"]
            manifest = json.loads((path / "manifest.json").read_text())
            assert manifest["fit_config"]["inr_pcg_refinement"] == report
            saved = torch.load(
                path / manifest["selected_checkpoint"], weights_only=True
            )["model"]
            for name, value in model.state_dict().items():
                torch.testing.assert_close(saved[name], value, rtol=0, atol=0)
            training = torch.load(path / manifest["training_state"], weights_only=True)
            assert training["optimizer"] is None and training["amp_scaler"] is None
