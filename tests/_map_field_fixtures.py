"""Small deterministic map-field fixtures for extracted SMACS evaluations."""

from __future__ import annotations

import torch
from torch import nn

from torchsky.map.field import SpectralPowerFoamField
from torchsky.field import Domain as FieldDomain, PSFSpec, model as make_field_model
from torchsky.scene import FieldObservation


def _header() -> dict[str, object]:
    return {
        "NAXIS": 2,
        "NAXIS1": 6,
        "NAXIS2": 6,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 3.5,
        "CRPIX2": 3.5,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def _foam() -> SpectralPowerFoamField:
    domain = FieldDomain(
        center_ra_deg=12.0,
        center_dec_deg=-3.0,
        spatial_scale_deg=0.02,
        wavelength_center=5000.0,
        wavelength_scale=500.0,
        coord_mode="local_tangent",
    )
    model = make_field_model(
        "spectral_power_foam",
        domain,
        sites=torch.tensor([[-0.25, 0.0], [0.25, 0.0]], dtype=torch.float64),
        radii=torch.tensor([0.8, 0.9], dtype=torch.float64),
        cell_coefficients=torch.tensor(
            [[0.5, 0.1], [0.8, -0.1]], dtype=torch.float64
        ),
        cell_gradient_coefficients=torch.tensor(
            [[[0.01, 0.02], [0.03, 0.04]], [[-0.01, 0.0], [0.02, -0.03]]],
            dtype=torch.float64,
        ),
        minimum_radius=0.1,
        temperature_start=3.0,
        temperature_end=12.0,
    )
    assert isinstance(model, SpectralPowerFoamField)
    model.set_curriculum_progress(0.5)
    return model


class Scene(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(
        self,
        ra: torch.Tensor,
        dec: torch.Tensor,
        wavelength: torch.Tensor,
        *,
        forward_context=None,
    ):
        del forward_context
        wavelength = torch.broadcast_to(wavelength, ra.shape)
        coords = torch.stack(
            ((ra - 1) * 10, dec * 10, wavelength / 2), -1
        ).to(self.net[0].weight.dtype)
        return self.net(coords).squeeze(-1)


def observation(side: int = 33) -> FieldObservation:
    image = torch.linspace(0.1, 0.9, side * (side + 3)).reshape(side, side + 3)
    return FieldObservation(
        image=image,
        header={
            "NAXIS1": side + 3,
            "NAXIS2": side,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 1.0,
            "CRVAL2": 0.0,
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CDELT1": 0.001,
            "CDELT2": 0.001,
        },
        weight=1 + image,
        mask=image > 0.2,
        spectral_min=1.0,
        spectral_max=2.0,
        psf_spec=PSFSpec.kernel(
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, 3.0, 2.0], [0.0, 1.0, 1.0]]
            )
        ),
        detector_background=torch.full_like(image, 0.04),
    )
