"""Correctness of ``torchsky.transforms`` against ``astropy.visualization`` and NumPy.

Stretches and ZScale limits are compared with ``astropy.visualization``;
scaling, crop, composition, noise statistics and differentiability use NumPy or
analytic oracles.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchsky.transforms as tt

visualization = pytest.importorskip("astropy.visualization")


def _image(seed: int = 1, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape)


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize(("a", "q"), [(0.2, 8.0), (0.1, 8.0), (1.0, 2.0)])
def test_asinh_stretch_matches_astropy(a: float, q: float) -> None:
    x = _image()
    # torchsky: asinh(Q x / a) / asinh(Q / a)  ==  astropy AsinhStretch(a / Q)
    expected = visualization.AsinhStretch(a / q)(x)
    np.testing.assert_allclose(tt.AsinhStretch(a=a, Q=q)(_t(x)).numpy(), expected, atol=1e-14)


@pytest.mark.parametrize("a", [10.0, 1000.0])
def test_log_stretch_matches_astropy(a: float) -> None:
    x = _image(2)
    np.testing.assert_allclose(tt.LogStretch(a=a)(_t(x)).numpy(), visualization.LogStretch(a)(x), atol=1e-14)


@pytest.mark.parametrize("gamma", [0.25, 0.5, 2.0])
def test_power_stretch_matches_astropy(gamma: float) -> None:
    x = _image(3)
    np.testing.assert_allclose(
        tt.PowerStretch(gamma)(_t(x)).numpy(), visualization.PowerStretch(gamma)(x), atol=1e-14
    )


def _zscale_limits(image: np.ndarray) -> tuple[float, float]:
    """Recover (z1, z2) from the affine, unclamped part of the ZScale output."""
    out = tt.ZScale()(_t(image)).numpy()
    interior = (out > 0.0) & (out < 1.0)
    slope, intercept = np.polyfit(image[interior], out[interior], 1)
    lo = -intercept / slope
    return float(lo), float(lo + 1.0 / slope)


def _zscale_image() -> np.ndarray:
    # <= 1000 pixels so astropy uses every pixel (no sub-sampling differences)
    image = np.random.default_rng(4).normal(100.0, 5.0, (30, 30))
    image[3:5, 3:5] = 5000.0
    return image


def test_zscale_lower_limit_matches_astropy_zscale_interval() -> None:
    image = _zscale_image()
    lo_ref, _ = visualization.ZScaleInterval(contrast=0.25, max_reject=0.5).get_limits(image)
    lo, _ = _zscale_limits(image)
    assert lo == pytest.approx(float(lo_ref), rel=1e-9)


@pytest.mark.xfail(
    strict=True,
    reason="ZScale upper limit is ~7% below astropy's IRAF z2 on a 900-pixel image (subject finding)",
)
def test_zscale_upper_limit_matches_astropy_zscale_interval() -> None:
    image = _zscale_image()
    _, hi_ref = visualization.ZScaleInterval(contrast=0.25, max_reject=0.5).get_limits(image)
    _, hi = _zscale_limits(image)
    assert hi == pytest.approx(float(hi_ref), rel=1e-3)


def test_minmax_normalize_and_robust_scale_oracles() -> None:
    x = _image(5)
    scaled = tt.MinMaxScale(-1.0, 2.0)(_t(x)).numpy()
    np.testing.assert_allclose(scaled, -1.0 + 3.0 * (x - x.min()) / (x.max() - x.min()), atol=1e-14)

    normal = tt.Normalize()(_t(x)).numpy()
    assert normal.mean() == pytest.approx(0.0, abs=1e-12)
    assert normal.std(ddof=1) == pytest.approx(1.0, rel=1e-12)

    odd = x.ravel()[:4095].reshape(63, 65)  # odd count: median is a sample, no tie convention
    q1, med, q3 = np.percentile(odd, [25, 50, 75])
    np.testing.assert_allclose(tt.RobustScale()(_t(odd)).numpy(), (odd - med) / (q3 - q1), atol=1e-4)


@pytest.mark.xfail(
    strict=True,
    reason="RobustScale centres on the lower median for even sample counts (np.median averages); subject finding",
)
def test_robust_scale_centre_is_the_conventional_median_for_even_counts() -> None:
    x = _image(5)  # 4096 samples
    out = tt.RobustScale(scale=False)(_t(x)).numpy()
    np.testing.assert_allclose(out, x - np.median(x), atol=1e-12)


def test_center_crop_selects_the_central_window() -> None:
    x = np.arange(10 * 12, dtype=np.float64).reshape(1, 10, 12)
    out = tt.CenterCrop((4, 6))(_t(x)).numpy()
    assert out.shape == (1, 4, 6)
    np.testing.assert_array_equal(out, x[:, 3:7, 3:9])


def test_compose_applies_left_to_right() -> None:
    x = _image(6)
    composed = tt.Compose([tt.PowerStretch(0.5), tt.MinMaxScale(0.0, 10.0)])(_t(x)).numpy()
    stretched = np.sqrt(x)
    expected = 10.0 * (stretched - stretched.min()) / (stretched.max() - stretched.min())
    np.testing.assert_allclose(composed, expected, atol=1e-12)


def test_gaussian_noise_has_requested_moments_and_is_seed_deterministic() -> None:
    zeros = torch.zeros(200_000, dtype=torch.float64)
    torch.manual_seed(0)
    first = tt.GaussianNoise(std=0.5)(zeros)
    torch.manual_seed(0)
    second = tt.GaussianNoise(std=0.5)(zeros)
    torch.testing.assert_close(first, second)
    assert float(first.mean()) == pytest.approx(0.0, abs=5e-3)
    assert float(first.std()) == pytest.approx(0.5, rel=1e-2)


def test_poisson_noise_preserves_the_mean_signal() -> None:
    torch.manual_seed(1)
    signal = torch.full((200_000,), 30.0, dtype=torch.float64)
    noisy = tt.PoissonNoise(lam=1.0)(signal)
    assert float(noisy.mean()) == pytest.approx(30.0, rel=1e-2)
    assert float(noisy.var()) == pytest.approx(30.0, rel=3e-2)


@pytest.mark.parametrize("stretch", [tt.AsinhStretch(), tt.LogStretch(), tt.PowerStretch(0.5)])
def test_stretches_are_monotonic_and_differentiable(stretch) -> None:
    x = torch.linspace(0.01, 1.0, 50, dtype=torch.float64, requires_grad=True)
    y = stretch(x)
    assert bool((y[1:] > y[:-1]).all())
    y.sum().backward()
    assert x.grad is not None and bool((x.grad > 0).all())
