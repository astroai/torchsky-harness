import numpy as np
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs.core import WCS


def _tpv_header() -> dict:
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0
    header["CRVAL1"] = 200.0
    header["CRVAL2"] = -20.0
    header["CD1_1"] = -2.8e-4
    header["CD1_2"] = 1.2e-7
    header["CD2_1"] = -1.1e-7
    header["CD2_2"] = 2.8e-4

    # Keep a mix of linear and non-linear terms; this remains stable for forward parity.
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = 0.0
    header["PV1_4"] = 2.0e-4
    header["PV1_5"] = -3.0e-4
    header["PV1_6"] = 1.5e-4
    header["PV1_7"] = 2.0e-6
    header["PV1_39"] = 2.0e-11

    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.0
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4
    header["PV2_5"] = 2.5e-4
    header["PV2_6"] = -2.0e-4
    header["PV2_7"] = -1.5e-6
    header["PV2_39"] = -1.0e-11

    return header


def _tpv_affine_linear_header() -> dict:
    header = _tpv_header()
    # Pure affine TPV (non-identity, with offsets and cross terms)
    for k in list(header.keys()):
        if k.startswith("PV1_") or k.startswith("PV2_"):
            del header[k]
    header["PV1_0"] = 0.05
    header["PV1_1"] = 1.12
    header["PV1_2"] = -0.08
    # Axis-2 TPV uses swapped arguments: PV2_2 contributes u, PV2_1 contributes v
    header["PV2_0"] = -0.03
    header["PV2_1"] = 0.04
    header["PV2_2"] = 0.91
    return header


def _tpv_invertible_rect_header() -> dict:
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 768
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 384.0
    header["CRVAL1"] = 12.0
    header["CRVAL2"] = -3.0
    header["CD1_1"] = -0.004465
    header["CD1_2"] = 0.0005488
    header["CD2_1"] = 0.0005488
    header["CD2_2"] = 0.004465
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = -0.02
    header["PV1_4"] = 2.0e-4 * 0.22
    header["PV1_5"] = -3.0e-4 * 0.22
    header["PV1_6"] = 1.5e-4 * 0.22
    header["PV1_7"] = 2.0e-6 * 0.22
    header["PV1_39"] = 2.0e-11 * 0.22
    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.02
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4 * 0.22
    header["PV2_5"] = 2.5e-4 * 0.22
    header["PV2_6"] = -2.0e-4 * 0.22
    header["PV2_7"] = -1.5e-6 * 0.22
    header["PV2_39"] = -1.0e-11 * 0.22
    return header


def _sample_pixels() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(430.0, 590.0, 9, dtype=np.float64)
    y = np.linspace(420.0, 600.0, 9, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def test_tpv_forward_parity_with_astropy() -> None:
    header = _tpv_header()
    awcs = AstropyWCS(header)
    twcs = WCS(dict(header))

    x, y = _sample_pixels()

    ra_ast, dec_ast = awcs.all_pix2world(x, y, 0)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=0
    )

    np.testing.assert_allclose(_ra_delta(ra_t.cpu().numpy(), ra_ast), 0.0, atol=1e-8)
    np.testing.assert_allclose(dec_t.cpu().numpy() - dec_ast, 0.0, atol=1e-8)














def test_tpv_inverse_parity_with_astropy_on_invertible_broad_field() -> None:
    header = _tpv_invertible_rect_header()
    awcs = AstropyWCS(header)
    twcs = WCS(dict(header))

    x = np.linspace(64.0, 960.0, 33, dtype=np.float64)
    y = np.linspace(64.0, 704.0, 29, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    ra_ast, dec_ast = awcs.all_pix2world(xx.ravel(), yy.ravel(), 0)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_ast),
        torch.from_numpy(dec_ast),
        origin=0,
    )

    np.testing.assert_allclose(x_t.cpu().numpy(), xx.ravel(), atol=1e-6)
    np.testing.assert_allclose(y_t.cpu().numpy(), yy.ravel(), atol=1e-6)
