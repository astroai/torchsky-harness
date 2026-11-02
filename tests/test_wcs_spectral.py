import numpy as np
import pytest
import torch
import astropy.units as u
from astropy.wcs import WCS as AstropyWCS

import torchsky
from torchsky.wcs import frequency_to_spectral, spectral_to_frequency


def _wave_1d_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 1,
        "CTYPE1": "WAVE",
        "CRPIX1": 1.0,
        "CRVAL1": 4000.0,
        "CDELT1": 10.0,
    }


def _cube_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 3,
        "CTYPE1": "RA---TAN",  # type: ignore
        "CTYPE2": "DEC--TAN",  # type: ignore
        "CTYPE3": "WAVE",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRPIX3": 1.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CRVAL3": 5000.0,
        "CDELT1": -0.01,  # type: ignore
        "CDELT2": 0.01,
        "CDELT3": 2.0,
    }


@pytest.mark.parametrize("origin", [0, 1])
def test_wcs_spectral_1d_matches_astropy(origin: int) -> None:
    header = _wave_1d_header()  # type: ignore

    twcs = torchsky.WCS(header)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore

    pixels = np.array([0.0, 1.0, 10.0, 25.0], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels[:, None]), origin=origin)
    world_a = awcs.all_pix2world(pixels[:, None], origin)

    np.testing.assert_allclose(world_t.cpu().numpy(), world_a, atol=1e-12)  # type: ignore

    pixels_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=origin)
    pixels_a = awcs.all_world2pix(world_a, origin)

    np.testing.assert_allclose(pixels_t.cpu().numpy(), pixels_a, atol=1e-9)  # type: ignore
    np.testing.assert_allclose(pixels_t.cpu().numpy(), pixels[:, None], atol=1e-9)  # type: ignore




@pytest.mark.parametrize("origin", [0, 1])
def test_wcs_spatial_plus_spectral_3d_matches_astropy(origin: int) -> None:
    header = _cube_header()  # type: ignore

    twcs = torchsky.WCS(header)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore

    # Keep points around CRPIX for robust inversion across projections.
    x = np.linspace(20.0, 44.0, 7, dtype=np.float64)
    y = np.linspace(20.0, 44.0, 7, dtype=np.float64)
    z = np.array([0.0, 3.0, 11.0], dtype=np.float64)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    pixels = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)  # type: ignore

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=origin)
    world_a = awcs.all_pix2world(pixels, origin)

    np.testing.assert_allclose(world_t.cpu().numpy(), world_a, atol=1e-8)  # type: ignore

    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=origin)
    pix_a = awcs.all_world2pix(world_a, origin)

    np.testing.assert_allclose(pix_t.cpu().numpy(), pix_a, atol=1e-6)  # type: ignore
    np.testing.assert_allclose(pix_t.cpu().numpy(), pixels, atol=1e-6)  # type: ignore


def test_spectral_frequency_wavelength_energy_match_astropy_units() -> None:
    freq = torch.tensor([1.0e9, 1.42e9, 3.0e14], dtype=torch.float64)
    freq_q = freq.numpy() * u.Hz

    wave = frequency_to_spectral(freq, "WAVE", unit="m")
    energy = frequency_to_spectral(freq, "ENER", unit="eV")

    np.testing.assert_allclose(
        wave.numpy(), freq_q.to(u.m, equivalencies=u.spectral()).value, rtol=1e-12
    )
    np.testing.assert_allclose(
        energy.numpy(), freq_q.to(u.eV, equivalencies=u.spectral()).value, rtol=1e-12
    )
    torch.testing.assert_close(spectral_to_frequency(wave, "WAVE", unit="m"), freq)
    torch.testing.assert_close(spectral_to_frequency(energy, "ENER", unit="eV"), freq)


def test_spectral_wavenumber_matches_astropy_units() -> None:
    freq = torch.tensor([3.0e10, 3.0e13, 6.0e14], dtype=torch.float64)
    freq_q = freq.numpy() * u.Hz

    wavn_cm = frequency_to_spectral(freq, "WAVN", unit="cm-1")
    expected = freq_q.to(1 / u.cm, equivalencies=u.spectral()).value

    np.testing.assert_allclose(wavn_cm.numpy(), expected, rtol=1e-12)
    torch.testing.assert_close(
        spectral_to_frequency(wavn_cm, "WAVN", unit="cm-1"),
        freq,
        rtol=1e-12,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("kind", "equiv"),
    [
        ("VRAD", "radio"),
        ("VOPT", "optical"),
        ("VELO", "relativistic"),
    ],
)
def test_spectral_velocity_conventions_match_astropy(kind: str, equiv: str) -> None:
    rest_hz = 1420.40575177e6
    freq = torch.tensor([1.419e9, rest_hz, 1.421e9], dtype=torch.float64)
    rest = rest_hz * u.Hz
    freq_q = freq.numpy() * u.Hz
    equivalency = {
        "radio": u.doppler_radio(rest),
        "optical": u.doppler_optical(rest),
        "relativistic": u.doppler_relativistic(rest),
    }[equiv]

    vel = frequency_to_spectral(freq, kind, unit="m/s", rest_frequency=rest_hz)
    np.testing.assert_allclose(
        vel.numpy(), freq_q.to(u.m / u.s, equivalencies=equivalency).value, rtol=1e-12
    )
    roundtrip = spectral_to_frequency(vel, kind, unit="m/s", rest_frequency=rest_hz)
    torch.testing.assert_close(roundtrip, freq, rtol=1e-12, atol=1e-6)








@pytest.mark.parametrize("ctype", ["FREQ-LOG", "WAVE-LOG", "WAVN-LOG"])
def test_wcs_spectral_log_axis_matches_astropy(ctype: str) -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": ctype,
        "CRPIX1": 1.0,
        "CRVAL1": 10.0,
        "CDELT1": 0.1,
    }
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=0)

    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pix_t.numpy(), pixels, atol=1e-9)


@pytest.mark.parametrize("ctype", ["WAVE-F2W", "FREQ-W2F"])
def test_wcs_spectral_compound_reciprocal_axis_matches_astropy(ctype: str) -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": ctype,
        "CRPIX1": 1.0,
        "CRVAL1": 10.0,
        "CDELT1": 0.1,
    }
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=0)

    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pix_t.numpy(), pixels, atol=1e-9)


_SUPPORTED_COMPOUND_CASES = [
    "FREQ-F2F",
    "FREQ-W2F",
    "FREQ-A2F",
    "FREQ-V2F",
    "WAVE-F2W",
    "WAVE-W2W",
    "WAVE-A2W",
    "WAVE-V2W",
    "AWAV-F2A",
    "AWAV-W2A",
    "AWAV-A2A",
    "AWAV-V2A",
    "VELO-F2V",
    "VELO-W2V",
    "VELO-A2V",
    "VELO-V2V",
]


def _compound_header(ctype: str) -> dict[str, float | int | str]:
    stype = ctype.split("-", 1)[0]
    header: dict[str, float | int | str] = {
        "NAXIS": 1,
        "NAXIS1": 128,
        "CTYPE1": ctype,
        "CRPIX1": 1.0,
        "RESTFRQ": 1.42040575177e9,
    }
    if stype == "FREQ":
        header.update({"CRVAL1": 1.420e9, "CDELT1": 1.0e5, "CUNIT1": "Hz"})
    elif stype in {"WAVE", "AWAV"}:
        header.update({"CRVAL1": 0.21, "CDELT1": 1.0e-4, "CUNIT1": "m"})
    elif stype == "VELO":
        header.update({"CRVAL1": 10.0, "CDELT1": 0.1, "CUNIT1": "m/s"})
    elif stype == "ENER":
        header.update({"CRVAL1": 9.407e-25, "CDELT1": 6.626e-29, "CUNIT1": "J"})
    elif stype == "WAVN":
        header.update({"CRVAL1": 4.737, "CDELT1": 0.01, "CUNIT1": "m-1"})
    elif stype == "ZOPT":
        header.update({"CRVAL1": 0.0, "CDELT1": 1.0e-4})
    elif stype == "BETA":
        header.update({"CRVAL1": 0.0, "CDELT1": 1.0e-6})
    return header


@pytest.mark.parametrize("ctype", _SUPPORTED_COMPOUND_CASES)
def test_wcs_spectral_supported_compound_matrix_matches_astropy(ctype: str) -> None:
    header = _compound_header(ctype)
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=0)

    assert np.isfinite(world_t.numpy()).all()
    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=1e-11, atol=1e-7)
    np.testing.assert_allclose(pix_t.numpy(), pixels, rtol=1e-10, atol=2e-6)






def _grism_header(ctype: str) -> dict[str, float | int | str]:
    if ctype.startswith("FREQ"):
        crval = 6.0e14
        cdelt = 1.0e9
        cunit = "Hz"
    else:
        crval = 5.0e-7
        cdelt = 1.0e-9
        cunit = "m"
    return {
        "NAXIS": 1,
        "NAXIS1": 128,
        "CTYPE1": ctype,
        "CUNIT1": cunit,
        "CRPIX1": 1.0,
        "CRVAL1": crval,
        "CDELT1": cdelt,
        "PV1_0": 300000.0,
        "PV1_1": 1.0,
        "PV1_2": 15.0,
        "PV1_3": 1.5,
        "PV1_4": -1.0e5,
        "PV1_5": 0.0,
        "PV1_6": 10.0,
    }


@pytest.mark.parametrize(
    "ctype",
    ["WAVE-GRI", "WAVE-GRA", "AWAV-GRI", "AWAV-GRA", "FREQ-GRI", "FREQ-GRA"],
)
def test_wcs_spectral_grism_axes_match_astropy(ctype: str) -> None:
    header = _grism_header(ctype)
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [2.0], [10.0], [1000.0]], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=0)

    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=5e-9, atol=1e-20)
    np.testing.assert_allclose(pix_t.numpy(), pixels, rtol=1e-12, atol=5e-6)
    if ctype == "FREQ-GRA":
        # Leftover ~0.6 Hz on ~1e14 Hz is float32 Edlén noise, not 29 Hz.
        rel = np.max(np.abs(world_t.numpy() - world_a) / np.clip(np.abs(world_a), 1.0, None))
        assert rel < 1e-11


@pytest.mark.parametrize(
    "header",
    [
        {
            "NAXIS": 1,
            "NAXIS1": 128,
            "CTYPE1": "WAVE-V2W",
            "CUNIT1": "m",
            "CRPIX1": 1.0,
            "CRVAL1": 0.21,
            "CDELT1": 1.0e-4,
            "RESTFRQ": 1.42040575177e9,
        },
        {
            "NAXIS": 1,
            "NAXIS1": 128,
            "CTYPE1": "VELO-F2V",
            "CUNIT1": "m/s",
            "CRPIX1": 1.0,
            "CRVAL1": 10.0,
            "CDELT1": 0.1,
            "RESTFRQ": 1.42040575177e9,
        },
        {
            "NAXIS": 1,
            "NAXIS1": 128,
            "CTYPE1": "FREQ-V2F",
            "CUNIT1": "Hz",
            "CRPIX1": 1.0,
            "CRVAL1": 1.420e9,
            "CDELT1": 1.0e5,
            "RESTFRQ": 1.42040575177e9,
        },
    ],
)
def test_wcs_spectral_velocity_compound_axes_match_astropy(
    header: dict[str, float | int | str],
) -> None:
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=0)

    assert np.isfinite(world_t.numpy()).all()
    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=1e-11, atol=1e-7)
    np.testing.assert_allclose(pix_t.numpy(), pixels, rtol=1e-11, atol=1e-6)


@pytest.mark.parametrize(
    "ctype,cunit,crval,cdelt",
    [
        ("FREQ-W2F", "Hz", 1.420e9, 1.0e5),
        ("WAVE-F2W", "m", 0.21, 1.0e-4),
        ("AWAV-F2A", "m", 5.0e-7, 1.0e-10),
        ("ENER", "eV", 1.0, 0.01),
        ("WAVN", "m-1", 1.0e4, 10.0),
        ("VOPT", "m/s", 1.0e4, 100.0),
        ("WAVE-V2W", "m", 0.21, 1.0e-4),
    ],
)
def test_paper3_named_algorithm_chains_match_astropy(
    ctype: str, cunit: str, crval: float, cdelt: float
) -> None:
    header = {
        "NAXIS": 1,
        "NAXIS1": 64,
        "CTYPE1": ctype,
        "CUNIT1": cunit,
        "CRPIX1": 1.0,
        "CRVAL1": crval,
        "CDELT1": cdelt,
        "RESTFRQ": 1.42040575177e9,
    }
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [8.0], [16.0]], dtype=np.float64)
    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    if ctype == "ENER":
        world_t = world_t * 1.602176634e-19
    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=5e-9, atol=1e-18)


def test_wcs_awav_1d_matches_astropy() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "AWAV",
        "CUNIT1": "Angstrom",
        "CRPIX1": 1.0,
        "CRVAL1": 5000.0,
        "CDELT1": 1.0,
    }
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array([[0.0], [1.0], [10.0], [25.0]], dtype=np.float64)
    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    # Astropy returns SI meters; torchsky keeps CUNIT (Angstrom).
    np.testing.assert_allclose(
        world_t.numpy() * 1.0e-10, world_a, rtol=1e-10, atol=1e-18
    )
    pix_t = twcs.world_to_pixel(world_t, origin=0)
    np.testing.assert_allclose(pix_t.numpy(), pixels, rtol=1e-12, atol=1e-9)






def test_spatial_cube_coupled_cd_roundtrip_matches_astropy() -> None:
    header = {
        "NAXIS": 3,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRPIX3": 5.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CRVAL3": 5000.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD1_3": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
        "CD2_3": 0.0,
        "CD3_1": 0.1,
        "CD3_2": -0.05,
        "CD3_3": 2.0,
    }
    twcs = torchsky.WCS(header)
    awcs = AstropyWCS(header)
    pixels = np.array(
        [[32.0, 32.0, 5.0], [30.0, 34.0, 7.0], [40.0, 28.0, 2.0]],
        dtype=np.float64,
    )
    world_t = twcs.pixel_to_world(torch.from_numpy(pixels), origin=0)
    world_a = awcs.all_pix2world(pixels, 0)
    pix_t = twcs.world_to_pixel(torch.from_numpy(world_a), origin=0)
    np.testing.assert_allclose(world_t.numpy(), world_a, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(pix_t.numpy(), pixels, rtol=1e-10, atol=1e-6)
