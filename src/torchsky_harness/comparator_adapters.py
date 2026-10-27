"""Fail-closed adapters for the bounded external comparator campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


class UnavailableEvidence(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ComparatorProducts:
    science: np.ndarray
    weight: np.ndarray | None = None
    variance_components: Mapping[str, np.ndarray] = field(default_factory=dict)
    dq: np.ndarray | None = None
    context: np.ndarray | None = None
    nexp: np.ndarray | None = None
    rejection: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")
_FLOAT_TEXT = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?$")


def _normalize_numeric_header_values(header: Any) -> list[str]:
    normalized: list[str] = []
    for key in tuple(header):
        value = header.get(key)
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if _INTEGER_TEXT.fullmatch(stripped):
            header[key] = int(stripped)
            normalized.append(str(key))
        elif _FLOAT_TEXT.fullmatch(stripped):
            header[key] = float(stripped)
            normalized.append(str(key))
    return normalized


def adapt_stcal_resample_output(output: Mapping[str, Any]) -> ComparatorProducts:
    if output.get("data") is None:
        raise UnavailableEvidence("stcal resample output has no data plane")
    variances = {
        name: _numpy(output[name])
        for name in ("var_poisson", "var_rnoise", "var_flat")
        if output.get(name) is not None
    }
    return ComparatorProducts(
        science=_numpy(output["data"]),
        weight=None if output.get("wht") is None else _numpy(output["wht"]),
        variance_components=variances,
        dq=None if output.get("dq") is None else _numpy(output["dq"]),
        context=None if output.get("con") is None else _numpy(output["con"]),
        nexp=None if output.get("nexp") is None else _numpy(output["nexp"]),
        metadata={"adapter": "stcal.resample.Resample"},
    )


def run_stcal_outlier_adapter(
    science: Any,
    error: Any,
    blotted_reference: Any,
    *,
    snr: tuple[float, float],
    scale: tuple[float, float],
    background: float = 0.0,
) -> ComparatorProducts:
    try:
        from stcal.outlier_detection.utils import flag_resampled_crs
    except Exception as exc:
        raise UnavailableEvidence(f"stcal outlier adapter unavailable: {exc}") from exc
    mask = flag_resampled_crs(
        _numpy(science),
        _numpy(error),
        _numpy(blotted_reference),
        float(snr[0]),
        float(snr[1]),
        float(scale[0]),
        float(scale[1]),
        float(background),
    )
    return ComparatorProducts(
        science=_numpy(science),
        rejection=np.asarray(mask, dtype=bool),
        metadata={
            "adapter": "stcal.outlier_detection.flag_resampled_crs",
            "snr": [float(value) for value in snr],
            "scale": [float(value) for value in scale],
        },
    )


def run_reproject_adaptive_adapter(
    image: Any,
    source_header: Mapping[str, Any],
    target_header: Mapping[str, Any],
    target_shape: tuple[int, int],
    *,
    conserve_flux: bool,
    boundary_mode: str = "ignore",
) -> ComparatorProducts:
    try:
        from astropy.wcs import WCS as AstropyWCS
        from reproject import reproject_adaptive
    except Exception as exc:
        raise UnavailableEvidence(f"reproject_adaptive unavailable: {exc}") from exc
    output, footprint = reproject_adaptive(
        (_numpy(image), AstropyWCS(dict(source_header))),
        AstropyWCS(dict(target_header)),
        shape_out=target_shape,
        conserve_flux=bool(conserve_flux),
        boundary_mode=boundary_mode,
        return_footprint=True,
    )
    return ComparatorProducts(
        science=np.asarray(output),
        weight=np.asarray(footprint),
        nexp=(np.asarray(footprint) > 0).astype(np.int32),
        metadata={
            "adapter": "reproject.reproject_adaptive",
            "conserve_flux": bool(conserve_flux),
            "boundary_mode": boundary_mode,
        },
    )


def _read_optional_plane(path: Path, names: Sequence[str]) -> np.ndarray | None:
    import torchfits

    for name in names:
        try:
            value = torchfits.read(str(path), hdu=name, mmap=False)
        except Exception:
            continue
        return _numpy(value)
    return None


def _serializable_header(header: Mapping[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key in header:
        value = header[key]
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            serialized[str(key)] = value
        elif isinstance(value, (list, tuple)):
            serialized[str(key)] = [
                item.item() if isinstance(item, np.generic) else item for item in value
            ]
        else:
            serialized[str(key)] = str(value)
    return serialized


def _read_product_header(path: Path) -> dict[str, Any]:
    import torchfits

    for hdu in ("SCI", 0):
        try:
            return _serializable_header(dict(torchfits.read_header(str(path), hdu)))
        except Exception:
            continue
    raise UnavailableEvidence(f"AstroDrizzle product has no readable header: {path}")


def adapt_astrodrizzle_product(path: Path) -> ComparatorProducts:
    science = _read_optional_plane(path, ("SCI", "PRIMARY"))
    if science is None:
        raise UnavailableEvidence(f"AstroDrizzle product has no SCI plane: {path}")
    weight = _read_optional_plane(path, ("WHT",))
    context = _read_optional_plane(path, ("CTX", "CON"))
    dq = _read_optional_plane(path, ("DQ",))
    nexp = _read_optional_plane(path, ("NEXP",))
    output_header = _read_product_header(path)
    return ComparatorProducts(
        science=science,
        weight=weight,
        dq=dq,
        context=context,
        nexp=nexp,
        metadata={
            "adapter": "drizzlepac.astrodrizzle",
            "product": path.name,
            "output_header": output_header,
            "target_wcs": output_header,
            "target_shape": [int(science.shape[-2]), int(science.shape[-1])],
        },
    )


def run_astrodrizzle_adapter(
    inputs: Sequence[Path],
    output_directory: Path,
    *,
    output_name: str = "bounded_hst",
    options: Mapping[str, Any] | None = None,
) -> ComparatorProducts:
    try:
        from drizzlepac.astrodrizzle import AstroDrizzle
    except Exception as exc:
        raise UnavailableEvidence(f"AstroDrizzle unavailable: {exc}") from exc
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise UnavailableEvidence("AstroDrizzle inputs missing: " + ", ".join(missing))
    output_directory.mkdir(parents=True, exist_ok=True)
    import torchfits

    prepared_inputs: list[Path] = []
    removed_reference_keywords: dict[str, list[str]] = {}
    normalized_numeric_keywords: dict[str, list[str]] = {}
    for index, source in enumerate(inputs):
        destination = output_directory / f"input-{index:02d}-{source.name}"
        removed: list[str] = []
        normalized: list[str] = []
        with torchfits.open(str(source)) as hdul:
            for hdu in hdul:
                normalized.extend(_normalize_numeric_header_values(hdu.header))
                for keyword in ("DGEOFILE", "NPOLFILE", "D2IMFILE"):
                    if keyword in hdu.header:
                        hdu.header.pop(keyword)
                        removed.append(keyword)
            hdul.write(str(destination), overwrite=True)
        prepared_inputs.append(destination)
        removed_reference_keywords[source.name] = sorted(set(removed))
        normalized_numeric_keywords[source.name] = sorted(set(normalized))
    output_root = output_directory / output_name
    parameters = {
        "output": str(output_root),
        "build": True,
        "clean": True,
        "preserve": False,
        "context": True,
        "resetbits": 0,
        "static": False,
        "skysub": False,
        "driz_cr": True,
        "median": True,
        "blot": True,
        "updatewcs": False,
    }
    parameters.update(dict(options or {}))
    try:
        AstroDrizzle([str(path) for path in prepared_inputs], **parameters)
    except Exception as exc:
        raise UnavailableEvidence(f"AstroDrizzle execution failed: {exc}") from exc
    products = sorted(output_directory.glob(f"{output_name}_dr[cz].fits"))
    if not products:
        products = sorted(output_directory.glob(f"{output_name}*.fits"))
    if not products:
        raise UnavailableEvidence("AstroDrizzle produced no bounded output product")
    result = adapt_astrodrizzle_product(products[0])
    rejection_planes = []
    for candidate in sorted(output_directory.glob("*_crmask.fits")):
        plane = _read_optional_plane(candidate, ("PRIMARY", "SCI"))
        if plane is not None:
            rejection_planes.append(np.asarray(plane) == 0)
    return ComparatorProducts(
        science=result.science,
        weight=result.weight,
        variance_components=result.variance_components,
        dq=result.dq,
        context=result.context,
        nexp=result.nexp,
        rejection=np.stack(rejection_planes) if rejection_planes else None,
        metadata={
            **dict(result.metadata),
            "parameters": _serializable_header(parameters),
            "removed_external_distortion_references": removed_reference_keywords,
            "normalized_numeric_header_keyword_count": {
                name: len(values)
                for name, values in normalized_numeric_keywords.items()
            },
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_association_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "id",
        "instrument",
        "availability",
        "association_env",
        "association_sha256",
        "crds_context",
        "members",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported association manifest schema")
    if payload["instrument"] not in {"NIRSPEC", "MIRI"}:
        raise ValueError(f"{path}: instrument must be NIRSPEC or MIRI")
    if payload["availability"] not in {"available", "unavailable"}:
        raise ValueError(f"{path}: invalid availability")
    if not isinstance(payload["members"], list):
        raise ValueError(f"{path}: members must be an array")
    return payload


def resolve_external_association(manifest: Mapping[str, Any]) -> Path:
    if manifest.get("availability") != "available":
        raise UnavailableEvidence(
            str(manifest.get("reason") or "external association unavailable")
        )
    context = manifest.get("crds_context")
    if not isinstance(context, str) or not context or context == "UNAVAILABLE":
        raise UnavailableEvidence("external association has no pinned CRDS context")
    active_context = os.environ.get("CRDS_CONTEXT")
    if active_context != context:
        raise UnavailableEvidence(
            f"CRDS_CONTEXT mismatch: expected {context!r}, found {active_context!r}"
        )
    environment = str(manifest["association_env"])
    value = os.environ.get(environment)
    if not value:
        raise UnavailableEvidence(f"{environment} is not set")
    association = Path(value).expanduser().resolve()
    if not association.is_file():
        raise UnavailableEvidence(f"association does not exist: {association}")
    expected = manifest.get("association_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise UnavailableEvidence("external association has no pinned sha256")
    actual = _sha256(association)
    if actual != expected:
        raise UnavailableEvidence(
            f"association sha256 mismatch: expected {expected}, found {actual}"
        )
    return association


def load_external_asset_manifest(
    path: Path, *, expected_family: str | None = None
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "id",
        "family",
        "availability",
        "asset_env",
        "asset_sha256",
        "required_identity_fields",
        "identity",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported external asset manifest schema")
    if expected_family is not None and payload["family"] != expected_family:
        raise ValueError(
            f"{path}: expected family {expected_family!r}, found {payload['family']!r}"
        )
    if payload["availability"] not in {"available", "unavailable"}:
        raise ValueError(f"{path}: invalid availability")
    fields = payload["required_identity_fields"]
    identity = payload["identity"]
    if not isinstance(fields, list) or any(
        not isinstance(value, str) or not value for value in fields
    ):
        raise ValueError(f"{path}: required_identity_fields must be string values")
    if not isinstance(identity, dict):
        raise ValueError(f"{path}: identity must be an object")
    return payload


def resolve_external_asset(manifest: Mapping[str, Any]) -> Path:
    if manifest.get("availability") != "available":
        raise UnavailableEvidence(str(manifest.get("reason") or "external asset unavailable"))
    identity = manifest.get("identity")
    required_fields = manifest.get("required_identity_fields")
    if not isinstance(identity, Mapping) or not isinstance(required_fields, list):
        raise UnavailableEvidence("external asset identity declaration is invalid")
    incomplete = [
        str(field)
        for field in required_fields
        if not identity.get(field) or identity.get(field) == "UNAVAILABLE"
    ]
    if incomplete:
        raise UnavailableEvidence(
            "external asset identity incomplete: " + ", ".join(incomplete)
        )
    environment = str(manifest.get("asset_env", ""))
    value = os.environ.get(environment)
    if not value:
        raise UnavailableEvidence(f"{environment} is not set")
    asset = Path(value).expanduser().resolve()
    if not asset.is_file():
        raise UnavailableEvidence(f"external asset does not exist: {asset}")
    expected = manifest.get("asset_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise UnavailableEvidence("external asset has no pinned sha256")
    actual = _sha256(asset)
    if actual != expected:
        raise UnavailableEvidence(
            f"external asset sha256 mismatch: expected {expected}, found {actual}"
        )
    return asset


def _load_array_product(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise UnavailableEvidence("external JSON product must be an object")
        return payload
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}
    raise UnavailableEvidence(f"unsupported external product format: {path.suffix}")


def run_supplied_centroid_adapter(
    peer: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("family") != "astrometry-v1":
        raise UnavailableEvidence("supplied-centroid asset has the wrong family")
    identity = manifest.get("identity", {})
    if identity.get("peer") != peer:
        raise UnavailableEvidence(
            f"supplied-centroid peer identity mismatch: expected {peer!r}"
        )
    if identity.get("detection") is not False:
        raise UnavailableEvidence("supplied-centroid contract must disable detection")
    if identity.get("centroid_source") != "supplied":
        raise UnavailableEvidence("centroid_source must be supplied")
    product = _load_array_product(resolve_external_asset(manifest))
    required = {"observed_tangent_arcsec", "predicted_tangent_arcsec", "row_id"}
    missing = sorted(required - set(product))
    if missing:
        raise UnavailableEvidence(
            "supplied-centroid product missing arrays: " + ", ".join(missing)
        )
    return product


def run_supplied_source_adapter(
    peer: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("family") != "deblend-v1":
        raise UnavailableEvidence("supplied-source asset has the wrong family")
    identity = manifest.get("identity", {})
    if identity.get("peer") != peer:
        raise UnavailableEvidence(
            f"supplied-source peer identity mismatch: expected {peer!r}"
        )
    if identity.get("detection") is not False:
        raise UnavailableEvidence("supplied-source contract must disable detection")
    if identity.get("source_selection") != "supplied":
        raise UnavailableEvidence("source_selection must be supplied")
    product = _load_array_product(resolve_external_asset(manifest))
    required = {"component_id", "flux", "lon_deg", "lat_deg", "covariance"}
    missing = sorted(required - set(product))
    if missing:
        raise UnavailableEvidence(
            "supplied-source product missing arrays: " + ", ".join(missing)
        )
    return product


def run_car_sht_product_adapter(
    peer: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("family") != "sphere-v1":
        raise UnavailableEvidence("CAR SHT asset has the wrong family")
    identity = manifest.get("identity", {})
    if identity.get("peer") != peer:
        raise UnavailableEvidence(f"CAR SHT peer identity mismatch: expected {peer!r}")
    if identity.get("geometry") != "CAR":
        raise UnavailableEvidence("CAR SHT product must declare CAR geometry")
    product = _load_array_product(resolve_external_asset(manifest))
    required = {"torchsky_values", "peer_values", "runtime_seconds", "peak_bytes"}
    missing = sorted(required - set(product))
    if missing:
        raise UnavailableEvidence("CAR SHT product missing arrays: " + ", ".join(missing))
    return product


def run_jwst_cube_build_adapter(
    manifest: Mapping[str, Any], *, weighting: str
) -> ComparatorProducts:
    association = resolve_external_association(manifest)
    try:
        from jwst.cube_build import CubeBuildStep
    except Exception as exc:
        raise UnavailableEvidence(f"jwst.cube_build unavailable: {exc}") from exc
    try:
        container = CubeBuildStep.call(
            str(association),
            weighting=weighting,
            save_results=False,
        )
    except Exception as exc:
        raise UnavailableEvidence(f"jwst.cube_build execution failed: {exc}") from exc
    if not container:
        raise UnavailableEvidence("jwst.cube_build returned no cube models")
    model = container[0]
    variances = {
        name: _numpy(getattr(model, name))
        for name in ("var_poisson", "var_rnoise", "var_flat")
        if getattr(model, name, None) is not None
    }
    weight = getattr(model, "weightmap", None)
    return ComparatorProducts(
        science=_numpy(model.data),
        weight=None if weight is None else _numpy(weight),
        variance_components=variances,
        dq=None if getattr(model, "dq", None) is None else _numpy(model.dq),
        nexp=None if weight is None else (_numpy(weight) > 0).astype(np.int32),
        metadata={
            "adapter": "jwst.cube_build.CubeBuildStep",
            "instrument": manifest["instrument"],
            "weighting": weighting,
            "crds_context": manifest["crds_context"],
            "association_sha256": manifest["association_sha256"],
        },
    )


__all__ = [
    "ComparatorProducts",
    "UnavailableEvidence",
    "adapt_astrodrizzle_product",
    "adapt_stcal_resample_output",
    "load_external_association_manifest",
    "load_external_asset_manifest",
    "resolve_external_association",
    "resolve_external_asset",
    "run_astrodrizzle_adapter",
    "run_jwst_cube_build_adapter",
    "run_car_sht_product_adapter",
    "run_reproject_adaptive_adapter",
    "run_supplied_centroid_adapter",
    "run_supplied_source_adapter",
    "run_stcal_outlier_adapter",
]
