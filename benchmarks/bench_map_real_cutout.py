#!/usr/bin/env python3
"""Map reprojection parity on a real CFHT MegaPipe cutout.

The fixture (``benchmarks/fixtures/real_map``) is a contiguous 256x256 cutout
of the actual 2 GB CFHT MegaPipe g-band mosaic with its original TAN WCS,
derived by ``scripts/derive_real_map_fixture.py``. This benchmark reprojects
that real image onto a rotated/rescaled TAN grid with torchsky ``map.resample``
and with astropy ``reproject``, then compares the resampled pixels on the
interior of the shared footprint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "real_map" / "cfht_g_cutout_256.npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-side", type=int, default=300)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--scale", type=float, default=1.2)
    parser.add_argument("--edge-erosion-px", type=int, default=2)
    parser.add_argument("--max-relative-error", type=float, default=None)
    parser.add_argument("--max-p99-relative-error", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _target_header(
    header: dict[str, Any], side: int, rotation_deg: float, scale: float
) -> dict[str, Any]:
    angle = math.radians(rotation_deg)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    cd = np.array(
        [
            [header["CD1_1"], header["CD1_2"]],
            [header["CD2_1"], header["CD2_2"]],
        ],
        dtype=np.float64,
    )
    target_cd = rotation @ cd * scale
    target = {
        key: value
        for key, value in header.items()
        if key
        not in {
            "NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        }
    }
    center = (side + 1) / 2.0
    target.update(
        {
            "NAXIS1": int(side),
            "NAXIS2": int(side),
            "CRPIX1": center,
            "CRPIX2": center,
            "CD1_1": float(target_cd[0, 0]),
            "CD1_2": float(target_cd[0, 1]),
            "CD2_1": float(target_cd[1, 0]),
            "CD2_2": float(target_cd[1, 1]),
        }
    )
    return target


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(iterations):
        previous = out.copy()
        out[1:, :] &= previous[:-1, :]
        out[:-1, :] &= previous[1:, :]
        out[:, 1:] &= previous[:, :-1]
        out[:, :-1] &= previous[:, 1:]
    return out


def main() -> int:
    args = _parse_args()
    try:
        import torch
        from astropy.wcs import WCS
        from reproject import reproject_interp
        from torchsky.map import resample
        from torchsky.map.observation import ImageObservation
    except ImportError as exc:  # pragma: no cover - benchmark environment
        raise SystemExit(f"bench_map_real_cutout requires map dependencies: {exc}")

    data = np.load(FIXTURE)
    image = data["image"].astype(np.float64)
    header = json.loads(str(data["header_json"]))
    target_header = _target_header(
        header, args.target_side, args.rotation_deg, args.scale
    )
    observation = ImageObservation(
        data=torch.from_numpy(image), header=dict(header)
    )
    t0 = time.perf_counter()
    result = resample(observation, target=target_header, kernel="bilinear")
    torch_ms = (time.perf_counter() - t0) * 1e3
    torch_image = result.image.detach().cpu().numpy()

    t0 = time.perf_counter()
    reference, footprint = reproject_interp(
        (image, WCS(header)),
        WCS(target_header),
        shape_out=(args.target_side, args.target_side),
    )
    reproject_ms = (time.perf_counter() - t0) * 1e3

    covered = (footprint > 0.5) & np.isfinite(reference)
    interior = _erode(covered, args.edge_erosion_px) & np.isfinite(torch_image)
    differences = np.abs(torch_image[interior] - reference[interior])
    scale = max(float(np.max(np.abs(reference[interior]))), 1.0e-30)
    max_abs = float(np.max(differences))
    p99_abs = float(np.percentile(differences, 99))
    median_abs = float(np.median(differences))
    max_rel = max_abs / scale
    p99_rel = p99_abs / scale
    coverage_fraction_diff = abs(
        float(np.mean(torch_image != 0.0)) - float(np.mean(covered))
    )
    rows = [
        {
            "case": "resample_bilinear_real",
            "n_points": int(interior.sum()),
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "p99_rel_diff": p99_rel,
            "median_abs_diff": median_abs,
            "coverage_fraction_diff": coverage_fraction_diff,
            "torch_ms": torch_ms,
            "reproject_ms": reproject_ms,
            "status": "OK",
        }
    ]
    print(
        f"resample_bilinear_real: n={rows[0]['n_points']} "
        f"max_rel={max_rel:.3e} p99_rel={p99_rel:.3e} "
        f"coverage_diff={coverage_fraction_diff:.3e} "
        f"torch={torch_ms:.2f}ms reproject={reproject_ms:.2f}ms"
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    failures: list[str] = []
    if args.max_relative_error is not None and max_rel > args.max_relative_error:
        failures.append(f"max relative diff {max_rel:.3e}")
    if (
        args.max_p99_relative_error is not None
        and p99_rel > args.max_p99_relative_error
    ):
        failures.append(f"p99 relative diff {p99_rel:.3e}")
    if failures:
        for failure in failures:
            print(f"Real-map threshold exceeded: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
