#!/usr/bin/env python3
"""CFHT MegaCam MEF mosaic benchmark: torchsky vs SWarp on full 36-CCD exposures."""

from __future__ import annotations

import sys

import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torchfits

ROOT = Path(__file__).resolve().parents[1]
from torchsky_harness.benchmarks import bench_map_compat  # noqa: E402

from torchsky.map.observation import ImageObservation  # noqa: E402
from torchsky.map.product import plan_map_product, stack_product  # noqa: E402

DATA_DIR = ROOT / "data" / "cfht" / "MegaPipe.028.181" / "g"

N_CCD = 36
CCD_H = 4644
CCD_W = 2112


def discover_exposures(data_dir: Path, max_exposures: int) -> list[Path]:
    files = sorted(data_dir.glob("*.fits*"))
    viable = [f for f in files if f.stat().st_size > 10_000_000]
    return viable[:max_exposures]


def exposure_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"


def estimate_background_offsets(
    exposures: list[Path], n_ccd: int
) -> dict[tuple[str, int], float]:
    offsets: dict[tuple[str, int], float] = {}
    for exp_path in exposures:
        for ext_idx in range(1, n_ccd + 1):
            img = torchfits.read(str(exp_path), hdu=int(ext_idx))
            finite = img[torch.isfinite(img)]
            if finite.numel() == 0:
                raise RuntimeError(f"no finite pixels in {exp_path} HDU {ext_idx}")
            offsets[(str(exp_path), ext_idx)] = float(finite.median())
    return offsets


def comparison_metrics(
    torchsky_data: np.ndarray,
    torchsky_weight: np.ndarray,
    swarp_data: np.ndarray,
    swarp_weight: np.ndarray,
) -> dict[str, float | int]:
    mask = (
        np.isfinite(torchsky_data)
        & np.isfinite(swarp_data)
        & np.isfinite(torchsky_weight)
        & np.isfinite(swarp_weight)
        & (torchsky_weight > 0)
        & (swarp_weight > 0)
    )
    if not mask.any():
        raise RuntimeError("Torchsky and SWarp outputs have no common finite support")
    residual = torchsky_data[mask] - swarp_data[mask]
    return {
        "common_pixels": int(mask.sum()),
        "max_abs_difference": float(np.abs(residual).max()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "median_difference": float(np.median(residual)),
        "mean_difference": float(np.mean(residual)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--band", choices=("u", "r"), required=True)
    parser.add_argument("--exposures", type=int, default=10)
    parser.add_argument("--ccds", type=int, default=36)
    parser.add_argument("--skip-swarp", action="store_true")
    parser.add_argument("--require-swarp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    if args.require_swarp and args.skip_swarp:
        print("--require-swarp and --skip-swarp are mutually exclusive")
        return 1
    if args.require_swarp and bench_map_compat._find_swarp_binary() is None:
        print("SWarp is required but unavailable")
        return 1
    if args.reference is not None and not args.reference.is_file():
        print(f"Reference not found: {args.reference}")
        return 1

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"Data dir not found: {data_dir}")
        return 1

    exposures = discover_exposures(data_dir, max_exposures=args.exposures)
    if not exposures:
        print(f"No exposures found in {data_dir}")
        return 1

    n_exp = len(exposures)
    if n_exp != args.exposures:
        print(f"Expected {args.exposures} exposures, found {n_exp}")
        return 1
    n_obs_total = n_exp * args.ccds
    src_mpix = n_obs_total * CCD_W * CCD_H / 1e6

    print("=" * 70)
    print("CFHT MegaCam MEF Mosaic Benchmark (full pipeline)")
    print("=" * 70)
    print(f"  Exposures:    {n_exp}")
    print(f"  CCDs/exp:     {args.ccds}")
    print(f"  Total obs:    {n_obs_total}")
    print(f"  Source pix:   {src_mpix:.0f}M  ({CCD_W}x{CCD_H} per CCD)")
    print(f"  Device:       {args.device}")
    print(f"  Band:         {args.band}")
    for p in exposures:
        print(f"    {p.name}  {p.stat().st_size / 1e6:.0f}MB")

    # ── Step 1: estimate per-CCD background ────────────────────────
    print("\n── Estimating per-CCD background (median) ──")
    t0 = time.perf_counter()
    bg_offsets = estimate_background_offsets(exposures, args.ccds)
    bg_elapsed = time.perf_counter() - t0
    bg_values = list(bg_offsets.values())
    print(f"  Estimated {len(bg_offsets)} CCD backgrounds in {bg_elapsed * 1000:.0f}ms")
    print(
        f"    min={min(bg_values):.1f}  max={max(bg_values):.1f}  "
        f"median={float(np.median(bg_values)):.1f}"
    )

    # ── Step 2: Build observations with background offsets ──────────
    print("\n── Building observations w/ background_offset ──")
    t0 = time.perf_counter()
    obs_list = []
    for exp_path in exposures:
        for ext_idx in range(1, args.ccds + 1):
            obs_list.append(
                ImageObservation(
                    str(exp_path),
                    hdu=int(ext_idx),
                    background_offset=bg_offsets[(str(exp_path), ext_idx)],
                )
            )
    print(
        f"  Created {len(obs_list)} observations in {(time.perf_counter() - t0) * 1000:.0f}ms"
    )

    # ── Step 3: Plan product from union footprint ─────────────────
    print("\n── Planning map product (union footprint) ──")
    t0 = time.perf_counter()
    product = plan_map_product(
        obs_list,
        union_footprint=True,
        max_footprint_pixels=None,
    )
    nx, ny = product.shape[1], product.shape[0]
    mpix_tgt = nx * ny / 1e6
    plan_time = time.perf_counter() - t0
    print(f"  Target: {nx} x {ny}  ({mpix_tgt:.1f}M pix)")
    print(f"  Plan:   {plan_time * 1000:.0f}ms")

    # ── Step 4: torchsky coadd ────────────────────────────────────
    print("\n── torchsky coadd (stack_product, bilinear) ──")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    result = stack_product(
        obs_list,
        product,
        resampling_kernel="bilinear",
        device=args.device,
    )
    ts_elapsed = time.perf_counter() - t0

    print(f"  Elapsed:    {exposure_str(ts_elapsed)}")
    print(
        f"  Output:     {list(result.data.shape)}  "
        f"range=[{result.data.min():.2f}, {result.data.max():.2f}]"
    )
    print(f"  Weight:     range=[{result.weight.min():.2f}, {result.weight.max():.2f}]")
    print(f"  Throughput: {mpix_tgt / ts_elapsed:.1f} mpix/s (tgt)")
    print(f"              {src_mpix / ts_elapsed:.1f} mpix/s (src)")

    import psutil

    try:
        proc = psutil.Process()
        print(f"  RSS:        {proc.memory_info().rss / 1e9:.2f} GB")
    except ImportError:
        pass
    n_obs = len(obs_list)
    print(f"  Per obs:    {ts_elapsed * 1000 / n_obs:.1f}ms")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile = (
        args.reference.name.removesuffix(".fits").rsplit(".", 1)[0]
        if args.reference is not None
        else "CFIS.unidentified"
    )
    torchsky_output = args.output_dir / f"{tile}.{args.band}.torchsky.fits"
    torchfits.write(
        str(torchsky_output),
        result.data.cpu(),
        header=dict(product.target_header),
        overwrite=True,
    )
    payload = {
        "schema_version": 1,
        "campaign": "cfht-union-pitcairn-bounded-smoke",
        "tile": tile,
        "band": args.band,
        "status": "smoke-complete",
        "claim": None,
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size} for path in exposures
        ],
        "reference": str(args.reference.resolve()) if args.reference else None,
        "target": {"width": nx, "height": ny, "pixels": nx * ny},
        "torchsky": {
            "elapsed_seconds": ts_elapsed,
            "planning_seconds": plan_time,
            "background_seconds": bg_elapsed,
            "output": str(torchsky_output),
            "combine": "weighted_mean",
            "resampling": "bilinear",
        },
        "swarp": None,
        "comparison": None,
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchsky_import": str(Path(sys.modules["torchsky"].__file__).resolve()),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "PYTHONPATH_set": bool(os.environ.get("PYTHONPATH")),
        },
    }

    # ── Step 5: SWarp coadd ─────────────────────────────────────────
    if not args.skip_swarp and bench_map_compat._find_swarp_binary() is not None:
        swarp_bin = bench_map_compat._find_swarp_binary()
        swarp_env = bench_map_compat._external_tool_env()

        # Derive SWarp parameters from the planned product WCS
        sw_crval1 = float(product.target_header["CRVAL1"])
        sw_crval2 = float(product.target_header["CRVAL2"])
        sw_pixscale = (
            abs(
                float(
                    product.target_header.get(
                        "CD1_1", product.target_header.get("CDELT1", 5e-5)
                    )
                )
            )
            * 3600.0
        )
        sw_nx = product.shape[1]
        sw_ny = product.shape[0]
        proj = "TAN"

        with tempfile.TemporaryDirectory(prefix="ts_swarp_") as tmpdir:
            tmp = Path(tmpdir)

            # Decompress
            decompressed: list[str] = []
            print(f"\n── SWarp: decompressing {n_exp} fpack files ──")
            t0 = time.perf_counter()
            for exp_path in exposures:
                out = tmp / f"{exp_path.stem}.fits"
                r = subprocess.run(
                    ["funpack", "-O", str(out), str(exp_path)],
                    capture_output=True,
                    text=True,
                    env=swarp_env,
                )
                if r.returncode != 0:
                    print(f"  funpack failed: {r.stderr[:200]}")
                    return 1
                decompressed.append(str(out))
            decomp_time = time.perf_counter() - t0
            print(f"  Decompressed {n_exp} files in {decomp_time:.1f}s")

            def _run_swarp() -> tuple[
                float, np.ndarray | None, np.ndarray | None, Path
            ]:
                cfg = tmp / "swarp.conf"
                imgout = args.output_dir / f"{tile}.{args.band}.swarp.fits"
                weightout = (
                    args.output_dir / f"{tile}.{args.band}.swarp.weight.fits"
                )
                lines = [
                    f"IMAGEOUT_NAME          {imgout}",
                    f"WEIGHTOUT_NAME         {weightout}",
                    "WEIGHT_TYPE            NONE",
                    "COMBINE                Y",
                    "COMBINE_TYPE           WEIGHTED",
                    "CELESTIAL_TYPE         EQUATORIAL",
                    f"PROJECTION_TYPE        {proj}",
                    "PROJECTION_ERR         0.0",
                    "CENTER_TYPE            MANUAL",
                    f"CENTER                 {sw_crval1:.10f}, {sw_crval2:.10f}",
                    "PIXELSCALE_TYPE        MANUAL",
                    f"PIXEL_SCALE            {sw_pixscale:.10f}",
                    f"IMAGE_SIZE             {sw_nx},{sw_ny}",
                    "RESAMPLE               Y",
                    "RESAMPLING_TYPE        BILINEAR",
                    "INTERPOLATE            N",
                    "FSCALASTRO_TYPE        NONE",
                    "SUBTRACT_BACK          Y",
                    "WRITE_XML              N",
                    "VERBOSE_TYPE           QUIET",
                    "NTHREADS               0",
                    "DELETE_TMPFILES        Y",
                ]
                cfg.write_text("\n".join(lines) + "\n")
                cmd = [str(swarp_bin), "-c", str(cfg), *decompressed]
                t0 = time.perf_counter()
                r = subprocess.run(cmd, capture_output=True, text=True, env=swarp_env)
                elapsed = time.perf_counter() - t0
                if r.returncode != 0:
                    print(f"  SWarp failed (exit {r.returncode})")
                    for line in r.stderr.splitlines()[-5:]:
                        print(f"    > {line}")
                    return elapsed, None, None, imgout
                if not imgout.exists():
                    print("  SWarp produced no output")
                    return elapsed, None, None, imgout
                data = (
                    torchfits.read(str(imgout)).numpy().astype(np.float32, copy=False)
                )
                weight = (
                    torchfits.read(str(weightout))
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                return elapsed, data, weight, imgout

            print("\n── SWarp coadd (matched weighted mean, background subtraction) ──")
            sw_elapsed, sw_data, sw_weight, swarp_output = _run_swarp()
            if sw_data is None or sw_weight is None:
                return 1
            print(f"  Resample/combine: {exposure_str(sw_elapsed)}")
            print(f"  Decompression:    {exposure_str(decomp_time)}")
            print(
                f"  Output:     {list(sw_data.shape)}  "
                f"range=[{np.nanmin(sw_data):.2f}, {np.nanmax(sw_data):.2f}]"
            )
            torchsky_data = result.data.detach().cpu().numpy().squeeze()
            torchsky_weight = result.weight.detach().cpu().numpy().squeeze()
            if sw_data.shape != torchsky_data.shape:
                print(
                    f"SWarp shape {sw_data.shape} != Torchsky shape {torchsky_data.shape}"
                )
                return 1
            metrics = comparison_metrics(
                torchsky_data, torchsky_weight, sw_data, sw_weight
            )
            payload["swarp"] = {
                "elapsed_seconds": sw_elapsed,
                "decompression_seconds": decomp_time,
                "output": str(swarp_output),
                "combine": "weighted_mean",
                "resampling": "bilinear",
                "background_subtraction": True,
            }
            payload["comparison"] = metrics
            print(f"  Common support: {metrics['common_pixels']} pixels")
            print(f"  RMSE:           {metrics['rmse']:.6g}")
    else:
        print("\n── SWarp skipped ──")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    print(f"  Exposures: {n_exp}")
    print(f"  CCDs:      {args.ccds} per exp ({n_obs_total} total)")
    print(f"  Source:    {src_mpix:.0f}M pixels")
    print(f"  Target:    {nx} x {ny} ({mpix_tgt:.1f}M pixels)")
    print(
        f"  torchsky:  {exposure_str(ts_elapsed)} ({mpix_tgt / ts_elapsed:.1f} mpix/s)"
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  Result:    {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
