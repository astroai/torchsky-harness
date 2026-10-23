#!/usr/bin/env python3
"""Benchmark sparse HEALPix map primitives and optional perf gates."""

from __future__ import annotations

from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402


import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from torchsky.sphere.sparse import SparseHealpixMap  # noqa: E402
from torchsky.sphere import healpix as hp  # noqa: E402
from torchsky_harness.benchmarks._random_sampling import (  # noqa: E402
    RandomSamplingCase,
    benchmark_random_sampling_case,
    discover_prepared_factory,
    sampling_sizes,
)
from torchsky_harness.optimization_evidence import (  # noqa: E402
    build_payload,
    write_payload,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if choice == "mps":
        raise RuntimeError("sparse benches need float64; MPS is float32-only")
    return torch.device(choice)


def _time_interleaved(
    fn_a: Callable[[], Any],
    fn_b: Callable[[], Any],
    runs: int,
    sync_b: torch.device | None = None,
) -> tuple[float, float]:
    fn_a()
    fn_b()
    if sync_b is not None:
        _sync(sync_b)
    a_samples: list[float] = []
    b_samples: list[float] = []
    for i in range(runs):
        if i % 2 == 0:
            t0 = time.perf_counter()
            fn_a()
            a_samples.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            fn_b()
            if sync_b is not None:
                _sync(sync_b)
            b_samples.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            fn_b()
            if sync_b is not None:
                _sync(sync_b)
            b_samples.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            fn_a()
            a_samples.append(time.perf_counter() - t0)
    return float(np.median(a_samples)), float(np.median(b_samples))


def _time_many(
    fn: Callable[[], Any], runs: int, device: torch.device | None = None
) -> float:
    fn()
    if device is not None:
        _sync(device)
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if device is not None:
            _sync(device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _sample_lonlat(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=n)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))).to(
        torch.float64
    )
    return lon, lat


def _sampling_sparse_map(
    n_cells: int,
    *,
    minimum_nside: int,
    nest: bool,
    device: torch.device,
) -> SparseHealpixMap:
    required_nside = 1
    while 12 * required_nside * required_nside < n_cells:
        required_nside *= 2
    nside = max(int(minimum_nside), required_nside)
    pixels = torch.arange(n_cells, dtype=torch.int64, device=device)
    values = torch.linspace(0.25, 2.0, n_cells, dtype=torch.float64, device=device)
    return SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=nside,
        nest=nest,
        sentinel=float(hp.UNSEEN),
    )


def _append_random_sampling_rows(
    rows: list[dict[str, Any]],
    *,
    profile: str,
    minimum_nside: int,
    runs: int,
    seed: int,
    device: torch.device,
) -> None:
    cell_counts, sample_counts = sampling_sizes(profile)
    for n_cells in cell_counts:
        for nest in (False, True):
            sparse = _sampling_sparse_map(
                n_cells, minimum_nside=minimum_nside, nest=nest, device=device
            )
            sorted_pixels, sorted_order = torch.sort(sparse.pixels)
            bin_count = min(16, n_cells)

            def locate(lon: torch.Tensor, lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                pixels = hp.ang2pix(
                    sparse.nside, lon, lat, nest=sparse.nest, lonlat=True
                ).to(torch.int64)
                indices = torch.searchsorted(sorted_pixels, pixels)
                safe_indices = indices.clamp(max=max(0, sorted_pixels.numel() - 1))
                valid = (indices < sorted_pixels.numel()) & (
                    sorted_pixels[safe_indices] == pixels
                )
                return safe_indices, valid

            for weighted in (False, True):
                weights: str | None = "values" if weighted else None
                prepared_factory, prepared_api, unavailable_reason = (
                    discover_prepared_factory(sparse, weights=weights, device=device)
                )
                sorted_weights = (
                    sparse.values.index_select(0, sorted_order).to(torch.float64)
                    if weighted
                    else torch.ones(n_cells, dtype=torch.float64, device=device)
                )
                expected = torch.zeros(bin_count, dtype=torch.float64, device=device)
                expected.scatter_add_(
                    0,
                    torch.arange(n_cells, device=device) % bin_count,
                    sorted_weights,
                )
                expected /= expected.sum()

                def contains(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
                    return locate(lon, lat)[1]

                def categories(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
                    indices, valid = locate(lon, lat)
                    return torch.where(valid, indices % bin_count, -1)

                for sample_index, n_samples in enumerate(sample_counts):
                    case = RandomSamplingCase(
                        subject="sparse",
                        n_cells=n_cells,
                        n_samples=n_samples,
                        device=device,
                        one_shot=lambda generator, n_samples=n_samples, weights=weights: sparse.sample_randoms(
                            n_samples,
                            weights=weights,
                            generator=generator,
                            device=device,
                        ),
                        contains=contains,
                        categories=categories,
                        expected_probabilities=expected,
                        ordering="NESTED" if nest else "RING",
                        weighted=weighted,
                        prepared_factory=prepared_factory,
                        prepared_api=prepared_api,
                        prepared_unavailable_reason=unavailable_reason,
                    )
                    rows.append(
                        benchmark_random_sampling_case(
                            case,
                            runs=runs,
                            seed=seed
                            + n_cells
                            + 100_000 * int(nest)
                            + 10_000 * int(weighted)
                            + sample_index,
                        )
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--coverage-frac", type=float, default=0.1)
    parser.add_argument("--n-queries", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--sampling-profile", choices=["off", "quick", "full"], default="quick"
    )
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "sphere_sparse.json"
    )
    parser.add_argument("--min-ratio-sparse-udgrade-vs-dense", type=float, default=1.0)
    args = parser.parse_args()

    if not (0.0 < args.coverage_frac <= 1.0):
        raise ValueError("coverage-frac must be in (0, 1]")

    device = _resolve_device(args.device)
    nside = int(args.nside)
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(args.seed)

    dense = torch.full((npix,), float(hp.UNSEEN), dtype=torch.float64)
    n_keep = max(1, int(round(npix * args.coverage_frac)))
    keep = torch.from_numpy(rng.choice(npix, size=n_keep, replace=False)).to(
        torch.int64
    )
    dense[keep] = torch.from_numpy(rng.normal(size=n_keep)).to(torch.float64)

    sparse = SparseHealpixMap.from_dense(dense.to(device), nside=nside, nest=False)
    lon_cpu, lat_cpu = _sample_lonlat(int(args.n_queries), args.seed + 1)
    lon = lon_cpu.to(device)
    lat = lat_cpu.to(device)
    coverage_frac = float(len(sparse.pixels)) / float(npix)

    rows: list[dict[str, Any]] = []

    t_sparse_nearest = _time_many(
        lambda: sparse.interpolate(lon, lat, method="nearest"), args.runs, device
    )
    rows.append(
        {
            "operation": "sparse_interpolate_nearest",
            "nside": nside,
            "device": str(device),
            "coverage_frac": coverage_frac,
            "n_queries": int(args.n_queries),
            "qps": float(args.n_queries) / t_sparse_nearest,
        }
    )

    t_sparse_bilinear = _time_many(
        lambda: sparse.interpolate(lon, lat, method="bilinear"), args.runs, device
    )
    rows.append(
        {
            "operation": "sparse_interpolate_bilinear",
            "nside": nside,
            "device": str(device),
            "coverage_frac": coverage_frac,
            "n_queries": int(args.n_queries),
            "qps": float(args.n_queries) / t_sparse_bilinear,
        }
    )

    nside_out = max(1, nside // 2)
    dense_dev = dense.to(device)
    t_sparse_ud = _time_many(
        lambda: sparse.ud_grade(nside_out, pess=False), args.runs, device
    )
    t_dense_ud = _time_many(
        lambda: hp.ud_grade(
            dense_dev,
            nside_out,
            pess=False,
            badval=float(hp.UNSEEN),
            order_in="RING",
            order_out="RING",
        ),
        args.runs,
        device,
    )
    ratio_ud = t_dense_ud / t_sparse_ud
    rows.append(
        {
            "operation": "ud_grade_sparse_vs_dense",
            "nside_in": nside,
            "nside_out": nside_out,
            "device": str(device),
            "coverage_frac": coverage_frac,
            "sparse_s": t_sparse_ud,
            "dense_s": t_dense_ud,
            "ratio_sparse_vs_dense": ratio_ud,
        }
    )

    _append_random_sampling_rows(
        rows,
        profile=args.sampling_profile,
        minimum_nside=nside,
        runs=args.runs,
        seed=args.seed,
        device=device,
    )

    try:
        import healsparse as _healsparse
    except Exception:
        _healsparse = None
    if _healsparse is not None:
        nside_cov = max(1, nside // 16)
        hs = _healsparse.HealSparseMap.make_empty(nside_cov, nside, np.float64)
        pix_np = sparse.pixels.detach().cpu()
        val_np = sparse.values.detach().cpu().numpy()
        pix_nest = pix_np
        if not bool(sparse.nest):
            pix_nest = hp.ring2nest(nside, pix_np)
        hs.update_values_pix(pix_nest.cpu().numpy(), val_np, nest=True)
        pix_nest_t = pix_nest.to(dtype=torch.int64)
        sparse_cov = SparseHealpixMap.from_pixels(
            pix_nest_t,
            sparse.values,
            nside_sparse=nside,
            nest=True,
            nside_coverage=nside_cov,
            sentinel=float(sparse.fill_value),
        )
        n_q = int(lon.numel())
        q_lon = lon[:n_q]
        q_lat = lat[:n_q]
        q_nest = hp.ang2pix(nside, q_lon, q_lat, nest=True, lonlat=True)
        q_nest_np = q_nest.detach().cpu().numpy()
        t_hs_get, t_ts_get = _time_interleaved(
            lambda: hs.get_values_pix(q_nest_np),
            lambda: sparse_cov.get_values(q_nest),
            args.runs,
            sync_b=device,
        )
        rows.append(
            {
                "operation": "sparse_get_values",
                "library": "healsparse",
                "n_queries": n_q,
                "get_values_s": t_hs_get,
                "timing": "interleaved",
            }
        )
        rows.append(
            {
                "operation": "sparse_get_values",
                "library": "torchsky",
                "n_queries": n_q,
                "get_values_s": t_ts_get,
                "ratio_torchsky_vs_healsparse": t_hs_get / t_ts_get,
                "timing": "interleaved",
            }
        )
        nside_ud = max(1, nside // 2)
        t_hs_ud, t_ts_ud = _time_interleaved(
            lambda: hs.degrade(nside_ud),
            lambda: sparse_cov.ud_grade(nside_ud, pess=False),
            args.runs,
            sync_b=device,
        )
        rows.append(
            {
                "operation": "ud_grade",
                "library": "healsparse",
                "nside_in": nside,
                "nside_out": nside_ud,
                "ud_grade_s": t_hs_ud,
                "timing": "interleaved",
            }
        )
        rows.append(
            {
                "operation": "ud_grade",
                "library": "torchsky",
                "nside_in": nside,
                "nside_out": nside_ud,
                "ud_grade_s": t_ts_ud,
                "ratio_torchsky_vs_healsparse": t_hs_ud / t_ts_ud,
                "timing": "interleaved",
            }
        )

    for row in rows:
        row.setdefault("device", str(device))
        print(", ".join(f"{k}={v}" for k, v in row.items()))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    sampling_rows = [row for row in rows if row.get("operation") == "random_sampling"]
    if sampling_rows:
        evidence_path = args.json_out.with_name(
            f"{args.json_out.stem}_evidence{args.json_out.suffix}"
        )
        write_payload(
            evidence_path,
            build_payload(
                evaluation="bench_sphere_sparse",
                workstream="O4",
                profile=args.sampling_profile,
                device=device,
                rows=sampling_rows,
            ),
        )
    print(f"\nJSON: {args.json_out}")

    if ratio_ud < float(args.min_ratio_sparse_udgrade_vs_dense):
        print(
            "Sparse ud_grade ratio threshold exceeded: "
            f"{ratio_ud:.3f} < {float(args.min_ratio_sparse_udgrade_vs_dense):.3f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
