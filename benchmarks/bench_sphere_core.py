#!/usr/bin/env python3
"""Benchmark torchsky spherical-core primitives."""

from __future__ import annotations

from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402


import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from torchsky.sphere.core import pairwise_angular_distance, sample_multiband_healpix  # noqa: E402
from torchsky.sphere.geom import query_ellipse  # noqa: E402
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

try:
    import healpy
except Exception:  # pragma: no cover - optional comparator
    healpy = None


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if choice == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(choice)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _bench(
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


def _sample_lonlat(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return lon, lat


def _append_moc_rows(
    results: list[dict[str, Any]],
    *,
    lon_t: torch.Tensor,
    lat_t: torch.Tensor,
    n_points: int,
    runs: int,
    max_order: int,
    sampling_profile: str,
    seed: int,
) -> None:
    from torchsky.sphere.moc import BMOC as TorchBMOC
    from torchsky.sphere.moc import MOC as TorchMOC

    try:
        import astropy.units as u
        from mocpy import MOC as MocpyMOC
    except Exception:
        u = None
        MocpyMOC = None

    moc_lon = lon_t[: min(50_000, n_points)].cpu()
    moc_lat = lat_t[: min(50_000, n_points)].cpu()
    cpu = torch.device("cpu")

    def _from_circle() -> Any:
        return TorchMOC.from_circle(20.0, -10.0, 5.0, max_order=max_order)

    t_cone = _bench(_from_circle, runs=runs, device=cpu)
    moc = _from_circle()
    results.append(
        {
            "operation": "moc_from_circle",
            "library": "torchsky",
            "max_order": int(max_order),
            "time_s": t_cone,
        }
    )
    if MocpyMOC is not None and hasattr(MocpyMOC, "from_cone"):
        t_ref_cone = _bench(
            lambda: MocpyMOC.from_cone(
                lon=20.0 * u.deg,
                lat=-10.0 * u.deg,
                radius=5.0 * u.deg,
                max_depth=max_order,
            ),
            runs=runs,
            device=None,
        )
        results.append(
            {
                "operation": "moc_from_cone",
                "library": "mocpy",
                "max_order": int(max_order),
                "time_s": t_ref_cone,
                "ratio_torchsky_vs_mocpy": t_ref_cone / t_cone,
            }
        )

    t_ts = _bench(
        lambda: moc.contains(moc_lon, moc_lat),
        runs=runs,
        device=cpu,
    )
    results.append(
        {
            "operation": "moc_contains",
            "library": "torchsky",
            "n_points": int(moc_lon.numel()),
            "contains_s": t_ts,
            "time_s": t_ts,
        }
    )
    if MocpyMOC is not None and hasattr(MocpyMOC, "from_cone"):
        ref = MocpyMOC.from_cone(
            lon=20.0 * u.deg, lat=-10.0 * u.deg, radius=5.0 * u.deg, max_depth=max_order
        )
        lon_u = moc_lon.numpy() * u.deg
        lat_u = moc_lat.numpy() * u.deg
        contains_fn = getattr(ref, "contains_lonlat", getattr(ref, "contains", None))
        if contains_fn is not None:
            t_ref = _bench(lambda: contains_fn(lon_u, lat_u), runs=runs, device=None)
            results.append(
                {
                    "operation": "moc_contains",
                    "library": "mocpy",
                    "n_points": int(moc_lon.numel()),
                    "contains_s": t_ref,
                    "time_s": t_ref,
                    "ratio_torchsky_vs_mocpy": t_ref / t_ts,
                }
            )

    a = TorchMOC.from_circle(0.0, 0.0, 2.0, max_order=max_order)
    b = TorchMOC.from_circle(3.0, 0.0, 2.0, max_order=max_order)
    t_union = _bench(lambda: a.union(b), runs=runs, device=cpu)
    results.append(
        {
            "operation": "moc_union",
            "library": "torchsky",
            "max_order": int(max_order),
            "time_s": t_union,
        }
    )
    if MocpyMOC is not None and hasattr(MocpyMOC, "from_cone"):
        ra = MocpyMOC.from_cone(
            lon=0.0 * u.deg, lat=0.0 * u.deg, radius=2.0 * u.deg, max_depth=max_order
        )
        rb = MocpyMOC.from_cone(
            lon=3.0 * u.deg, lat=0.0 * u.deg, radius=2.0 * u.deg, max_depth=max_order
        )
        t_ref_union = _bench(lambda: ra.union(rb), runs=runs, device=None)
        results.append(
            {
                "operation": "moc_union",
                "library": "mocpy",
                "max_order": int(max_order),
                "time_s": t_ref_union,
                "ratio_torchsky_vs_mocpy": t_ref_union / t_union,
            }
        )

    cell_counts, sample_counts = sampling_sizes(sampling_profile)
    for n_cells in cell_counts:
        uniq, group_ids = _mixed_order_uniq(
            n_cells, max_order=max_order, device=lon_t.device
        )
        moc_case = TorchMOC(uniq)
        moc_groups = [TorchMOC(uniq[group_ids == index]) for index in range(3)]
        _append_sampling_case_rows(
            results,
            subject=moc_case,
            subject_name="moc",
            groups=moc_groups,
            n_cells=int(moc_case.uniq.numel()),
            sample_counts=sample_counts,
            runs=runs,
            seed=seed + n_cells,
            mixed_orders=True,
        )
        flags = (torch.arange(uniq.numel(), dtype=torch.uint8) % 2).to(uniq.device)
        bmoc_case = TorchBMOC(uniq, flags)
        bmoc_groups = [
            TorchBMOC(uniq[group_ids == index], flags[group_ids == index])
            for index in range(3)
        ]
        _append_sampling_case_rows(
            results,
            subject=bmoc_case,
            subject_name="bmoc",
            groups=bmoc_groups,
            n_cells=int(bmoc_case.uniq.numel()),
            sample_counts=sample_counts,
            runs=runs,
            seed=seed + 10_000_000 + n_cells,
            mixed_orders=True,
        )
    if sample_counts:
        from torchsky.sphere.moc import nest2uniq

        full_uniq = nest2uniq(
            1, torch.arange(12, dtype=torch.int64, device=lon_t.device)
        )
        full_groups = [TorchMOC(full_uniq[index : index + 1]) for index in range(12)]
        full_moc = TorchMOC(full_uniq)
        _append_sampling_case_rows(
            results,
            subject=full_moc,
            subject_name="moc",
            groups=full_groups,
            n_cells=12,
            sample_counts=sample_counts,
            runs=runs,
            seed=seed + 20_000_000,
            mixed_orders=False,
            all_sky=True,
        )
        full_bmoc = TorchBMOC(
            full_uniq, torch.ones(12, dtype=torch.uint8, device=lon_t.device)
        )
        full_bmoc_groups = [
            TorchBMOC(
                full_uniq[index : index + 1],
                torch.ones(1, dtype=torch.uint8, device=lon_t.device),
            )
            for index in range(12)
        ]
        _append_sampling_case_rows(
            results,
            subject=full_bmoc,
            subject_name="bmoc",
            groups=full_bmoc_groups,
            n_cells=12,
            sample_counts=sample_counts,
            runs=runs,
            seed=seed + 30_000_000,
            mixed_orders=False,
            all_sky=True,
        )


def _mixed_order_uniq(
    n_cells: int, *, max_order: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    from torchsky.sphere.moc import nest2uniq

    counts = [n_cells // 3, n_cells // 3, n_cells - 2 * (n_cells // 3)]
    largest = max(counts)
    required_top_order = int(np.ceil(np.log(max(1, largest)) / np.log(4.0))) + 3
    top_order = max(int(max_order), required_top_order)
    uniq_parts: list[torch.Tensor] = []
    group_parts: list[torch.Tensor] = []
    for group, (count, order) in enumerate(
        zip(counts, (top_order - 2, top_order - 1, top_order), strict=True)
    ):
        nside = 1 << order
        pixels = group * nside * nside + 4 * torch.arange(
            count, dtype=torch.int64, device=device
        )
        uniq_parts.append(nest2uniq(nside, pixels))
        group_parts.append(
            torch.full((count,), group, dtype=torch.int64, device=device)
        )
    return torch.cat(uniq_parts), torch.cat(group_parts)


def _append_sampling_case_rows(
    results: list[dict[str, Any]],
    *,
    subject: Any,
    subject_name: str,
    groups: list[Any],
    n_cells: int,
    sample_counts: tuple[int, ...],
    runs: int,
    seed: int,
    mixed_orders: bool,
    all_sky: bool = False,
) -> None:
    device = subject.uniq.device
    prepared_factory, prepared_api, unavailable_reason = discover_prepared_factory(
        subject, device=device
    )
    expected = torch.tensor([group.area for group in groups], dtype=torch.float64)
    expected /= expected.sum()

    def categories(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        category = torch.full(lon.shape, -1, dtype=torch.int64, device=lon.device)
        for index, group in enumerate(groups):
            category[group.contains(lon, lat)] = index
        return category

    for index, n_samples in enumerate(sample_counts):
        case = RandomSamplingCase(
            subject=subject_name,
            n_cells=n_cells,
            n_samples=n_samples,
            device=device,
            one_shot=lambda generator, n_samples=n_samples: subject.sample_randoms(
                n_samples, generator=generator, device=device
            ),
            contains=lambda lon, lat: subject.contains(lon, lat),
            categories=categories,
            expected_probabilities=expected,
            ordering="NUNIQ",
            weighted=False,
            prepared_factory=prepared_factory,
            prepared_api=prepared_api,
            prepared_unavailable_reason=unavailable_reason,
            all_sky=all_sky,
        )
        row = benchmark_random_sampling_case(case, runs=runs, seed=seed + index)
        row["mixed_orders"] = mixed_orders
        results.append(row)


def _write_results(
    results: list[dict[str, Any]],
    *,
    json_out: Path,
    sampling_profile: str,
    device: torch.device,
) -> None:
    for row in results:
        print(", ".join(f"{key}={value}" for key, value in row.items()))
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sampling_rows = [row for row in results if row.get("operation") == "random_sampling"]
    if sampling_rows:
        evidence_path = json_out.with_name(f"{json_out.stem}_evidence{json_out.suffix}")
        write_payload(
            evidence_path,
            build_payload(
                evaluation="bench_sphere_core",
                workstream="O4",
                profile=sampling_profile,
                device=device,
                rows=sampling_rows,
            ),
        )
    print(f"\nJSON: {json_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--n-points", type=int, default=100_000)
    parser.add_argument("--n-bands", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu"
    )
    parser.add_argument("--moc-only", action="store_true")
    parser.add_argument("--moc-max-order", type=int, default=8)
    parser.add_argument(
        "--sampling-profile", choices=["off", "quick", "full"], default="quick"
    )
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "sphere_core.json"
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    f_dtype = torch.float32 if device.type == "mps" else torch.float64

    lon_np, lat_np = _sample_lonlat(args.n_points, args.seed)
    lon_t = torch.from_numpy(lon_np).to(device=device, dtype=f_dtype)
    lat_t = torch.from_numpy(lat_np).to(device=device, dtype=f_dtype)
    results: list[dict[str, Any]] = []

    if args.moc_only:
        _append_moc_rows(
            results,
            lon_t=lon_t,
            lat_t=lat_t,
            n_points=args.n_points,
            runs=args.runs,
            max_order=args.moc_max_order,
            sampling_profile=args.sampling_profile,
            seed=args.seed,
        )
        _write_results(
            results,
            json_out=args.json_out,
            sampling_profile=args.sampling_profile,
            device=device,
        )
        return 0

    npix = 12 * args.nside * args.nside
    rng = np.random.default_rng(args.seed + 1)
    cube_np = rng.normal(size=(args.n_bands, npix)).astype(np.float64)
    cube_t = torch.from_numpy(cube_np).to(device=device, dtype=f_dtype)

    t_bilinear = _bench(
        lambda: sample_multiband_healpix(
            cube_t,
            args.nside,
            lon_t,
            lat_t,
            interpolation="bilinear",
        ),
        runs=args.runs,
        device=device,
    )
    results.append(
        {
            "operation": "sample_multiband_bilinear",
            "device": device.type,
            "nside": args.nside,
            "n_points": args.n_points,
            "n_bands": args.n_bands,
            "m_samples_s": (args.n_points * args.n_bands) / t_bilinear / 1e6,
        }
    )

    t_nearest = _bench(
        lambda: sample_multiband_healpix(
            cube_t,
            args.nside,
            lon_t,
            lat_t,
            interpolation="nearest",
        ),
        runs=args.runs,
        device=device,
    )
    results.append(
        {
            "operation": "sample_multiband_nearest",
            "device": device.type,
            "nside": args.nside,
            "n_points": args.n_points,
            "n_bands": args.n_bands,
            "m_samples_s": (args.n_points * args.n_bands) / t_nearest / 1e6,
        }
    )

    # Pairwise is O(N^2); keep N small for benchmark sanity.
    pair_n = min(8_000, args.n_points)
    lon_small = lon_t[:pair_n]
    lat_small = lat_t[:pair_n]
    t_pairwise = _bench(
        lambda: pairwise_angular_distance(lon_small, lat_small),
        runs=max(2, args.runs // 2),
        device=device,
    )
    results.append(
        {
            "operation": "pairwise_angular_distance",
            "device": device.type,
            "n_points": pair_n,
            "m_pairwise_s": (pair_n * pair_n) / t_pairwise / 1e6,
        }
    )

    t_query = _bench(
        lambda: query_ellipse(
            args.nside, 125.0, -30.0, 7.0, 2.0, pa_deg=15.0, nest=False
        ),
        runs=args.runs,
        device=torch.device("cpu"),
    )
    results.append(
        {
            "operation": "query_ellipse",
            "device": "cpu",
            "nside": args.nside,
            "queries_s": 1.0 / t_query,
        }
    )

    if healpy is not None and device.type == "cpu":

        def _healpy_loop_interp() -> None:
            for i in range(args.n_bands):
                healpy.get_interp_val(
                    cube_np[i], lon_np, lat_np, lonlat=True, nest=False
                )

        t_hp = _bench(_healpy_loop_interp, runs=max(2, args.runs // 2), device=None)
        results.append(
            {
                "operation": "healpy_loop_interp_baseline",
                "device": "cpu",
                "nside": args.nside,
                "n_points": args.n_points,
                "n_bands": args.n_bands,
                "m_samples_s": (args.n_points * args.n_bands) / t_hp / 1e6,
                "ratio_torchsky_bilinear_vs_healpy_loop": t_hp / t_bilinear,
            }
        )

    _append_moc_rows(
        results,
        lon_t=lon_t,
        lat_t=lat_t,
        n_points=args.n_points,
        runs=args.runs,
        max_order=args.moc_max_order,
        sampling_profile=args.sampling_profile,
        seed=args.seed,
    )

    _write_results(
        results,
        json_out=args.json_out,
        sampling_profile=args.sampling_profile,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
