"""Measure CPU spin scheduling and ring FFT policy evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from torchsky_harness.benchmarks.config import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class Case:
    bucket: str
    nside: int
    lmax: int
    mmax: int
    spin: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.bucket}_n{self.nside}_l{self.lmax}_m{self.mmax}_s{self.spin}"
        )


DEFAULT_CASES = (
    Case("small", 16, 12, 6, 1),
    Case("small", 16, 12, 12, 2),
    Case("medium", 64, 48, 24, 1),
    Case("medium", 64, 48, 48, 2),
    Case("large", 128, 128, 64, 1),
    Case("large", 128, 128, 128, 2),
)
AVAILABLE_POLICIES = ("hybrid", "aten")
UNAVAILABLE_POLICIES = {
    "pocketfft": (
        "No runtime switch forces grouped and singleton rings through pocketfft; "
        "the installed subject only exposes hybrid or Python ATen selection."
    ),
    "accelerate": (
        "No runtime switch forces all rings through Accelerate; Accelerate is an "
        "internal macOS radix-2 leaf of the hybrid C++ policy."
    ),
}


def _parse_cases(spec: str | None) -> tuple[Case, ...]:
    if spec is None or not spec.strip():
        return DEFAULT_CASES
    cases: list[Case] = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) != 5:
            raise ValueError(
                "cases must use bucket:nside:lmax:mmax:spin comma-separated syntax"
            )
        bucket = parts[0]
        if bucket not in {"small", "medium", "large"}:
            raise ValueError(f"invalid bucket {bucket!r}")
        case = Case(bucket, *(int(value) for value in parts[1:]))
        if (
            case.nside <= 0
            or case.lmax < 0
            or case.mmax < 0
            or case.mmax > case.lmax
            or case.spin < 0
        ):
            raise ValueError(f"invalid case {item!r}")
        cases.append(case)
    return tuple(cases)


def _parse_csv_ints(spec: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value.strip()) for value in spec.split(",")))
    if not values or any(value <= 0 for value in values):
        raise ValueError("thread counts must be positive")
    return values


def _parse_csv_strings(spec: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(value.strip() for value in spec.split(",")))
    valid = set(AVAILABLE_POLICIES) | set(UNAVAILABLE_POLICIES)
    if not values or any(value not in valid for value in values):
        raise ValueError(f"policies must be selected from {sorted(valid)}")
    return values


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return _rss_bytes()


def _stats(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "samples_s": samples,
        "runs": len(samples),
        "median_s": median,
        "mean_s": float(np.mean(values)),
        "stdev_s": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "mad_s": mad,
        "p25_s": float(np.percentile(values, 25)),
        "p75_s": float(np.percentile(values, 75)),
        "iqr_s": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
    }


def _measure(
    fn: Callable[[], Any], *, warmups: int, runs: int, inner_loops: int = 1
) -> tuple[dict[str, Any], Any]:
    result = None
    for _ in range(warmups):
        for _ in range(inner_loops):
            result = fn()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(inner_loops):
            result = fn()
        samples.append((time.perf_counter() - start) / inner_loops)
    stats = _stats(samples)
    stats["inner_loops_per_sample"] = inner_loops
    return stats, result


def _inner_loops(case: Case) -> int:
    return {"small": 20, "medium": 3, "large": 1}[case.bucket]


def _rel_l2(left: Any, right: Any) -> float:
    import torch

    left_t = left if torch.is_tensor(left) else torch.as_tensor(np.asarray(left))
    right_t = right if torch.is_tensor(right) else torch.as_tensor(np.asarray(right))
    denominator = float(torch.linalg.vector_norm(right_t))
    numerator = float(torch.linalg.vector_norm(left_t - right_t))
    return numerator if denominator == 0.0 else numerator / denominator


def _random_spin_alms(
    lmax: int, mmax: int, spin: int, seed: int
) -> tuple[Any, Any]:
    import torch

    rng = np.random.default_rng(seed)
    count = sum(lmax - m + 1 for m in range(mmax + 1))
    a_e = np.zeros(count, dtype=np.complex128)
    a_b = np.zeros(count, dtype=np.complex128)
    offset = 0
    for m in range(mmax + 1):
        size = lmax - m + 1
        real_e = rng.normal(size=size)
        real_b = rng.normal(size=size)
        imag_e = np.zeros(size) if m == 0 else rng.normal(size=size)
        imag_b = np.zeros(size) if m == 0 else rng.normal(size=size)
        a_e[offset : offset + size] = real_e + 1j * imag_e
        a_b[offset : offset + size] = real_b + 1j * imag_b
        low = max(min(spin - m, size), 0)
        a_e[offset : offset + low] = 0.0
        a_b[offset : offset + low] = 0.0
        offset += size
    return torch.from_numpy(a_e), torch.from_numpy(a_b)


def _packed_weights(lmax: int, mmax: int) -> Any:
    import torch

    values: list[float] = []
    for m in range(mmax + 1):
        values.extend([1.0 if m == 0 else 2.0] * (lmax - m + 1))
    return torch.tensor(values, dtype=torch.float64)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_metadata(torch: Any, torchsky: Any, spectral: Any) -> dict[str, Any]:
    import importlib.metadata

    root = Path(torchsky.__file__).resolve().parent
    native = next(root.glob("_cpp_ext*.so"), None)
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        isa_leaf = "aarch64-neon-compile-time"
    elif machine in {"x86_64", "amd64"}:
        isa_leaf = "x86-compile-time-macros"
    else:
        isa_leaf = "portable-scalar-or-unknown"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torchsky_version": importlib.metadata.version("torchsky"),
        "torchsky_import_path": str(root),
        "spectral_sha256": _hash(Path(spectral.__file__).resolve()),
        "native_extension": None if native is None else str(native),
        "native_extension_sha256": None if native is None else _hash(native),
        "selected_isa": isa_leaf,
        "runtime_isa_dispatch": False,
    }


def _policy_env(policy: str) -> dict[str, str]:
    enabled = "1" if policy == "hybrid" else "0"
    return {
        "TORCHSKY_RING_FOURIER_CPP": enabled,
        "TORCHSKY_RING_FOURIER_MODES_CPP": enabled,
        "TORCHSKY_RING_FOURIER_SYNTH_CPP": enabled,
        "TORCHSKY_SPIN_MAP2ALM_RING_TORCH": "force",
        "TORCHSKY_SPIN_ALM2MAP_RING_TORCH": "force",
        "TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP": "1",
        "TORCHSKY_SPIN_ALM2MAP_RECURRENCE_CPP": "1",
        "TORCHSKY_SPIN_RING_FINALIZE_CPP": "1",
    }


def _fft_evidence(
    case: Case, policy: str, warmups: int, runs: int, seed: int
) -> dict[str, Any]:
    import torch
    from torchsky.sphere import spectral

    starts, lengths, _, _ = spectral._ring_layout_for_nside(case.nside)
    npix = 12 * case.nside * case.nside
    nrings = int(lengths.numel())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real = torch.randn(npix, dtype=torch.float64, generator=generator)
    imag = torch.randn(npix, dtype=torch.float64, generator=generator)
    row = torch.complex(real, imag)
    modes = torch.complex(
        torch.randn(2, case.mmax + 1, nrings, dtype=torch.float64, generator=generator),
        torch.randn(2, case.mmax + 1, nrings, dtype=torch.float64, generator=generator),
    )
    rss_before = _current_rss_bytes()

    def analysis() -> Any:
        return spectral._ring_fourier_modes_spin_conj(
            row,
            starts_cpu=starts,
            lengths_cpu=lengths,
            mmax=case.mmax,
            nside=case.nside,
        )

    def synthesis() -> Any:
        return spectral._ring_fourier_synthesis(
            modes,
            starts_cpu=starts,
            lengths_cpu=lengths,
            nside=case.nside,
        )

    inner_loops = _inner_loops(case)
    analysis_stats, analysis_out = _measure(
        analysis, warmups=warmups, runs=runs, inner_loops=inner_loops
    )
    synthesis_stats, synthesis_out = _measure(
        synthesis, warmups=warmups, runs=runs, inner_loops=inner_loops
    )
    rss_after_timing = _current_rss_bytes()
    peak_after_timing = _rss_bytes()

    flags = (
        spectral._RING_FOURIER_CPP_ENABLE,
        spectral._RING_FOURIER_MODES_CPP_ENABLE,
        spectral._RING_FOURIER_SYNTH_CPP_ENABLE,
    )
    spectral._RING_FOURIER_CPP_ENABLE = False
    spectral._RING_FOURIER_MODES_CPP_ENABLE = False
    spectral._RING_FOURIER_SYNTH_CPP_ENABLE = False
    try:
        analysis_ref = analysis()
        synthesis_ref = synthesis()
    finally:
        (
            spectral._RING_FOURIER_CPP_ENABLE,
            spectral._RING_FOURIER_MODES_CPP_ENABLE,
            spectral._RING_FOURIER_SYNTH_CPP_ENABLE,
        ) = flags

    test_modes = modes[0:1]
    generic_analysis = spectral._ring_fourier_modes(
        row.unsqueeze(0),
        starts_cpu=starts,
        lengths_cpu=lengths,
        mmax=case.mmax,
        nside=case.nside,
    )
    generic_synthesis = spectral._ring_fourier_synthesis(
        test_modes,
        starts_cpu=starts,
        lengths_cpu=lengths,
        nside=case.nside,
    )
    lhs = torch.vdot(generic_analysis.reshape(-1), test_modes.reshape(-1))
    rhs = torch.vdot(row.reshape(-1), generic_synthesis.reshape(-1))
    adjoint_error = float(
        torch.abs(lhs - rhs)
        / torch.maximum(
            torch.maximum(torch.abs(lhs), torch.abs(rhs)),
            torch.tensor(1.0e-30, dtype=torch.float64),
        )
    )
    return {
        "policy": policy,
        "analysis": analysis_stats,
        "synthesis": synthesis_stats,
        "analysis_rel_l2_vs_forced_aten": max(
            _rel_l2(analysis_out[0], analysis_ref[0]),
            _rel_l2(analysis_out[1], analysis_ref[1]),
        ),
        "synthesis_rel_l2_vs_forced_aten": _rel_l2(synthesis_out, synthesis_ref),
        "adjoint_rel_error": adjoint_error,
        "rss_before_bytes": rss_before,
        "rss_after_timing_bytes": rss_after_timing,
        "rss_timing_delta_bytes": max(rss_after_timing - rss_before, 0),
        "peak_rss_after_timing_bytes": peak_after_timing,
        "peak_rss_with_correctness_bytes": _rss_bytes(),
    }


def _sht_evidence(case: Case, warmups: int, runs: int, seed: int) -> dict[str, Any]:
    import healpy
    import torch
    from torchsky.sphere import spectral

    npix = 12 * case.nside * case.nside
    rng = np.random.default_rng(seed)
    maps_np = rng.normal(size=(2, npix)).astype(np.float64)
    maps = torch.from_numpy(maps_np.copy())
    alm_e, alm_b = _random_spin_alms(
        case.lmax, case.mmax, case.spin, seed=seed + 1
    )
    rss_before = _current_rss_bytes()

    def analysis() -> Any:
        return spectral.map2alm_spin(
            maps,
            spin=case.spin,
            nside=case.nside,
            lmax=case.lmax,
            mmax=case.mmax,
        )

    def synthesis() -> Any:
        return spectral.alm2map_spin(
            (alm_e, alm_b),
            nside=case.nside,
            spin=case.spin,
            lmax=case.lmax,
            mmax=case.mmax,
        )

    inner_loops = _inner_loops(case)
    analysis_stats, analyzed = _measure(
        analysis, warmups=warmups, runs=runs, inner_loops=inner_loops
    )
    synthesis_stats, synthesized = _measure(
        synthesis, warmups=warmups, runs=runs, inner_loops=inner_loops
    )
    rss_after_timing = _current_rss_bytes()
    peak_after_timing = _rss_bytes()

    hp_e, hp_b = healpy.map2alm_spin(
        maps_np,
        spin=case.spin,
        lmax=case.lmax,
        mmax=case.mmax,
    )
    hp_maps = healpy.alm2map_spin(
        [alm_e.numpy(), alm_b.numpy()],
        nside=case.nside,
        spin=case.spin,
        lmax=case.lmax,
        mmax=case.mmax,
    )
    weights = _packed_weights(case.lmax, case.mmax)
    pixel_weight = 4.0 * math.pi / npix
    lhs = pixel_weight * torch.sum(maps * synthesized)
    rhs = torch.sum(
        weights
        * torch.real(torch.conj(analyzed[0]) * alm_e + torch.conj(analyzed[1]) * alm_b)
    )
    adjoint_error = float(
        torch.abs(lhs - rhs)
        / torch.maximum(
            torch.maximum(torch.abs(lhs), torch.abs(rhs)),
            torch.tensor(1.0e-30, dtype=torch.float64),
        )
    )
    return {
        "scheduler": "SpinMPool",
        "map2alm": analysis_stats,
        "alm2map": synthesis_stats,
        "map2alm_rel_l2_vs_healpy": max(
            _rel_l2(analyzed[0], hp_e), _rel_l2(analyzed[1], hp_b)
        ),
        "alm2map_rel_l2_vs_healpy": _rel_l2(synthesized, hp_maps),
        "adjoint_rel_error": adjoint_error,
        "rss_before_bytes": rss_before,
        "rss_after_timing_bytes": rss_after_timing,
        "rss_timing_delta_bytes": max(rss_after_timing - rss_before, 0),
        "peak_rss_after_timing_bytes": peak_after_timing,
        "peak_rss_with_correctness_bytes": _rss_bytes(),
    }


def _run_worker(args: argparse.Namespace) -> int:
    cases = tuple(Case(**item) for item in json.loads(args.worker_cases))
    os.environ.update(_policy_env(args.worker_policy))
    os.environ["TORCHSKY_SPIN_THREADS"] = str(args.worker_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.worker_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.vendor_fft_threads)
    os.environ["TORCHSKY_SHT_PROFILE"] = "1"

    import torch
    import torchsky
    from torchsky.sphere import spectral

    torch.set_num_threads(args.worker_threads)
    torch.set_num_interop_threads(1)
    metadata = _worker_metadata(torch, torchsky, spectral)
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        case_seed = args.seed + case_index * 1000
        fft = _fft_evidence(
            case,
            args.worker_policy,
            args.warmups,
            args.runs,
            case_seed,
        )
        os.environ["TORCHSKY_SHT_PROFILE"] = "0"
        sht = None
        if args.worker_policy == "hybrid" and not args.fft_only:
            sht = _sht_evidence(case, args.warmups, args.runs, case_seed + 1000)
        rows.append(
            {
                "case": asdict(case),
                "case_id": case.case_id,
                "policy": args.worker_policy,
                "threads": {
                    "requested": args.worker_threads,
                    "torch_intraop": torch.get_num_threads(),
                    "torch_interop": torch.get_num_interop_threads(),
                    "TORCHSKY_SPIN_THREADS": os.environ["TORCHSKY_SPIN_THREADS"],
                    "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                    "VECLIB_MAXIMUM_THREADS": os.environ[
                        "VECLIB_MAXIMUM_THREADS"
                    ],
                },
                "warmups": args.warmups,
                "runs": args.runs,
                "seed": case_seed,
                "metadata": metadata,
                "fft": fft,
                "sht": sht,
                "worker_peak_rss_bytes": _rss_bytes(),
            }
        )
    payload = {"rows": rows, "worker_peak_rss_bytes": _rss_bytes()}
    Path(args.worker_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _summarize_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keyed = {
        (row["case_id"], int(row["threads"]["requested"]), row["policy"]): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    speedups: dict[str, list[float]] = {"analysis": [], "synthesis": []}
    for case_id, threads, policy in sorted(keyed):
        if policy != "hybrid":
            continue
        hybrid = keyed[(case_id, threads, "hybrid")]
        aten = keyed.get((case_id, threads, "aten"))
        if aten is None:
            continue
        item: dict[str, Any] = {"case_id": case_id, "threads": threads}
        for operation in ("analysis", "synthesis"):
            hybrid_time = float(hybrid["fft"][operation]["median_s"])
            aten_time = float(aten["fft"][operation]["median_s"])
            speedup = aten_time / hybrid_time
            item[f"{operation}_hybrid_over_aten_speedup"] = speedup
            speedups[operation].append(speedup)
        hybrid_rss = int(hybrid["fft"]["peak_rss_after_timing_bytes"])
        aten_rss = int(aten["fft"]["peak_rss_after_timing_bytes"])
        item["hybrid_over_aten_peak_rss_ratio"] = hybrid_rss / max(aten_rss, 1)
        comparisons.append(item)
    return {
        "rows": comparisons,
        "geomean_hybrid_over_aten_speedup": {
            operation: _geomean(values) for operation, values in speedups.items()
        },
        "hybrid_regressions_over_5pct": {
            operation: sum(value < (1.0 / 1.05) for value in values)
            for operation, values in speedups.items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--threads", default="1,4,8")
    parser.add_argument("--policies", default="hybrid,aten,pocketfft,accelerate")
    parser.add_argument("--vendor-fft-threads", type=int, default=1)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--fft-only", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sht_runtime_policies.json",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-cases", help=argparse.SUPPRESS)
    parser.add_argument("--worker-policy", help=argparse.SUPPRESS)
    parser.add_argument("--worker-threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.runs <= 0 or args.warmups < 0:
        parser.error("runs must be positive and warmups must be non-negative")
    if args.vendor_fft_threads <= 0:
        parser.error("vendor FFT threads must be positive")
    if args.worker:
        if not all(
            (
                args.worker_cases,
                args.worker_policy,
                args.worker_threads,
                args.worker_json,
            )
        ):
            parser.error("worker arguments are incomplete")
        return _run_worker(args)
    try:
        cases = _parse_cases(args.cases)
        threads = _parse_csv_ints(args.threads)
        policies = _parse_csv_strings(args.policies)
    except ValueError as exc:
        parser.error(str(exc))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    policy_status = {
        policy: (
            {"available": True, "reason": None}
            if policy in AVAILABLE_POLICIES
            else {"available": False, "reason": UNAVAILABLE_POLICIES[policy]}
        )
        for policy in policies
    }
    available = [policy for policy in policies if policy in AVAILABLE_POLICIES]
    with tempfile.TemporaryDirectory(prefix="torchsky-sht-policy-") as temp_dir:
        root = Path(temp_dir)
        job_index = 0
        for thread_index, thread_count in enumerate(threads):
            thread_seed = args.seed + thread_index
            order = available if thread_index % 2 == 0 else list(reversed(available))
            for policy in order:
                output = root / f"matrix_{thread_count}_{policy}.json"
                command = [
                    sys.executable,
                    "-m",
                    "torchsky_harness.benchmarks.bench_sht_runtime_policies",
                    "--worker",
                    "--worker-cases",
                    json.dumps([asdict(case) for case in cases]),
                    "--worker-policy",
                    policy,
                    "--worker-threads",
                    str(thread_count),
                    "--worker-json",
                    str(output),
                    "--runs",
                    str(args.runs),
                    "--warmups",
                    str(args.warmups),
                    "--seed",
                    str(thread_seed),
                    "--vendor-fft-threads",
                    str(args.vendor_fft_threads),
                ]
                if args.fft_only:
                    command.append("--fft-only")
                env = os.environ.copy()
                env.update(_policy_env(policy))
                env["TORCHSKY_SPIN_THREADS"] = str(thread_count)
                env["OMP_NUM_THREADS"] = str(thread_count)
                env["MKL_NUM_THREADS"] = str(thread_count)
                env["OPENBLAS_NUM_THREADS"] = str(thread_count)
                env["VECLIB_MAXIMUM_THREADS"] = str(args.vendor_fft_threads)
                completed = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0 or not output.exists():
                    failures.append(
                        {
                            "case_id": "matrix",
                            "threads": thread_count,
                            "policy": policy,
                            "returncode": completed.returncode,
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                        }
                    )
                else:
                    worker_payload = json.loads(output.read_text(encoding="utf-8"))
                    match = re.search(
                        r"SpinMPool workers=(\d+) aten_threads=(\d+)",
                        completed.stderr,
                    )
                    for row in worker_payload["rows"]:
                        row["threads"]["spin_pool_workers"] = (
                            None if match is None else int(match.group(1))
                        )
                        rows.append(row)
                job_index += 1

    payload = {
        "schema_version": 1,
        "benchmark": "sht_runtime_policies",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "warmups": args.warmups,
        "runs": args.runs,
        "decision_eligible": args.runs >= 9 and args.warmups >= 2 and not failures,
        "cases": [asdict(case) for case in cases],
        "threads": list(threads),
        "vendor_fft_threads": args.vendor_fft_threads,
        "policy_status": policy_status,
        "methodology": {
            "process_isolation": "fresh subprocess per policy and thread count; cases run small to large",
            "policy_order": "alternated hybrid/ATen order by job",
            "vendor_fft_threads": args.vendor_fft_threads,
            "timing": "perf_counter wall time after explicit warmups; 20/3/1 calls per sample for small/medium/large buckets",
            "dispersion": "raw samples, stdev, MAD, quartiles, IQR, min, max",
            "rss": "process RSS and ru_maxrss; macOS ru_maxrss interpreted as bytes",
            "fft_correctness": "relative L2 against forced Python ATen and complex adjoint identity",
            "sht_correctness": "relative L2 against healpy and packed-alm adjoint identity",
        },
        "rows": rows,
        "comparisons": _summarize_comparisons(rows),
        "failures": failures,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} measured rows to {args.json_out}")
    for policy, status in policy_status.items():
        if not status["available"]:
            print(f"{policy}: unavailable: {status['reason']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
