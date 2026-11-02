"""Shared helpers for comparator-parity suites (device-aware, CPU oracles)."""

from __future__ import annotations

import numpy as np
import torch

from torchsky.device import default_compute_device


def device() -> torch.device:
    """Device the subject computes on (honours ``TORCHSKY_DEVICE``)."""
    return default_compute_device()


def f64(x, dev: torch.device | None = None) -> torch.Tensor:
    """float64 tensor on ``dev`` (MPS has no float64, so it falls back to CPU)."""
    dev = dev or device()
    if dev.type == "mps":
        dev = torch.device("cpu")
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=dev)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def sep_arcsec(lon1_deg, lat1_deg, lon2_deg, lat2_deg) -> np.ndarray:
    """Great-circle separation in arcsec (Vincenty), numpy float64."""
    l1, b1, l2, b2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lon1_deg, lat1_deg, lon2_deg, lat2_deg))
    dl = l2 - l1
    num = np.hypot(np.cos(b2) * np.sin(dl), np.cos(b1) * np.sin(b2) - np.sin(b1) * np.cos(b2) * np.cos(dl))
    den = np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(dl)
    return np.degrees(np.arctan2(num, den)) * 3600.0
