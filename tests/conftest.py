"""Deterministic CPU settings without subject checkout path injection."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def pytest_configure(config):
    import torch
    torch.set_num_threads(1)


def pytest_runtest_teardown(item, nextitem):
    import torchfits
    torchfits.clear_file_cache()
