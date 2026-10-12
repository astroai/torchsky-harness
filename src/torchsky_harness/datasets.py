"""Paths for the small licensed fixtures distributed with the harness."""
from pathlib import Path


def fixture_directory(name: str) -> Path:
    return Path(__file__).resolve().parent / "fixtures" / name
