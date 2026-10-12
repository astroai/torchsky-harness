from __future__ import annotations
import pytest

pytest.importorskip("requests")


import pytest

pytest.importorskip("requests")

import pytest

pytest.importorskip("requests")

import pytest

pytest.importorskip("requests")

from torchsky_harness.scripts.download_smacs0723_multiband_public import (
    _default_max_exposures_for_band,
)


def test_default_public_exposure_caps_match_short_and_long_bands() -> None:
    assert _default_max_exposures_for_band("F090W") == 100
    assert _default_max_exposures_for_band("F277W") == 100
