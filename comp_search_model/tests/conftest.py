"""Shared fixtures for the comp-finder test suite."""

import numpy as np
import pandas as pd
import pytest

from comp_finder.sdk.features import CompFinderFeatures, GEO_VELOCITY_FEATURES


@pytest.fixture
def geo_none(monkeypatch):
    """Emulate a venv without `expected_transaction_price`.

    There, the geo derived feature functions fail to import and stay
    ``None``, so every geographic feature emits NaN regardless of input.
    Patching them to NaN producers makes the fail-safe tests behave the
    same whether or not the regressor package happens to be installed.
    """
    for name in GEO_VELOCITY_FEATURES:
        feature = CompFinderFeatures.__dict__[name]
        monkeypatch.setattr(
            feature,
            "_derived_feature_fn",
            lambda df: pd.Series(np.nan, index=df.index),
        )
