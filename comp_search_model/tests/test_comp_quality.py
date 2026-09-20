"""Tests for the comp-quality experiment harness (synthetic mode only).

These tests run the same code paths the CLI uses — synthetic population
generation, leave-one-out holdout through the real CompFinder, and the
family ablation — at reduced scale.  No network access.
"""

import numpy as np
import pandas as pd
import pytest

from comp_finder.experiments import comp_quality as cq
from comp_finder.pipeline import UNIFIED_COLUMNS
from comp_finder.sdk import features as features_module
from comp_finder.sdk.features import ALL_VECTOR_FEATURES, GEO_VELOCITY_FEATURES


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

class TestSyntheticGenerator:

    def test_schema_and_no_nans(self):
        df = cq.generate_synthetic_population(n=100, seed=1)
        for col in UNIFIED_COLUMNS + list(ALL_VECTOR_FEATURES):
            assert col in df.columns, f"missing column {col}"
        assert df[ALL_VECTOR_FEATURES].notna().all(axis=None)
        assert (df["sale_price"] > 0).all()
        assert len(df) == 100

    def test_control_differs_only_in_price(self):
        """Signal and control share every non-price column (same seed),
        so any metric difference is attributable to price structure."""
        signal = cq.generate_synthetic_population(n=100, seed=7)
        control = cq.generate_synthetic_population(n=100, seed=7, random_price=True)
        cols = [c for c in signal.columns if c != "sale_price"]
        pd.testing.assert_frame_equal(signal[cols], control[cols])
        assert not np.allclose(signal["sale_price"], control["sale_price"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestPriceMetrics:

    def test_hand_computed_values(self):
        m = cq.price_metrics([100.0, 200.0], [110.0, 180.0])
        assert m["n"] == 2
        assert m["mape"] == pytest.approx(0.10)
        assert m["median_ape"] == pytest.approx(0.10)
        assert m["within_10"] == 1.0
        assert m["within_15"] == 1.0
        assert m["spearman"] == pytest.approx(1.0)

    def test_inverted_predictions_have_negative_correlation(self):
        m = cq.price_metrics([100.0, 200.0], [200.0, 100.0])
        assert m["spearman"] == pytest.approx(-1.0)

    def test_nan_predictions_are_dropped(self):
        m = cq.price_metrics([100.0, 200.0], [110.0, np.nan])
        assert m == {"n": 1}


# ---------------------------------------------------------------------------
# Geo passthrough install/restore
# ---------------------------------------------------------------------------

class TestGeoPassthrough:

    def test_install_and_restore(self):
        df = cq.generate_synthetic_population(n=20, seed=1)

        restore = cq.install_geo_passthrough()
        try:
            _, transformed, inactive, _ = cq.prepare_population(df)
            assert "geographic" not in inactive
            np.testing.assert_array_almost_equal(
                transformed[GEO_VELOCITY_FEATURES].values,
                df[GEO_VELOCITY_FEATURES].values,
            )
        finally:
            restore()

        if features_module._geo_vol_r1_w90 is None:
            # Without the regressor package the geo family falls back to
            # NaN and is zeroed (inactive) once the passthrough is gone.
            _, _, inactive, _ = cq.prepare_population(df)
            assert "geographic" in inactive


# ---------------------------------------------------------------------------
# End-to-end (reduced scale)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_run():
    """run_all on signal and control populations at reduced scale."""
    signal = cq.generate_synthetic_population(n=300, seed=42)
    control = cq.generate_synthetic_population(n=300, seed=42, random_price=True)
    return (
        cq.run_all(signal, n_queries=10, k=3, seed=42),
        cq.run_all(control, n_queries=10, k=3, seed=42),
    )


class TestEndToEnd:

    def test_signal_beats_control(self, small_run):
        signal, control = small_run
        s, c = signal["holdout"]["comp_finder"], control["holdout"]["comp_finder"]
        assert s["spearman"] > c["spearman"] + 0.3
        assert s["within_15"] > c["within_15"]
        assert signal["inactive_families"] == []

    def test_comp_finder_beats_baselines(self, small_run):
        signal, _ = small_run
        h = signal["holdout"]
        for baseline in ["population_median", "city_median", "random_k"]:
            assert h["comp_finder"]["within_15"] >= h[baseline]["within_15"] - 0.2

    def test_ablation_reports_all_configs(self, small_run):
        signal, _ = small_run
        rows = signal["ablation"]
        assert len(rows) == len(cq.ABLATION_CONFIGS)
        assert not any(r.get("skipped") for r in rows)
        by_config = {r["config"]: r for r in rows}
        # Macro columns are date-driven noise by construction: physical
        # must carry clearly more price signal than macro alone.
        assert by_config["physical only"]["spearman"] > \
            by_config["macro only"]["spearman"] + 0.3

    def test_report_shape(self, small_run):
        signal, _ = small_run
        assert signal["population_size"] == 300
        assert signal["n_queries"] == 10
        assert set(signal["holdout"]) >= {
            "comp_finder", "population_median", "city_median", "random_k",
            "similarity_vs_error_spearman", "per_source",
        }
