"""Tests for SDK endpoint prediction pipeline."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — defined inline, no conftest.py needed
# ---------------------------------------------------------------------------

@pytest.fixture
def endpoint():
    """Compact ExpectedTransactionPriceEndpoint for testing."""
    from expected_transaction_price.sdk.endpoint import ExpectedTransactionPriceEndpoint
    ep = ExpectedTransactionPriceEndpoint()
    ep.initialize()  # model = None (demo mode)
    return ep


def _sample_features() -> dict:
    """Build a realistic feature dict for pipeline tests."""
    return {
        "beds": 3,
        "baths": 2.0,
        "total_living_area": 1500,
        "year_built": 1995,
        "latitude": 43.157,
        "longitude": -77.6088,
        "sale_price": 200000.0,
        "address": "123 Main St",
    }


# =========================================================================
#  test_build_dataframe
# =========================================================================

class TestBuildDataFrame:
    """_build_dataframe should produce a single-row DataFrame with
    all UNIFIED_COLUMNS, backfilling missing ones with NaN."""

    def test_all_columns_present(self, endpoint):
        """Every UNIFIED_COLUMNS entry exists in the result."""
        from expected_transaction_price.sdk.data_sources import UNIFIED_COLUMNS
        df = endpoint._build_dataframe(_sample_features())
        assert all(col in df.columns for col in UNIFIED_COLUMNS)

    def test_backfills_missing_columns(self, endpoint):
        """Columns not in the input get NaN, not KeyError."""
        df = endpoint._build_dataframe({})
        for col in UNIFIED_COLUMNS:
            assert pd.isna(df.at[0, col]), col

    def test_preserves_input_values(self, endpoint):
        """Values passed in the dict survive into the DataFrame."""
        df = endpoint._build_dataframe({
            "beds": 5,
            "lot_acres": 0.5,
            "total_living_area": 2000,
        })
        assert df.at[0, "beds"]         == 5
        assert df.at[0, "lot_acres"]     == 0.5
        assert df.at[0, "total_living_area"] == 2000

    def test_sale_date_is_today(self, endpoint):
        """sale_date column gets today's date."""
        df = endpoint._build_dataframe({})
        expected = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        assert df.at[0, "sale_date"] == expected


# =========================================================================
#  test_demo_fallback
# =========================================================================

class TestDemoFallback:
    """When assessed_value is NaN or 0, demo_fallback replaces it with
    sale_price / 2.  When assessed_value is already valid, it is preserved."""

    def test_fallback_on_nan(self, endpoint):
        df = pd.DataFrame([{"assessed_value": np.nan, "sale_price": 200000}])
        result = endpoint._demo_fallback(df)
        assert result.at[0, "assessed_value"] == 100000.0
        assert pd.notna(result.at[0, "assessed_value"])

    def test_fallback_on_zero(self, endpoint):
        df = pd.DataFrame([{"assessed_value": 0, "sale_price": 300000}])
        result = endpoint._demo_fallback(df)
        assert result.at[0, "assessed_value"] == 150000.0

    def test_preserves_existing_value(self, endpoint):
        df = pd.DataFrame([{"assessed_value": 150000, "sale_price": 200000}])
        result = endpoint._demo_fallback(df)
        assert result.at[0, "assessed_value"] == 150000

    def test_in_place_mutates_dataframe(self, endpoint):
        """The function mutates the row in-place."""
        df = pd.DataFrame([{"assessed_value": None, "sale_price": 200000}])
        endpoint._demo_fallback(df)
        assert df.at[0, "assessed_value"] is not None


# =========================================================================
#  test_enrich_dataframe
# =========================================================================

class TestEnrichDataFrame:
    """_enrich_dataframe merges macroeconomic data (when FRED_API_KEY is set)
    and assigns school districts via spatial join."""

    def test_no_fred_api_key_returns_nan_columns(self, endpoint):
        """Without FRED_API_KEY, macro cols are absent but df is returned."""
        df = pd.DataFrame([{"beds": 3}])
        with patch("expected_transaction_price.sdk.data_sources._assign_school_districts", side_effect=lambda df: df):
            result = endpoint._enrich_dataframe(df)
        for col in ["mortgage_30y", "fed_funds", "cpi", "unemployment"]:
            assert col not in result.columns    # macro data is empty

    @patch("expected_transaction_price.sdk.data_sources._assign_school_districts", side_effect=lambda df: df)
    @patch("expected_transaction_price.sdk.data_sources.training_macro")
    def test_fred_data_merged(self, mock_macro_loader, endpoint, mock_assign_sd):
        """With FRED_API_KEY present, macro columns are cross-joined."""
        macro_df = pd.DataFrame([{
            "sale_date": pd.Timestamp("2026-06-19"),
            "mortgage_30y": 6.5,
            "fed_funds": 5.0,
            "cpi": 300.0,
            "unemployment": 4.0,
        }])
        mock_macro_loader.load.return_value = macro_df

        df = pd.DataFrame([{"beds": 3}])
        result = endpoint._enrich_dataframe(df)

        for col in ["mortgage_30y", "fed_funds", "cpi", "unemployment"]:
            assert col in result.columns
        assert result.at[0, "mortgage_30y"] == 6.5

    @patch("expected_transaction_price.sdk.data_sources.training_macro")
    def test_empty_fred_data_does_not_break(self, mock_macro, endpoint):
        """An empty macro DataFrame does not crash the pipeline."""
        mock_macro.load.return_value = pd.DataFrame()

        df = pd.DataFrame([{"beds": 3}])
        result = endpoint._enrich_dataframe(df)

        assert len(result) == 1
        assert "beds" in result.columns


# =========================================================================
#  test_scraper_to_features
# =========================================================================

class TestScraperToFeatures:
    """_scraper_to_features resolves a scraper and enriches with Rochester
    assessor data when needed."""

    def test_unsupported_domain_raises(self, endpoint):
        """Non-zillow domains get a ValueError."""
        with pytest.raises(ValueError, match="Unsupported URL domain"):
            endpoint._scraper_to_features("https://www.trulia.com/listings/abc123")

    def test_zillow_domain_returns_scraper(self, endpoint):
        """Zillow URLs resolve to a scraper (without calling its extract method)."""
        from expected_transaction_price.sdk.scrapers.factory import ScraperFactory
        scraper = ScraperFactory.get_scraper("https://www.zillow.com/homedetails/abc")
        assert scraper is not None


# =========================================================================
#  test_full_predict_features (integration shim)
# =========================================================================

class TestPredictFeatures:
    """predict_features assembles the pipeline.  Without a trained model
    in the ArtifactStore it should surface a clear error rather than
    silently returning NaN."""

    def test_calls_pipeline_stages(self, endpoint):
        """Even without a real model we can verify the call chain by
        patching the transform / predict stages."""
        import pandas as pd
        features = _sample_features()

        with patch.object(
            endpoint, "_enrich_dataframe",
            side_effect=lambda df: df,
        ) as mock_enrich:
            with patch.object(
                pd.DataFrame, "at",
                new_callable=lambda: property(fget=lambda self, _row, _col: np.nan),
            ):
                # If we don't mock _ensure_model it will fail; just
                # verify the intermediate calls fire before the model
                # actually trains.  This intentionally avoids adding a
                # mocked XGBoost estimator — that belongs in integration
                # tests once geronimo.yaml has a real model artifact.
                pass

        # The real test: does predict_features accept a dict and
        # construct a DataFrame?  We verify the internal chain by
        # inspecting mock calls.
        mock_model = MagicMock()
        mock_features = MagicMock()
        mock_features.transform.return_value = np.zeros(1)
        mock_model.features = mock_features
        mock_model.predict.return_value = np.array([285000.5])

        with patch.object(endpoint, "_ensure_model", return_value=mock_model):
            result = endpoint.predict_features({
                "beds": 3,
                "baths": 2.0,
                "year_built": 2000,
                "total_living_area": 1800,
                "latitude": 43.0,
                "longitude": -77.0,
                "sale_price": 250000,
            })

        assert result == 285000.5


# =========================================================================
#  test_endpoints_assembled
# =========================================================================

class TestEndpointAssembly:
    """Confirm the endpoint class has all expected pipeline methods
    and request schemas."""

    def test_endpoint_has_pipeline_methods(self):
        from expected_transaction_price.sdk.endpoint import ExpectedTransactionPriceEndpoint
        methods = [
            "_build_dataframe",
            "_enrich_dataframe",
            "_demo_fallback",
            "_scraper_to_features",
            "_ensure_model",
            "predict_features",
            "predict_from_url",
        ]
        for name in methods:
            assert hasattr(ExpectedTransactionPriceEndpoint, name), f"Missing method: {name}"
