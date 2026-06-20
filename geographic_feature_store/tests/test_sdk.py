"""Tests for geographic feature store SDK modules."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# =============================================================================
# Helpers
# =============================================================================


def _make_synthetic_sales(
    n: int = 100,
    center_lat: float = 42.89,
    center_lng: float = -78.88,
    source: str = "buffalo",
) -> pd.DataFrame:
    """Create synthetic sales data for testing."""
    rng = np.random.RandomState(42)
    now = datetime.utcnow()

    return pd.DataFrame({
        "source": source,
        "parcel_id": [f"parcel_{i}" for i in range(n)],
        "latitude": center_lat + rng.normal(0, 0.01, n),
        "longitude": center_lng + rng.normal(0, 0.01, n),
        "sale_date": pd.date_range(end=now - timedelta(days=1), periods=n, freq="D"),
        "sale_price": rng.uniform(50_000, 500_000, n),
        "total_living_area": rng.uniform(800, 3000, n),
    })


def _make_input_locations(
    center_lat: float = 42.89,
    center_lng: float = -78.88,
    n: int = 3,
) -> pd.DataFrame:
    """Create input locations for testing transform/predict."""
    return pd.DataFrame({
        "latitude": [center_lat + i * 0.001 for i in range(n)],
        "longitude": [center_lng + i * 0.001 for i in range(n)],
    })


# =============================================================================
# Feature Declarations
# =============================================================================


class TestFeatureDeclarations:
    """Tests that the FeatureSet declares the expected Feature objects."""

    def test_has_24_features(self):
        """Should have 24 declared Features (12 combos × 2 metrics)."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        assert len(features.feature_names) == 24

    def test_feature_names_match_pattern(self):
        """Feature names should follow the radius/window naming pattern."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        names = features.feature_names

        # Check some specific expected names
        assert "sales_volume_r1_w30" in names
        assert "median_ppsf_r1_w30" in names
        assert "sales_volume_r5_w90" in names
        assert "median_ppsf_r5_w90" in names
        assert "sales_volume_r10_w365" in names
        assert "median_ppsf_r10_w365" in names

    def test_all_features_have_derived_fn(self):
        """Every declared Feature should have a derived_feature_fn."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        for name, feat in features._features.items():
            assert feat.has_derived_fn, f"Feature {name} is missing derived_feature_fn"

    def test_all_features_are_numeric(self):
        """All velocity features should be dtype='numeric'."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        for name, feat in features._features.items():
            assert feat.dtype == "numeric", f"Feature {name} has dtype={feat.dtype}"

    def test_all_features_have_descriptions(self):
        """All velocity features should have a description."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        for name, feat in features._features.items():
            assert feat.description, f"Feature {name} missing description"


# =============================================================================
# Fit (Grid + Velocity Computation)
# =============================================================================


class TestFit:
    """Tests for the fit path (grid generation + velocity computation)."""

    def test_fit_generates_grid(self):
        """Fitting should generate H3 grid cells."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(50))

        assert features.grid is not None
        assert len(features.grid) > 0
        assert features.is_fitted

    def test_fit_produces_12_velocity_grids(self):
        """Fitting should produce 12 velocity DataFrames (3 radii × 4 windows)."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(50))

        assert len(features.velocity_data) == 12

    def test_grid_covers_buffalo_city_hall(self):
        """Fitted grid should contain a cell near Buffalo City Hall."""
        import h3
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(50))

        city_hall_cell = h3.latlng_to_cell(42.8864, -78.8784, 8)
        assert city_hall_cell in features.grid["h3_index"].values

    def test_no_duplicate_cells(self):
        """Grid should not contain duplicate H3 cells."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(50))

        assert features.grid["h3_index"].nunique() == len(features.grid)

    def test_resolution_is_tunable(self):
        """Higher resolution should produce more cells."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        sales = _make_synthetic_sales(50)

        low = GeoVelocityFeatures(h3_resolution=7)
        low.fit(sales)

        high = GeoVelocityFeatures(h3_resolution=8)
        high.fit(sales)

        assert len(high.grid) > len(low.grid)

    def test_larger_radius_more_volume(self):
        """Larger radius should capture >= sales than smaller."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(100))

        vol_1mi = features.velocity_data["geo_velocity_r1_w365"]["sales_volume"].max()
        vol_10mi = features.velocity_data["geo_velocity_r10_w365"]["sales_volume"].max()
        assert vol_10mi >= vol_1mi

    def test_larger_window_more_volume(self):
        """Larger lookback window should capture >= sales than smaller."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(100))

        vol_30d = features.velocity_data["geo_velocity_r10_w30"]["sales_volume"].max()
        vol_365d = features.velocity_data["geo_velocity_r10_w365"]["sales_volume"].max()
        assert vol_365d >= vol_30d

    def test_no_future_sales_leak(self):
        """Should not include sales on or after reference_date."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        now = datetime.utcnow()
        future_only = pd.DataFrame({
            "source": ["buffalo"],
            "parcel_id": ["p1"],
            "latitude": [42.89],
            "longitude": [-78.88],
            "sale_date": [now + timedelta(days=1)],
            "sale_price": [200_000],
            "total_living_area": [1500],
        })

        features = GeoVelocityFeatures()
        features.fit(future_only)

        # Velocity grids created but all are empty (no sales before reference_date)
        for name, vel_df in features.velocity_data.items():
            assert vel_df.empty or vel_df["sales_volume"].sum() == 0, \
                f"{name} should have zero sales volume"

    def test_correct_ppsf_calculation(self):
        """Median PPSF should be correctly computed."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        now = datetime.utcnow()
        sales = pd.DataFrame({
            "source": ["buffalo"] * 3,
            "parcel_id": ["p1", "p2", "p3"],
            "latitude": [42.89, 42.89, 42.89],
            "longitude": [-78.88, -78.88, -78.88],
            "sale_date": [
                now - timedelta(days=10),
                now - timedelta(days=20),
                now - timedelta(days=30),
            ],
            "sale_price": [150_000, 200_000, 100_000],
            "total_living_area": [1000, 1000, 1000],
        })

        features = GeoVelocityFeatures()
        features.fit(sales)

        df = features.velocity_data["geo_velocity_r10_w365"]
        max_vol_idx = df["sales_volume"].idxmax()
        assert df.loc[max_vol_idx, "sales_volume"] == 3
        assert df.loc[max_vol_idx, "median_ppsf"] == pytest.approx(150.0, rel=1e-2)


# =============================================================================
# Transform (H3 Cell Lookup)
# =============================================================================


class TestTransform:
    """Tests for the transform path (H3 cell lookup)."""

    def test_transform_returns_24_feature_columns(self):
        """Transform should return one column per declared Feature."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(50))

        result = features.transform(_make_input_locations())
        assert len(result.columns) == 24
        assert "sales_volume_r1_w30" in result.columns
        assert "median_ppsf_r10_w365" in result.columns

    def test_transform_preserves_row_count(self):
        """Transform should not add or remove rows."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(50))

        locations = _make_input_locations(n=5)
        result = features.transform(locations)
        assert len(result) == 5

    def test_transform_raises_if_not_fitted(self):
        """Transform should raise if not fitted."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        with pytest.raises(ValueError, match="not fitted"):
            features.transform(_make_input_locations())

    def test_transform_raises_without_lat_lng(self):
        """Transform should raise if lat/lng columns missing."""
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(50))

        with pytest.raises(ValueError, match="latitude"):
            features.transform(pd.DataFrame({"name": ["test"]}))


# =============================================================================
# Model
# =============================================================================


class TestModel:
    """Tests for the Model wrapper."""

    def test_train_returns_feature_names(self):
        """Model.train() metrics should include the list of feature names."""
        from geographic_feature_store.sdk.model import GeoFeatureStoreModel
        from geographic_feature_store.sdk import data_sources

        sales = _make_synthetic_sales(50)
        original_load = data_sources.training_sales.load
        data_sources.training_sales.load = lambda: sales

        try:
            model = GeoFeatureStoreModel()
            metrics = model.train()
            assert metrics["status"] == "success"
            assert "sales_volume_r1_w30" in metrics["features"]
            assert "median_ppsf_r10_w365" in metrics["features"]
            assert len(metrics["features"]) == 24
        finally:
            data_sources.training_sales.load = original_load

    def test_predict_returns_feature_columns(self):
        """Model.predict() should return all declared Feature columns."""
        from geographic_feature_store.sdk.model import GeoFeatureStoreModel
        from geographic_feature_store.sdk import data_sources

        sales = _make_synthetic_sales(50)
        original_load = data_sources.training_sales.load
        data_sources.training_sales.load = lambda: sales

        try:
            model = GeoFeatureStoreModel()
            model.train()

            result = model.predict(_make_input_locations())
            assert "sales_volume_r1_w30" in result.columns
            assert "median_ppsf_r10_w365" in result.columns
            assert len(result) == 3
        finally:
            data_sources.training_sales.load = original_load

    def test_predict_raises_if_not_fitted(self):
        """Model.predict() should raise if not trained."""
        from geographic_feature_store.sdk.model import GeoFeatureStoreModel

        model = GeoFeatureStoreModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(_make_input_locations())

    def test_model_features_are_discoverable(self):
        """Model.features.feature_names should list all precomputed grids."""
        from geographic_feature_store.sdk.model import GeoFeatureStoreModel

        model = GeoFeatureStoreModel()
        names = model.features.feature_names

        # 24 features: 3 radii × 4 windows × 2 metrics
        assert len(names) == 24
        assert "sales_volume_r1_w30" in names
        assert "median_ppsf_r5_w90" in names
        assert "sales_volume_r10_w365" in names


# =============================================================================
# Artifact Name
# =============================================================================


class TestArtifactName:
    """Tests for artifact naming convention."""

    def test_integer_radius(self):
        from geographic_feature_store.sdk.features import artifact_name

        assert artifact_name(1.0, 30) == "geo_velocity_r1_w30"
        assert artifact_name(5.0, 90) == "geo_velocity_r5_w90"
        assert artifact_name(10.0, 365) == "geo_velocity_r10_w365"


# =============================================================================
# Repr
# =============================================================================


class TestRepr:
    """Tests for __repr__."""

    def test_features_repr_unfitted(self):
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        r = repr(features)
        assert "not fitted" in r
        assert "24 features" in r

    def test_features_repr_fitted(self):
        from geographic_feature_store.sdk.features import GeoVelocityFeatures

        features = GeoVelocityFeatures()
        features.fit(_make_synthetic_sales(50))
        r = repr(features)
        assert "fitted" in r
        assert "not fitted" not in r
