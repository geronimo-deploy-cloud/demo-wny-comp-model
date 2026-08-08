"""Unit tests for the comp vectorization layer.

Covers:
  - Vector shape/length matches expected feature count
  - Scaler persistence and reload produces consistent results
  - Family weighting changes the vector as expected
  - Deterministic output (same input → same vector)
  - Categorical features are excluded and documented
  - Missing features raise ValueError
  - NaN handling
  - Integration with FeatureSet and Model
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from comp_finder.sdk.features import (
    ALL_VECTOR_FEATURES,
    CATEGORICAL_FEATURES,
    GEO_VELOCITY_FEATURES,
    MACRO_FEATURES,
    PHYSICAL_FEATURES,
    DEFAULT_FAMILY_WEIGHTS,
    SCALER_ARTIFACT_NAME,
    SCALER_PROJECT,
    SCALER_VERSION,
    get_feature_categories,
    get_comp_vector_info,
    CompFinderFeatures,
)
from comp_finder.sdk.model import CompFinderModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_features_df():
    """A small DataFrame mimicking raw property data."""
    return pd.DataFrame(
        {
            # Physical (12 features)
            "total_living_area": [1500.0, 2000.0, 1200.0],
            "first_floor_area": [900.0, 1100.0, 700.0],
            "second_floor_area": [600.0, 900.0, 500.0],
            "beds": [3.0, 4.0, 2.0],
            "kitchens": [1.0, 1.0, 1.0],
            "stories": [2.0, 1.0, 1.0],
            "lot_frontage": [80.0, 100.0, 60.0],
            "lot_depth": [120.0, 150.0, 90.0],
            "lot_acres": [0.25, 0.35, 0.15],
            "year_built": [1990, 2005, 1985],
            "assessed_value": [150000.0, 200000.0, 120000.0],
            "land_value": [30000.0, 50000.0, 20000.0],
            # Macro (4 features)
            "mortgage_30y": [6.5, 6.8, 7.0],
            "fed_funds": [5.25, 5.25, 5.5],
            "cpi": [300.0, 302.0, 298.0],
            "unemployment": [4.0, 3.8, 4.2],
            # Geographic velocity (12 features)
            "geo_vol_r1_w90": [10.0, 15.0, 8.0],
            "geo_ppsf_r1_w90": [120.0, 130.0, 110.0],
            "geo_vol_r1_w365": [30.0, 40.0, 25.0],
            "geo_ppsf_r1_w365": [115.0, 125.0, 105.0],
            "geo_vol_r5_w90": [50.0, 60.0, 45.0],
            "geo_ppsf_r5_w90": [118.0, 128.0, 108.0],
            "geo_vol_r5_w180": [80.0, 90.0, 70.0],
            "geo_ppsf_r5_w180": [116.0, 126.0, 106.0],
            "geo_vol_r10_w180": [200.0, 250.0, 180.0],
            "geo_ppsf_r10_w180": [114.0, 124.0, 104.0],
            "geo_vol_r10_w365": [400.0, 450.0, 350.0],
            "geo_ppsf_r10_w365": [112.0, 122.0, 102.0],
        }
    )


@pytest.fixture
def training_df():
    """A small training DataFrame for scaler fitting."""
    return pd.DataFrame(
        {
            "total_living_area": [1500.0, 2000.0, 1200.0, 1800.0, 2200.0],
            "first_floor_area": [900.0, 1100.0, 700.0, 1000.0, 1200.0],
            "second_floor_area": [600.0, 900.0, 500.0, 800.0, 1000.0],
            "beds": [3.0, 4.0, 2.0, 3.0, 5.0],
            "kitchens": [1.0, 1.0, 1.0, 1.0, 1.0],
            "stories": [2.0, 1.0, 1.0, 2.0, 1.0],
            "lot_frontage": [80.0, 100.0, 60.0, 90.0, 110.0],
            "lot_depth": [120.0, 150.0, 90.0, 130.0, 160.0],
            "lot_acres": [0.25, 0.35, 0.15, 0.30, 0.40],
            "year_built": [1990, 2005, 1985, 1995, 2010],
            "assessed_value": [150000.0, 200000.0, 120000.0, 170000.0, 230000.0],
            "land_value": [30000.0, 50000.0, 20000.0, 40000.0, 60000.0],
            "mortgage_30y": [6.5, 6.8, 7.0, 6.6, 7.1],
            "fed_funds": [5.25, 5.25, 5.5, 5.25, 5.5],
            "cpi": [300.0, 302.0, 298.0, 301.0, 299.0],
            "unemployment": [4.0, 3.8, 4.2, 3.9, 4.1],
            "geo_vol_r1_w90": [10.0, 15.0, 8.0, 12.0, 18.0],
            "geo_ppsf_r1_w90": [120.0, 130.0, 110.0, 125.0, 135.0],
            "geo_vol_r1_w365": [30.0, 40.0, 25.0, 35.0, 45.0],
            "geo_ppsf_r1_w365": [115.0, 125.0, 105.0, 120.0, 130.0],
            "geo_vol_r5_w90": [50.0, 60.0, 45.0, 55.0, 65.0],
            "geo_ppsf_r5_w90": [118.0, 128.0, 108.0, 123.0, 133.0],
            "geo_vol_r5_w180": [80.0, 90.0, 70.0, 85.0, 95.0],
            "geo_ppsf_r5_w180": [116.0, 126.0, 106.0, 121.0, 131.0],
            "geo_vol_r10_w180": [200.0, 250.0, 180.0, 220.0, 270.0],
            "geo_ppsf_r10_w180": [114.0, 124.0, 104.0, 119.0, 129.0],
            "geo_vol_r10_w365": [400.0, 450.0, 350.0, 420.0, 470.0],
            "geo_ppsf_r10_w365": [112.0, 122.0, 102.0, 117.0, 127.0],
        }
    )


# ---------------------------------------------------------------------------
# Tests: Constants / metadata
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify that comp_vector.py constants are correct."""

    def test_vector_length_matches_feature_count(self):
        assert len(ALL_VECTOR_FEATURES) == 28  # 12 + 4 + 12

    def test_physical_feature_count(self):
        assert len(PHYSICAL_FEATURES) == 12

    def test_macro_feature_count(self):
        assert len(MACRO_FEATURES) == 4

    def test_geo_velocity_feature_count(self):
        assert len(GEO_VELOCITY_FEATURES) == 12

    def test_excluded_categorical_count(self):
        assert len(CATEGORICAL_FEATURES) == 6

    def test_all_vector_features_is_concatenation(self):
        assert ALL_VECTOR_FEATURES == PHYSICAL_FEATURES + MACRO_FEATURES + GEO_VELOCITY_FEATURES

    def test_no_categorical_in_vector(self):
        for cat in CATEGORICAL_FEATURES:
            assert cat not in ALL_VECTOR_FEATURES

    def test_artifact_config(self):
        assert SCALER_ARTIFACT_NAME == "comp_vector_scaler"
        assert SCALER_PROJECT == "comp-finder"
        assert SCALER_VERSION == "1.0.0"

    def test_default_family_weights(self):
        assert DEFAULT_FAMILY_WEIGHTS == {
            "physical": 1.0,
            "macro": 1.0,
            "geographic": 1.0,
        }


# ---------------------------------------------------------------------------
# Tests: FeatureSet integration
# ---------------------------------------------------------------------------

class TestFeatureSet:
    """Verify CompFinderFeatures integrates with the FeatureSet pattern."""

    def test_features_can_be_instantiated(self):
        features = CompFinderFeatures()
        assert features is not None

    def test_features_has_comp_vector_method(self):
        features = CompFinderFeatures()
        assert hasattr(features, "build_comp_vector")
        assert callable(features.build_comp_vector)

    def test_features_has_set_scaler_method(self):
        features = CompFinderFeatures()
        assert hasattr(features, "set_scaler")
        assert callable(features.set_scaler)

    def test_build_comp_vector_with_fitted_features(self, training_df):
        """build_comp_vector on a fitted FeatureSet produces a valid vector."""
        features = CompFinderFeatures()
        scaler = StandardScaler().fit(training_df[ALL_VECTOR_FEATURES].values)
        features.fit(training_df)
        features.set_scaler(scaler)
        vector = features.build_comp_vector(training_df.iloc[:1])
        assert len(vector) == len(ALL_VECTOR_FEATURES)
        assert vector.ndim == 1

    def test_build_comp_vector_without_scaler_returns_raw(self, training_df):
        """Without a scaler, raw values are returned."""
        features = CompFinderFeatures()
        features.fit(training_df)
        vector = features.build_comp_vector(training_df.iloc[:1])
        assert len(vector) == len(ALL_VECTOR_FEATURES)

    def test_build_comp_vector_deterministic(self, training_df):
        """Same input → same vector."""
        features = CompFinderFeatures()
        scaler = StandardScaler().fit(training_df[ALL_VECTOR_FEATURES].values)
        features.fit(training_df)
        features.set_scaler(scaler)
        v1 = features.build_comp_vector(training_df.iloc[:1])
        v2 = features.build_comp_vector(training_df.iloc[:1])
        np.testing.assert_array_equal(v1, v2)

    def test_build_comp_vector_with_family_weights(self, training_df):
        """Family weights change the vector."""
        features = CompFinderFeatures()
        scaler = StandardScaler().fit(training_df[ALL_VECTOR_FEATURES].values)
        features.fit(training_df)
        features.set_scaler(scaler)
        vector_default = features.build_comp_vector(
            training_df.iloc[:1],
            family_weights=dict(DEFAULT_FAMILY_WEIGHTS),
        )
        vector_heavy_physical = features.build_comp_vector(
            training_df.iloc[:1],
            family_weights={"physical": 2.0, "macro": 1.0, "geographic": 1.0},
        )
        n_physical = len(PHYSICAL_FEATURES)
        np.testing.assert_array_almost_equal(
            vector_heavy_physical[:n_physical],
            vector_default[:n_physical] * 2.0,
        )

    def test_build_comp_vector_missing_features_raises(self, sample_features_df):
        """Missing required features raises ValueError."""
        features = CompFinderFeatures()
        features.fit(sample_features_df)
        # Only include a subset of physical features
        small = sample_features_df[["total_living_area", "beds"]]
        with pytest.raises(ValueError, match="features"):
            features.build_comp_vector(small)

    def test_build_comp_vector_empty_raises(self):
        """Empty DataFrame raises ValueError."""
        features = CompFinderFeatures()
        empty_df = pd.DataFrame({f: [] for f in ALL_VECTOR_FEATURES})
        with pytest.raises(ValueError):
            features.build_comp_vector(empty_df)

    def test_build_comp_vector_nan_handled(self, training_df):
        """NaN values are handled gracefully."""
        features = CompFinderFeatures()
        df_with_nan = training_df.copy()
        df_with_nan.loc[0, "total_living_area"] = float("nan")
        scaler = StandardScaler().fit(df_with_nan[ALL_VECTOR_FEATURES].dropna().values)
        features.fit(df_with_nan)
        features.set_scaler(scaler)
        vector = features.build_comp_vector(df_with_nan.iloc[:1])
        assert np.isnan(vector[0])  # first feature is NaN


# ---------------------------------------------------------------------------
# Tests: Model integration
# ---------------------------------------------------------------------------

class TestModelIntegration:
    """Verify CompFinderModel integrates with the Model pattern."""

    def test_model_can_be_instantiated(self):
        model = CompFinderModel()
        assert model is not None

    def test_model_has_build_comp_vector(self):
        model = CompFinderModel()
        assert hasattr(model, "build_comp_vector")

    def test_model_has_comp_vector_info(self):
        model = CompFinderModel()
        assert hasattr(model, "comp_vector_info")
        info = model.comp_vector_info
        assert info["vector_length"] == 28
        assert "all_features" in info
        assert "excluded_categorical" in info

    def test_model_save_includes_scaler(self, training_df):
        """Saving the model persists the comp vector scaler."""
        model = CompFinderModel()
        model.features = CompFinderFeatures()
        scaler = StandardScaler().fit(training_df[ALL_VECTOR_FEATURES].values)
        model.features.fit(training_df)
        model.features.set_scaler(scaler)
        model._is_fitted = True
        model.estimator = None  # No estimator for this test

        from geronimo.artifacts import ArtifactStore
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(
                project="comp-finder",
                version="1.0.0",
                base_path=os.path.join(tmpdir, "artifacts"),
            )
            paths = model.save(store)
            # Should have saved the comp vector scaler.
            comp_paths = [p for p in paths if "comp_vector_scaler" in p]
            assert len(comp_paths) == 1

    def test_model_load_restores_scaler(self, training_df):
        """Loading the model restores the comp vector scaler."""
        from geronimo.artifacts import ArtifactStore
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save.
            model1 = CompFinderModel()
            model1.features = CompFinderFeatures()
            scaler = StandardScaler().fit(training_df[ALL_VECTOR_FEATURES].values)
            model1.features.fit(training_df)
            model1.features.set_scaler(scaler)
            model1._is_fitted = True
            model1.estimator = None

            store = ArtifactStore(
                project="comp-finder",
                version="1.0.0",
                base_path=os.path.join(tmpdir, "artifacts"),
            )
            model1.save(store)

            # Load.
            model2 = CompFinderModel()
            comp_store = ArtifactStore(
                project="comp-finder",
                version="1.0.0",
                base_path=os.path.join(tmpdir, "artifacts"),
            )
            model2.load(comp_store)
            assert model2._is_fitted
            assert model2.features._comp_vector_scaler is not None


# ---------------------------------------------------------------------------
# Tests: Categorical handling documentation
# ---------------------------------------------------------------------------

class TestCategoricalHandling:
    """Verify that categorical features are excluded and documented."""

    def test_feature_categories_documentation(self):
        categories = get_feature_categories()
        for cat in CATEGORICAL_FEATURES:
            assert cat in categories
            assert len(categories[cat]) > 20  # Reasonable explanation text

    def test_comp_vector_info_includes_all_metadata(self):
        info = get_comp_vector_info()
        assert "all_features" in info
        assert "physical_features" in info
        assert "macro_features" in info
        assert "geo_velocity_features" in info
        assert "excluded_categorical" in info
        assert "vector_length" in info
        assert "family_weights" in info
        assert "categorical_handling" in info
        assert info["vector_length"] == len(ALL_VECTOR_FEATURES)
        assert info["all_features"] == ALL_VECTOR_FEATURES

    def test_comp_vector_info_model_property(self):
        model = CompFinderModel()
        info = model.comp_vector_info
        assert info["vector_length"] == 28
        assert len(info["excluded_categorical"]) == 6


# ---------------------------------------------------------------------------
# Tests: Error handling (standalone)
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Verify that missing features and bad inputs are handled properly."""

    def test_build_comp_vector_missing_features_error(self, sample_features_df):
        features = CompFinderFeatures()
        features.fit(sample_features_df)
        small = sample_features_df[["total_living_area", "beds"]]
        with pytest.raises(ValueError, match="features"):
            features.build_comp_vector(small)

    def test_build_comp_vector_empty_dataframe(self):
        features = CompFinderFeatures()
        empty_df = pd.DataFrame({f: [] for f in ALL_VECTOR_FEATURES})
        with pytest.raises(ValueError):
            features.build_comp_vector(empty_df)

    def test_build_comp_vector_all_nan_handles_gracefully(self, training_df):
        """All-NaN input is handled gracefully (NaN vector output)."""
        nan_df = pd.DataFrame({f: [float("nan")] * 5 for f in ALL_VECTOR_FEATURES})
        features = CompFinderFeatures()
        features.fit(nan_df)
        vector = features.build_comp_vector(nan_df.iloc[:1])
        assert len(vector) == len(ALL_VECTOR_FEATURES)
        assert np.isnan(vector[0])  # First feature is NaN


# ---------------------------------------------------------------------------
# Tests: Scaler persistence via ArtifactStore (integration)
# ---------------------------------------------------------------------------

class TestScalerPersistence:
    """Verify that the scaler is fit once, persisted, and produces consistent results."""

    def test_scaler_persisted_and_reloaded_consistent(self, training_df):
        """Saving and reloading via ArtifactStore yields a consistent vector."""
        from geronimo.artifacts import ArtifactStore
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save.
            model1 = CompFinderModel()
            model1.features = CompFinderFeatures()
            scaler = StandardScaler().fit(training_df[ALL_VECTOR_FEATURES].values)
            model1.features.fit(training_df)
            model1.features.set_scaler(scaler)
            model1._is_fitted = True
            model1.estimator = None

            store = ArtifactStore(
                project="comp-finder",
                version="1.0.0",
                base_path=os.path.join(tmpdir, "artifacts"),
            )
            model1.save(store)

            # Load and verify vector consistency.
            model2 = CompFinderModel()
            comp_store = ArtifactStore(
                project="comp-finder",
                version="1.0.0",
                base_path=os.path.join(tmpdir, "artifacts"),
            )
            model2.load(comp_store)

            v1 = model1.features.build_comp_vector(training_df.iloc[:1])
            v2 = model2.features.build_comp_vector(training_df.iloc[:1])
            np.testing.assert_array_almost_equal(v1, v2)

    def test_model_build_comp_vector_convenience(self, training_df):
        """Model.build_comp_vector delegates to features."""
        model = CompFinderModel()
        model.features = CompFinderFeatures()
        scaler = StandardScaler().fit(training_df[ALL_VECTOR_FEATURES].values)
        model.features.fit(training_df)
        model.features.set_scaler(scaler)
        model._is_fitted = True
        model.estimator = None

        vector = model.build_comp_vector(training_df.iloc[:1])
        assert len(vector) == len(ALL_VECTOR_FEATURES)
