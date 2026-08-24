"""Tests for the comp-vector index pipeline.

Covers:
  - Pipeline initialization and state management
  - Comp vector building from synthetic sales data
  - BallTree index construction and query
  - Lookup table construction and round-trip
  - ArtifactStore save/load round-trip for index and lookup
  - Error handling (unfitted pipeline, missing scaler)
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

from comp_finder.pipeline import (
    CompSearchPipeline,
    INDEX_PROJECT,
    INDEX_VERSION,
    INDEX_ARTIFACT_NAME,
    LOOKUP_ARTIFACT_NAME,
    UNIFIED_COLUMNS,
)
from comp_finder.sdk.features import ALL_VECTOR_FEATURES
from geronimo.artifacts import ArtifactStore


# =============================================================================
# Helpers
# =============================================================================


def _make_synthetic_sales(
    n: int = 100,
    center_lat: float = 42.89,
    center_lng: float = -78.88,
) -> pd.DataFrame:
    """Create synthetic sales data for pipeline testing."""
    rng = np.random.RandomState(42)
    now = datetime.utcnow()

    return pd.DataFrame({
        "source": ["buffalo"] * n,
        "parcel_id": [f"parcel_{i}" for i in range(n)],
        "print_key": [f"PK_{i}" for i in range(n)],
        "address": [f"{100 + i} Main St" for i in range(n)],
        "city": ["Buffalo"] * n,
        "zip_code": ["14201"] * n,
        "property_class": ["210"] * n,
        "property_class_desc": ["1 Family Residential"] * n,
        "sale_price": rng.uniform(100_000, 500_000, n),
        "sale_date": pd.date_range(end=now - timedelta(days=1), periods=n, freq="D"),
        "assessed_value": rng.uniform(80_000, 400_000, n),
        "land_value": rng.uniform(20_000, 100_000, n),
        "year_built": rng.randint(1950, 2020, n).astype(float),
        "total_living_area": rng.uniform(800, 3000, n),
        "first_floor_area": rng.uniform(500, 2000, n),
        "second_floor_area": rng.uniform(200, 1000, n),
        "beds": rng.randint(2, 6, n).astype(float),
        "baths": rng.uniform(1, 4, n),
        "half_baths": 0.0,
        "kitchens": 1.0,
        "stories": rng.choice([1.0, 2.0], n),
        "fireplaces": rng.randint(0, 3, n).astype(float),
        "lot_frontage": rng.uniform(50, 150, n),
        "lot_depth": rng.uniform(80, 200, n),
        "lot_acres": rng.uniform(0.1, 0.5, n),
        "building_style": ["Ranch"] * n,
        "overall_condition": ["Normal"] * n,
        "construction_grade": ["Average"] * n,
        "exterior_wall": ["Brick"] * n,
        "heat_type": ["Gas"] * n,
        "central_air": True,
        "fuel_type": ["Gas"] * n,
        "basement_type": ["Full"] * n,
        "latitude": center_lat + rng.normal(0, 0.01, n),
        "longitude": center_lng + rng.normal(0, 0.01, n),
        "school_district": ["Buffalo City"] * n,
    })


def _make_synthetic_features(
    n: int = 100,
) -> pd.DataFrame:
    """Create synthetic feature data (output of FeatureSet.transform)."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        # Physical (12)
        "total_living_area": rng.uniform(800, 3000, n),
        "first_floor_area": rng.uniform(500, 2000, n),
        "second_floor_area": rng.uniform(200, 1000, n),
        "beds": rng.randint(2, 6, n).astype(float),
        "kitchens": 1.0,
        "stories": rng.choice([1.0, 2.0], n),
        "lot_frontage": rng.uniform(50, 150, n),
        "lot_depth": rng.uniform(80, 200, n),
        "lot_acres": rng.uniform(0.1, 0.5, n),
        "year_built": rng.randint(1950, 2020, n).astype(float),
        "assessed_value": rng.uniform(80_000, 400_000, n),
        "land_value": rng.uniform(20_000, 100_000, n),
        # Macro (4)
        "mortgage_30y": rng.uniform(5.0, 8.0, n),
        "fed_funds": rng.uniform(4.0, 6.0, n),
        "cpi": rng.uniform(280, 320, n),
        "unemployment": rng.uniform(3.0, 6.0, n),
        # Geographic velocity (12)
        "geo_vol_r1_w90": rng.uniform(5, 50, n),
        "geo_ppsf_r1_w90": rng.uniform(100, 200, n),
        "geo_vol_r1_w365": rng.uniform(15, 100, n),
        "geo_ppsf_r1_w365": rng.uniform(90, 180, n),
        "geo_vol_r5_w90": rng.uniform(20, 100, n),
        "geo_ppsf_r5_w90": rng.uniform(95, 190, n),
        "geo_vol_r5_w180": rng.uniform(30, 150, n),
        "geo_ppsf_r5_w180": rng.uniform(90, 185, n),
        "geo_vol_r10_w180": rng.uniform(50, 200, n),
        "geo_ppsf_r10_w180": rng.uniform(85, 175, n),
        "geo_vol_r10_w365": rng.uniform(80, 300, n),
        "geo_ppsf_r10_w365": rng.uniform(80, 170, n),
    })


def _make_pipeline_with_index(n: int = 50) -> tuple:
    """Helper: create a pipeline with a fitted BallTree index.

    Returns (pipeline, comp_vectors, features_df).
    """
    pipeline = CompSearchPipeline()
    pipeline.initialize()

    features_df = _make_synthetic_features(n)
    pipeline.features.fit(features_df)

    scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
    pipeline.features.set_scaler(scaler)

    comp_vectors = pipeline.features.build_comp_vector(features_df)

    from sklearn.neighbors import BallTree
    pipeline._index = BallTree(comp_vectors, metric="euclidean")

    # Also store sales data as the lookup table (for tests that check unified columns)
    sales = _make_synthetic_sales(n)
    pipeline._lookup_df = sales[UNIFIED_COLUMNS].copy()
    pipeline._lookup_df["comp_vector_index"] = range(n)

    return pipeline, comp_vectors, features_df


# =============================================================================
# Pipeline Initialization
# =============================================================================


class TestPipelineInit:
    """Tests for pipeline initialization."""

    def test_pipeline_can_be_instantiated(self):
        pipeline = CompSearchPipeline()
        assert pipeline is not None
        assert not pipeline.is_fitted

    def test_pipeline_default_resolution(self):
        pipeline = CompSearchPipeline()
        assert pipeline.h3_resolution == 8

    def test_pipeline_has_required_attributes(self):
        pipeline = CompSearchPipeline()
        pipeline.initialize()
        assert hasattr(pipeline, "features")
        assert hasattr(pipeline, "run")
        assert hasattr(pipeline, "load")
        assert hasattr(pipeline, "find_comps")

    def test_pipeline_not_fitted_by_default(self):
        pipeline = CompSearchPipeline()
        assert not pipeline.is_fitted
        assert not getattr(pipeline, "_is_initialized", False)

    def test_pipeline_is_initialized_after_initialize(self):
        pipeline = CompSearchPipeline()
        pipeline.initialize()
        assert pipeline._is_initialized


# =============================================================================
# Comp Vector Building
# =============================================================================


class TestCompVectorBuilding:
    """Tests for comp vector building from synthetic data."""

    def test_builds_comp_vector_from_features(self):
        pipeline = CompSearchPipeline()
        pipeline.initialize()
        features_df = _make_synthetic_features(50)
        pipeline.features.fit(features_df)

        scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
        pipeline.features.set_scaler(scaler)

        vector = pipeline.features.build_comp_vector(features_df.iloc[:1])
        assert len(vector) == len(ALL_VECTOR_FEATURES)
        assert vector.ndim == 1

    def test_comp_vector_length_is_28(self):
        pipeline = CompSearchPipeline()
        pipeline.initialize()
        features_df = _make_synthetic_features(10)
        pipeline.features.fit(features_df)
        scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
        pipeline.features.set_scaler(scaler)

        vector = pipeline.features.build_comp_vector(features_df.iloc[:1])
        assert len(vector) == 28  # 12 + 4 + 12

    def test_comp_vector_deterministic(self):
        pipeline = CompSearchPipeline()
        pipeline.initialize()
        features_df = _make_synthetic_features(10)
        pipeline.features.fit(features_df)
        scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
        pipeline.features.set_scaler(scaler)

        v1 = pipeline.features.build_comp_vector(features_df.iloc[:1])
        v2 = pipeline.features.build_comp_vector(features_df.iloc[:1])
        np.testing.assert_array_equal(v1, v2)


# =============================================================================
# BallTree Index
# =============================================================================


class TestBallTreeIndex:
    """Tests for BallTree index construction and querying."""

    def test_deterministic_index_rebuild(self):
        """Two independent pipeline runs on the same data produce identical indices."""
        pipeline1 = CompSearchPipeline()
        pipeline1.initialize()
        features_df = _make_synthetic_features(50)
        pipeline1.features.fit(features_df)
        scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
        pipeline1.features.set_scaler(scaler)
        comp_vectors1 = pipeline1.features.build_comp_vector(features_df)
        from sklearn.neighbors import BallTree
        pipeline1._index = BallTree(comp_vectors1, metric="euclidean")

        pipeline2 = CompSearchPipeline()
        pipeline2.initialize()
        pipeline2.features.fit(features_df)
        pipeline2.features.set_scaler(scaler)
        comp_vectors2 = pipeline2.features.build_comp_vector(features_df)
        pipeline2._index = BallTree(comp_vectors2, metric="euclidean")

        # Both pipelines query the same vector — results must match exactly
        v = comp_vectors1[:1]
        d1, i1 = pipeline1._index.query(v, k=3)
        d2, i2 = pipeline2._index.query(v, k=3)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(i1, i2)

    def test_balltree_returns_self_as_nearest(self):
        """A vector should find itself as the nearest neighbor."""
        pipeline = CompSearchPipeline()
        pipeline.initialize()
        features_df = _make_synthetic_features(50)
        pipeline.features.fit(features_df)

        scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
        pipeline.features.set_scaler(scaler)

        comp_vectors = pipeline.features.build_comp_vector(features_df)

        from sklearn.neighbors import BallTree
        index = BallTree(comp_vectors, metric="euclidean")

        distances, indices = index.query(comp_vectors[:1], k=1)
        assert indices[0, 0] == 0

    def test_balltree_queries_return_valid_indices(self):
        pipeline = CompSearchPipeline()
        pipeline.initialize()
        features_df = _make_synthetic_features(50)
        pipeline.features.fit(features_df)
        scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
        pipeline.features.set_scaler(scaler)

        comp_vectors = pipeline.features.build_comp_vector(features_df)
        from sklearn.neighbors import BallTree
        index = BallTree(comp_vectors, metric="euclidean")

        distances, indices = index.query(comp_vectors, k=5)
        assert distances.shape == (50, 5)
        assert indices.shape == (50, 5)
        assert (distances >= 0).all()

    def test_pipeline_find_comps_returns_lookup_rows(self):
        """Pipeline.find_comps should return lookup table rows for nearest comps."""
        pipeline, comp_vectors, _ = _make_pipeline_with_index(50)

        result = pipeline.find_comps(comp_vectors[0], k=5)
        assert len(result) == 5
        assert "total_living_area" in result.columns
        assert "beds" in result.columns

    def test_pipeline_find_comps_k_equals_n(self):
        """When k equals the number of records, should return all."""
        pipeline, comp_vectors, _ = _make_pipeline_with_index(20)

        result = pipeline.find_comps(comp_vectors[0], k=20)
        assert len(result) == 20

    def test_pipeline_is_fitted_after_initialize_index(self):
        pipeline, _, _ = _make_pipeline_with_index(50)
        assert pipeline.is_fitted


# =============================================================================
# Lookup Table
# =============================================================================


class TestLookupTable:
    """Tests for the lookup table construction."""

    def test_lookup_table_has_required_columns(self):
        pipeline, _, features_df = _make_pipeline_with_index(50)

        for col in UNIFIED_COLUMNS:
            assert col in pipeline._lookup_df.columns

    def test_lookup_table_row_count_matches_features(self):
        pipeline, _, features_df = _make_pipeline_with_index(100)
        assert len(pipeline._lookup_df) == 100

    def test_lookup_table_has_comp_vector_index_column(self):
        pipeline, _, features_df = _make_pipeline_with_index(50)
        assert "comp_vector_index" in pipeline._lookup_df.columns
        assert list(pipeline._lookup_df["comp_vector_index"]) == list(range(50))

    def test_lookup_table_preserves_all_features(self):
        pipeline, _, features_df = _make_pipeline_with_index(50)

        # The lookup table is sales data, so check for sales columns
        assert "sale_price" in pipeline._lookup_df.columns
        assert "sale_date" in pipeline._lookup_df.columns
        assert "address" in pipeline._lookup_df.columns


# =============================================================================
# ArtifactStore Round-Trip
# =============================================================================


class TestArtifactRoundTrip:
    """Tests for ArtifactStore save/load round-trip."""

    def test_save_and_load_index(self):
        """Index should be savable and loadable from ArtifactStore."""
        import joblib
        from io import BytesIO

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline, comp_vectors, features_df = _make_pipeline_with_index(50)

            store = ArtifactStore(
                project=INDEX_PROJECT,
                version=INDEX_VERSION,
                base_path=tmpdir,
            )

            # Save
            saved = pipeline._publish(store)
            assert INDEX_ARTIFACT_NAME in saved

            # Load into a new pipeline
            pipeline2 = CompSearchPipeline()
            store2 = ArtifactStore(
                project=INDEX_PROJECT,
                version=INDEX_VERSION,
                base_path=tmpdir,
            )
            pipeline2.load(store2)

            assert pipeline2.is_fitted
            assert pipeline2._index is not None

    def test_save_and_load_lookup_table(self):
        """Lookup table should be savable and loadable from ArtifactStore."""
        import joblib
        from io import BytesIO

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline, _, features_df = _make_pipeline_with_index(50)

            store = ArtifactStore(
                project=INDEX_PROJECT,
                version=INDEX_VERSION,
                base_path=tmpdir,
            )

            # Save
            saved = pipeline._publish(store)
            assert LOOKUP_ARTIFACT_NAME in saved

            # Load into a new pipeline
            pipeline2 = CompSearchPipeline()
            store2 = ArtifactStore(
                project=INDEX_PROJECT,
                version=INDEX_VERSION,
                base_path=tmpdir,
            )
            pipeline2.load(store2)

            assert pipeline2.is_fitted
            assert pipeline2._lookup_df is not None
            assert len(pipeline2._lookup_df) == 50

    def test_find_comps_after_load(self):
        """Should be able to find comps after loading from ArtifactStore."""
        import joblib
        from io import BytesIO

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline, comp_vectors, _ = _make_pipeline_with_index(50)

            store = ArtifactStore(
                project=INDEX_PROJECT,
                version=INDEX_VERSION,
                base_path=tmpdir,
            )
            pipeline._publish(store)

            # Load and query
            pipeline2 = CompSearchPipeline()
            store2 = ArtifactStore(
                project=INDEX_PROJECT,
                version=INDEX_VERSION,
                base_path=tmpdir,
            )
            pipeline2.load(store2)

            result = pipeline2.find_comps(comp_vectors[0], k=5)
            assert len(result) == 5

    def test_artifact_names_are_correct(self):
        assert INDEX_ARTIFACT_NAME == "comp_vector_index"
        assert LOOKUP_ARTIFACT_NAME == "comp_vector_lookup"


# =============================================================================
# Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_find_comps_raises_if_not_fitted(self):
        pipeline = CompSearchPipeline()
        pipeline.initialize()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.find_comps(np.zeros(28), k=5)

    def test_load_raises_for_missing_artifact(self):
        pipeline = CompSearchPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(
                project=INDEX_PROJECT,
                version=INDEX_VERSION,
                base_path=tmpdir,
            )
            with pytest.raises(Exception):  # Artifact not found
                pipeline.load(store)

    def test_find_comps_wrong_vector_length(self):
        pipeline, comp_vectors, _ = _make_pipeline_with_index(10)
        with pytest.raises(Exception):  # Wrong length
            pipeline.find_comps(np.zeros(10), k=5)

    def test_run_raises_if_not_initialized(self):
        pipeline = CompSearchPipeline()
        with pytest.raises(RuntimeError, match="not initialized"):
            pipeline.run()


# =============================================================================
# Constants
# =============================================================================


class TestConstants:
    """Verify pipeline constants."""

    def test_index_project(self):
        assert INDEX_PROJECT == "comp-finder"

    def test_index_version(self):
        assert INDEX_VERSION == "1.0.0"

    def test_index_artifact_name(self):
        assert INDEX_ARTIFACT_NAME == "comp_vector_index"

    def test_lookup_artifact_name(self):
        assert LOOKUP_ARTIFACT_NAME == "comp_vector_lookup"

    def test_unified_columns_list_is_complete(self):
        assert "sale_price" in UNIFIED_COLUMNS
        assert "address" in UNIFIED_COLUMNS
        assert "sale_date" in UNIFIED_COLUMNS
        assert "latitude" in UNIFIED_COLUMNS
        assert "longitude" in UNIFIED_COLUMNS
        assert len(UNIFIED_COLUMNS) == 36  # All unified schema columns
