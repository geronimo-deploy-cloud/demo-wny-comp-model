"""Tests for CompFinder — comp-vector nearest-neighbor query logic.

CompFinder loads the index / lookup / scaler artifacts from a local
ArtifactStore (temp dir), mirroring how the deployed service will load
them at startup, then exercises ``find_comps()``.

The synthetic data mirrors ``tests/test_pipeline.py``: historical sales
plus a feature table, from which the BallTree index and lookup table are
built and published before CompFinder loads them.
"""

import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from geronimo.artifacts import ArtifactStore
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

# Add src to path for test imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comp_finder.pipeline import (
    INDEX_ARTIFACT_NAME,
    INDEX_PROJECT,
    INDEX_VERSION,
    LOOKUP_ARTIFACT_NAME,
    UNIFIED_COLUMNS,
)
from comp_finder.sdk.features import (
    ALL_VECTOR_FEATURES,
    CompFinderFeatures,
    GEO_VELOCITY_FEATURES,
    SCALER_ARTIFACT_NAME,
)
from comp_finder.sdk.model import CompFinder, DEFAULT_K

RANDOM_SEED = 42


def _make_synthetic_sales(n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create synthetic historical sales rows (all UNIFIED_COLUMNS)."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "source": np.array(["buffalo", "rochester"], dtype=object)[
                rng.randint(0, 2, n)
            ],
            "parcel_id": [f"PARCEL-{i:08d}" for i in range(n)],
            "print_key": [f"PK-{i:08d}" for i in range(n)],
            "address": [f"{100 + i} Example St" for i in range(n)],
            "city": np.array(["Buffalo", "Rochester"], dtype=object)[
                rng.randint(0, 2, n)
            ],
            "zip_code": np.array(["14201", "14604"], dtype=object)[
                rng.randint(0, 2, n)
            ],
            "property_class": ["210"] * n,
            "property_class_desc": ["1 Family Residential"] * n,
            "sale_price": rng.uniform(100_000, 600_000, n),
            "sale_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "assessed_value": rng.uniform(80_000, 400_000, n),
            "land_value": rng.uniform(20_000, 100_000, n),
            "year_built": rng.randint(1950, 2020, n).astype(float),
            "total_living_area": rng.randint(800, 4000, n).astype(float),
            "first_floor_area": rng.randint(500, 2500, n).astype(float),
            "second_floor_area": rng.randint(300, 2000, n).astype(float),
            "beds": rng.randint(2, 6, n).astype(float),
            "baths": rng.randint(1, 4, n).astype(float),
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
            "latitude": 42.886 + rng.normal(0, 0.02, n),
            "longitude": -78.878 + rng.normal(0, 0.02, n),
            "school_district": np.array(
                ["Buffalo City", "Rochester City"], dtype=object
            )[rng.randint(0, 2, n)],
        }
    )[UNIFIED_COLUMNS]


def _make_synthetic_features(n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create the feature table the index is built from.

    Contains every comp-vector feature plus latitude/longitude (used to
    derive the geo features in production and echoed in query inputs).
    """
    rng = np.random.RandomState(seed + 1)
    return pd.DataFrame(
        {
            "total_living_area": rng.randint(800, 4000, n).astype(float),
            "first_floor_area": rng.randint(500, 2500, n).astype(float),
            "second_floor_area": rng.randint(300, 2000, n).astype(float),
            "beds": rng.randint(2, 6, n).astype(float),
            "kitchens": 1.0,
            "stories": rng.choice([1.0, 2.0], n),
            "lot_frontage": rng.uniform(50, 150, n),
            "lot_depth": rng.uniform(80, 200, n),
            "lot_acres": rng.uniform(0.1, 0.5, n),
            "year_built": rng.randint(1950, 2020, n).astype(float),
            "assessed_value": rng.uniform(80_000, 400_000, n),
            "land_value": rng.uniform(20_000, 100_000, n),
            "mortgage_30y": rng.uniform(5.0, 8.0, n),
            "fed_funds": rng.uniform(4.0, 6.0, n),
            "cpi": rng.uniform(280, 320, n),
            "unemployment": rng.uniform(3.0, 6.0, n),
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
            "latitude": 42.886 + rng.normal(0, 0.02, n),
            "longitude": -78.878 + rng.normal(0, 0.02, n),
        }
    )


def _random_property(rng: np.random.RandomState) -> dict:
    """A plausible incoming property with every comp-vector feature."""
    prop = {
        "total_living_area": rng.randint(800, 4000),
        "first_floor_area": rng.randint(500, 2500),
        "second_floor_area": rng.randint(300, 2000),
        "beds": rng.randint(2, 6),
        "kitchens": 1,
        "stories": rng.choice([1, 2]),
        "lot_frontage": rng.uniform(50, 150),
        "lot_depth": rng.uniform(80, 200),
        "lot_acres": rng.uniform(0.1, 0.5),
        "year_built": rng.randint(1950, 2020),
        "assessed_value": rng.uniform(80_000, 400_000),
        "land_value": rng.uniform(20_000, 100_000),
        "mortgage_30y": rng.uniform(5.0, 8.0),
        "fed_funds": rng.uniform(4.0, 6.0),
        "cpi": rng.uniform(280, 320),
        "unemployment": rng.uniform(3.0, 6.0),
    }
    for name in GEO_VELOCITY_FEATURES:
        prop[name] = rng.uniform(5, 300) if "vol" in name else rng.uniform(80, 200)
    prop["latitude"] = 42.886 + rng.normal(0, 0.02)
    prop["longitude"] = -78.878 + rng.normal(0, 0.02)
    return prop


def _publish_index(n: int, base_path: str):
    """Build and publish the index artifacts the way the pipeline does."""
    sales = _make_synthetic_sales(n)
    features_df = _make_synthetic_features(n)

    comp_finder = CompFinderFeatures()
    comp_finder.fit(features_df)
    scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
    comp_finder.set_scaler(scaler)

    comp_vectors = comp_finder.build_comp_vector(features_df)
    index = BallTree(comp_vectors, metric="euclidean")

    lookup = sales[UNIFIED_COLUMNS].copy()
    lookup["comp_vector_index"] = range(n)

    store = ArtifactStore(
        project=INDEX_PROJECT, version=INDEX_VERSION, base_path=base_path
    )
    buf = io.BytesIO()
    joblib.dump(index, buf)
    store.save(INDEX_ARTIFACT_NAME, buf.getvalue())
    store.save(LOOKUP_ARTIFACT_NAME, lookup)
    store.save(SCALER_ARTIFACT_NAME, scaler)

    return sales, features_df, store


@pytest.fixture
def indexed(tmp_path) -> dict:
    """Publish a 50-sale index to a temp store; return inputs + finder."""
    base_path = str(tmp_path)
    sales, features_df, store = _publish_index(50, base_path)
    return {
        "sales": sales,
        "features_df": features_df,
        "store": store,
        "finder": CompFinder(store=store),
    }


@pytest.fixture
def geo_passthrough(monkeypatch):
    """Make the geo velocity derived features pass their input column
    through unchanged.

    The real derived functions resolve H3 cells against the geographic
    feature store (an optional dependency of this package); in a minimal
    environment they are None, and in tests we don't want the result to
    depend on which is the case.  Pass-through keeps the tests
    deterministic in both environments.
    """
    for name in GEO_VELOCITY_FEATURES:
        feature = CompFinderFeatures.__dict__[name]
        fn = (lambda name: (lambda df: df[name]))(name)
        monkeypatch.setattr(feature, "_derived_feature_fn", fn)


# =============================================================================
# Loading
# =============================================================================


class TestCompFinderLoading:
    """Tests for artifact loading from ArtifactStore."""

    def test_loads_index_lookup_and_scaler(self, indexed):
        finder = indexed["finder"]
        assert finder.population_size == 50
        assert len(finder._lookup_df) == 50
        assert list(finder._lookup_df["comp_vector_index"]) == list(range(50))
        assert finder.features._comp_vector_scaler is not None

    def test_accepts_base_path(self, tmp_path):
        base_path = str(tmp_path / "artifacts")
        _publish_index(12, base_path)
        finder = CompFinder(base_path=base_path)
        assert finder.population_size == 12

    def test_raises_for_missing_artifacts(self, tmp_path):
        base_path = str(tmp_path / "empty")
        with pytest.raises(Exception):
            CompFinder(base_path=base_path)


# =============================================================================
# find_comps: count and configuration
# =============================================================================


class TestFindCompsCount:
    """Tests for result count and k configuration."""

    def test_returns_exactly_k_by_default(self, indexed, geo_passthrough):
        finder = indexed["finder"]
        rng = np.random.RandomState(0)
        results = finder.find_comps(_random_property(rng))
        assert len(results) == DEFAULT_K == 5

    def test_k_is_configurable(self, indexed, geo_passthrough):
        finder = indexed["finder"]
        rng = np.random.RandomState(0)
        prop = _random_property(rng)
        assert len(finder.find_comps(prop, k=3)) == 3
        assert len(finder.find_comps(prop, k=1)) == 1
        assert len(finder.find_comps(prop, k=20)) == 20

    def test_k_larger_than_population_returns_full_population(self, tmp_path, geo_passthrough):
        base_path = str(tmp_path)
        sales, _, _ = _publish_index(10, base_path)
        finder = CompFinder(base_path=base_path)
        assert finder.population_size == 10

        rng = np.random.RandomState(0)
        results = finder.find_comps(_random_property(rng), k=100)
        assert len(results) == 10
        assert [r["rank"] for r in results] == list(range(1, 11))
        # All 10 distinct sales, nearest first
        assert len({r["parcel_id"] for r in results}) == 10

    def test_invalid_k_raises(self, indexed, geo_passthrough):
        finder = indexed["finder"]
        rng = np.random.RandomState(0)
        prop = _random_property(rng)
        with pytest.raises(ValueError):
            finder.find_comps(prop, k=0)
        with pytest.raises(ValueError):
            finder.find_comps(prop, k=-3)
        with pytest.raises(ValueError):
            finder.find_comps(prop, k=True)

    def test_accepts_dataframe_input(self, indexed, geo_passthrough):
        finder = indexed["finder"]
        rng = np.random.RandomState(0)
        prop = _random_property(rng)
        from_dict = finder.find_comps(prop, k=3)
        from_df = finder.find_comps(pd.DataFrame([prop]), k=3)
        assert len(from_df) == 3
        assert from_dict[0]["parcel_id"] == from_df[0]["parcel_id"]

    def test_missing_required_features_raise(self, indexed, geo_passthrough):
        finder = indexed["finder"]
        rng = np.random.RandomState(0)
        prop = _random_property(rng)
        del prop["beds"]
        with pytest.raises(ValueError):
            finder.find_comps(prop)

    def test_non_dict_input_raises_type_error(self, indexed):
        finder = indexed["finder"]
        with pytest.raises(TypeError):
            finder.find_comps(["not", "a", "property"])  # type: ignore[arg-type]


# =============================================================================
# find_comps: ranking and similarity
# =============================================================================


class TestFindCompsRanking:
    """Tests for ranking order and similarity scores."""

    def test_ranks_nearest_first(self, indexed, geo_passthrough):
        finder = indexed["finder"]
        rng = np.random.RandomState(7)
        for _ in range(5):
            results = finder.find_comps(_random_property(rng), k=5)
            distances = [r["distance"] for r in results]
            similarities = [r["similarity"] for r in results]
            assert distances == sorted(distances)
            assert similarities == sorted(similarities, reverse=True)
            assert [r["rank"] for r in results] == [1, 2, 3, 4, 5]

    def test_similarity_bounded_and_monotonic(self, indexed, geo_passthrough):
        finder = indexed["finder"]
        rng = np.random.RandomState(8)
        for _ in range(5):
            for r in finder.find_comps(_random_property(rng), k=5):
                assert 0 < r["similarity"] <= 1.0
                assert r["similarity"] == pytest.approx(
                    1.0 / (1.0 + r["distance"])
                )

    def test_result_fields(self, indexed, geo_passthrough):
        finder = indexed["finder"]
        rng = np.random.RandomState(9)
        r = finder.find_comps(_random_property(rng), k=1)[0]
        for key in (
            "rank",
            "address",
            "city",
            "zip_code",
            "parcel_id",
            "print_key",
            "sale_price",
            "sale_date",
            "latitude",
            "longitude",
            "distance",
            "similarity",
        ):
            assert key in r
        assert isinstance(r["sale_date"], str)  # ISO 8601 string
        assert 100_000 < r["sale_price"] < 600_000
        # Metadata actually matches the historical sale in the lookup
        match = finder._lookup_df[
            finder._lookup_df["parcel_id"] == r["parcel_id"]
        ].iloc[0]
        assert r["address"] == match["address"]
        assert r["sale_price"] == match["sale_price"]

    def test_identical_property_returns_itself_as_top_match(
        self, indexed, geo_passthrough
    ):
        """A property identical to a historical sale returns that sale
        as rank 1 with the maximum possible similarity (distance 0)."""
        finder = indexed["finder"]
        features_df = indexed["features_df"]
        sales = indexed["sales"]

        for sale_row in (0, 17, 49):
            prop = features_df.iloc[sale_row].to_dict()
            results = finder.find_comps(prop, k=5)
            assert results[0]["distance"] == pytest.approx(0.0)
            assert results[0]["similarity"] == pytest.approx(1.0)
            expected = sales.iloc[sale_row]
            assert results[0]["parcel_id"] == expected["parcel_id"]
            assert results[0]["sale_price"] == expected["sale_price"]

    def test_identical_property_via_dataframe(
        self, indexed, geo_passthrough
    ):
        finder = indexed["finder"]
        features_df = indexed["features_df"]
        prop = features_df.iloc[5]
        results = finder.find_comps(pd.DataFrame([prop]), k=3)
        assert results[0]["similarity"] == pytest.approx(1.0)


# =============================================================================
# Regression: existing pipeline logic untouched
# =============================================================================


class TestCompFinderStandalone:
    """CompFinder does not alter pipeline / model / features behavior."""

    def test_pipeline_still_imports_and_runs_shape_checks(self):
        from comp_finder.pipeline import CompSearchPipeline

        pipeline = CompSearchPipeline()
        pipeline.initialize()
        features_df = _make_synthetic_features(20)
        pipeline.features.fit(features_df)
        scaler = StandardScaler().fit(features_df[ALL_VECTOR_FEATURES].values)
        pipeline.features.set_scaler(scaler)
        vectors = pipeline.features.build_comp_vector(features_df)
        pipeline._index = BallTree(vectors, metric="euclidean")
        assert vectors.shape == (20, len(ALL_VECTOR_FEATURES))
