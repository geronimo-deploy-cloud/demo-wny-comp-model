"""Feature definitions for expected-transaction-price.

Features are organized into three groups:
  1. Physical — numeric measurements of the property
  2. Derived — computed from raw columns (property_age, log_area, assessment_ratio)
  3. Geographic Velocity — precomputed from the geographic feature store (H3 lookup)
  4. Categorical — building style, condition, materials, school district

Geographic velocity features come from the deployed Geographic Feature Store
(ArtifactStore project="geographic-feature-store"), NOT from inline BallTree
computation. This ensures parity between training and live inference.
"""

import logging
from datetime import datetime
from typing import Optional

import h3
import numpy as np
import pandas as pd

from geronimo.artifacts import ArtifactStore
from geronimo.features import FeatureSet, Feature

logger = logging.getLogger(__name__)


# =============================================================================
# Canonical feature lists — single source of truth for all projects.
#
# Any project that consumes property data should import these lists rather
# than re-declaring feature names.  If a feature is added or removed here,
# every consumer picks it up automatically.
# =============================================================================

# Physical features used in the comp vector (numeric measurements).
PHYSICAL_FEATURES = [
    "total_living_area",
    "first_floor_area",
    "second_floor_area",
    "beds",
    "kitchens",
    "stories",
    "lot_frontage",
    "lot_depth",
    "lot_acres",
    "year_built",
    "assessed_value",
    "land_value",
]

# Macro-economic features from FRED.
MACRO_FEATURES = [
    "mortgage_30y",
    "fed_funds",
    "cpi",
    "unemployment",
]

# Geographic velocity features from the feature store (derived).
GEO_VELOCITY_FEATURES = [
    "geo_vol_r1_w90",
    "geo_ppsf_r1_w90",
    "geo_vol_r1_w365",
    "geo_ppsf_r1_w365",
    "geo_vol_r5_w90",
    "geo_ppsf_r5_w90",
    "geo_vol_r5_w180",
    "geo_ppsf_r5_w180",
    "geo_vol_r10_w180",
    "geo_ppsf_r10_w180",
    "geo_vol_r10_w365",
    "geo_ppsf_r10_w365",
]

# Categorical features excluded from the comp vector (high-cardinality nominal).
CATEGORICAL_FEATURES = [
    "building_style",
    "exterior_wall",
    "heat_type",
    "central_air",
    "basement_type",
    "school_district",
]

# Combined comp-vector feature list (physical + macro + geo velocity).
ALL_VECTOR_FEATURES = PHYSICAL_FEATURES + MACRO_FEATURES + GEO_VELOCITY_FEATURES

# Velocity artifact configs: (radius_miles, lookback_days, artifact_name)
# Selected combos that span hyperlocal → metro, short → annual
GEO_VELOCITY_CONFIGS = [
    (1.0, 90, "geo_velocity_r1_w90"),
    (1.0, 365, "geo_velocity_r1_w365"),
    (5.0, 90, "geo_velocity_r5_w90"),
    (5.0, 180, "geo_velocity_r5_w180"),
    (10.0, 180, "geo_velocity_r10_w180"),
    (10.0, 365, "geo_velocity_r10_w365"),
]

# Module-level cache for the velocity lookups (loaded once per process)
_geo_velocity_cache: Optional[dict[str, pd.DataFrame]] = None
_geo_h3_resolution: int = 8


def _load_geo_velocity_store() -> dict[str, pd.DataFrame]:
    """Load velocity grids from the geographic feature store ArtifactStore.

    Returns:
        Dict mapping artifact_name -> DataFrame indexed by h3_index.
    """
    global _geo_velocity_cache, _geo_h3_resolution

    if _geo_velocity_cache is not None:
        return _geo_velocity_cache

    logger.info("Loading geographic velocity data from ArtifactStore...")
    store = ArtifactStore(project="geographic-feature-store", version="1.0.0")

    try:
        config = store.get("features_config")
        _geo_h3_resolution = config.get("h3_resolution", 8)
        logger.info("  H3 resolution: %d", _geo_h3_resolution)
    except Exception as e:
        logger.warning("  Could not load features_config: %s. Using default res=8.", e)

    cache = {}
    for _, _, artifact_name in GEO_VELOCITY_CONFIGS:
        try:
            vel_df = store.get(artifact_name)
            # Index by h3_index for O(1) lookup
            vel_df = vel_df.set_index("h3_index")
            cache[artifact_name] = vel_df
            nonzero = int((vel_df["sales_volume"] > 0).sum())
            logger.info("  %s: %d cells, %d with sales", artifact_name, len(vel_df), nonzero)
        except Exception as e:
            logger.warning("  Failed to load %s: %s", artifact_name, e)

    _geo_velocity_cache = cache
    logger.info("Loaded %d velocity artifacts from geographic feature store", len(cache))
    return cache


def _h3_lookup(df: pd.DataFrame, artifact_name: str, metric: str) -> pd.Series:
    """Look up precomputed velocity values for each row via H3 cell.

    Args:
        df: DataFrame with latitude, longitude columns.
        artifact_name: Which velocity artifact to query.
        metric: "sales_volume" or "median_ppsf".

    Returns:
        Series aligned to df.index with the looked-up values.
    """
    cache = _load_geo_velocity_store()
    vel_df = cache.get(artifact_name)

    if vel_df is None or vel_df.empty:
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Map each row to its H3 cell
    resolution = _geo_h3_resolution
    cells = df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], resolution)
        if pd.notna(r["latitude"]) and pd.notna(r["longitude"])
        else None,
        axis=1,
    )

    # Vectorized lookup
    result = cells.map(
        lambda cell: vel_df.loc[cell, metric] if cell and cell in vel_df.index else np.nan
    )

    return result.astype(float)


# =============================================================================
# Derived feature functions
# =============================================================================

def _property_age(df):
    """Years since construction."""
    current_year = datetime.now().year
    return current_year - pd.to_numeric(df["year_built"], errors="coerce")


def _log_total_living_area(df):
    """Log-transform living area to normalize right-skewed distribution."""
    return np.log1p(pd.to_numeric(df["total_living_area"], errors="coerce").fillna(0))


def _assessment_ratio(df):
    """Ratio of improvements value to land value (assessed_value / land_value).
    
    High ratio = more improvements relative to land.
    Low ratio = land-heavy (teardowns, vacant lots).
    """
    assessed = pd.to_numeric(df["assessed_value"], errors="coerce").fillna(0)
    land = pd.to_numeric(df["land_value"], errors="coerce").fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(land > 0, assessed / land, 0.0)
    return pd.Series(ratio, index=df.index)


def _bathrooms(df):
    """Combined bathroom count: full baths + 0.5 * half baths."""
    baths = pd.to_numeric(df.get("baths", 0), errors="coerce").fillna(0)
    half = pd.to_numeric(df.get("half_baths", 0), errors="coerce").fillna(0)
    return baths + 0.5 * half


# --- Geographic velocity feature functions (H3 lookup from store) ---

def _geo_vol_r1_w90(df): return _h3_lookup(df, "geo_velocity_r1_w90", "sales_volume")
def _geo_ppsf_r1_w90(df): return _h3_lookup(df, "geo_velocity_r1_w90", "median_ppsf")
def _geo_vol_r1_w365(df): return _h3_lookup(df, "geo_velocity_r1_w365", "sales_volume")
def _geo_ppsf_r1_w365(df): return _h3_lookup(df, "geo_velocity_r1_w365", "median_ppsf")
def _geo_vol_r5_w90(df): return _h3_lookup(df, "geo_velocity_r5_w90", "sales_volume")
def _geo_ppsf_r5_w90(df): return _h3_lookup(df, "geo_velocity_r5_w90", "median_ppsf")
def _geo_vol_r5_w180(df): return _h3_lookup(df, "geo_velocity_r5_w180", "sales_volume")
def _geo_ppsf_r5_w180(df): return _h3_lookup(df, "geo_velocity_r5_w180", "median_ppsf")
def _geo_vol_r10_w180(df): return _h3_lookup(df, "geo_velocity_r10_w180", "sales_volume")
def _geo_ppsf_r10_w180(df): return _h3_lookup(df, "geo_velocity_r10_w180", "median_ppsf")
def _geo_vol_r10_w365(df): return _h3_lookup(df, "geo_velocity_r10_w365", "sales_volume")
def _geo_ppsf_r10_w365(df): return _h3_lookup(df, "geo_velocity_r10_w365", "median_ppsf")


# =============================================================================
# Feature Definitions
# =============================================================================


class ExpectedTransactionPriceFeatures(FeatureSet):
    """Feature engineering for expected-transaction-price (Rochester).

    Physical + derived + geographic velocity (from feature store) + categorical.
    """

    # --- Physical Features (Numeric) ---
    total_living_area = Feature(dtype='numeric')
    first_floor_area = Feature(dtype='numeric')
    second_floor_area = Feature(dtype='numeric')
    beds = Feature(dtype='numeric')
    kitchens = Feature(dtype='numeric')
    stories = Feature(dtype='numeric')
    lot_frontage = Feature(dtype='numeric')
    lot_depth = Feature(dtype='numeric')
    lot_acres = Feature(dtype='numeric')
    year_built = Feature(dtype='numeric')

    # --- Assessment Features (Numeric) ---
    assessed_value = Feature(dtype='numeric')
    land_value = Feature(dtype='numeric')

    # --- Derived Features ---
    bathrooms = Feature(
        dtype='derived',
        source_columns=['baths', 'half_baths'],
        derived_feature_fn=_bathrooms,
        description="Combined full + half bath count",
    )
    property_age = Feature(
        dtype='derived',
        source_columns=['year_built'],
        derived_feature_fn=_property_age,
        description="Years since construction",
    )
    log_total_living_area = Feature(
        dtype='derived',
        source_columns=['total_living_area'],
        derived_feature_fn=_log_total_living_area,
        description="Log-transformed total living area",
    )
    assessment_ratio = Feature(
        dtype='derived',
        source_columns=['assessed_value', 'land_value'],
        derived_feature_fn=_assessment_ratio,
        description="Improvements-to-land value ratio",
    )

    # --- Geographic Velocity (from Feature Store) ---
    # 1-mile radius: hyperlocal
    geo_vol_r1_w90 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r1_w90,
        description="Sales volume within 1mi, 90-day (from feature store)",
    )
    geo_ppsf_r1_w90 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r1_w90,
        description="Median PPSF within 1mi, 90-day (from feature store)",
    )
    geo_vol_r1_w365 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r1_w365,
        description="Sales volume within 1mi, annual (from feature store)",
    )
    geo_ppsf_r1_w365 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r1_w365,
        description="Median PPSF within 1mi, annual (from feature store)",
    )

    # 5-mile radius: neighborhood
    geo_vol_r5_w90 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r5_w90,
        description="Sales volume within 5mi, 90-day (from feature store)",
    )
    geo_ppsf_r5_w90 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r5_w90,
        description="Median PPSF within 5mi, 90-day (from feature store)",
    )
    geo_vol_r5_w180 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r5_w180,
        description="Sales volume within 5mi, 180-day (from feature store)",
    )
    geo_ppsf_r5_w180 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r5_w180,
        description="Median PPSF within 5mi, 180-day (from feature store)",
    )

    # 10-mile radius: metro-level context
    geo_vol_r10_w180 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r10_w180,
        description="Sales volume within 10mi, 180-day (from feature store)",
    )
    geo_ppsf_r10_w180 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r10_w180,
        description="Median PPSF within 10mi, 180-day (from feature store)",
    )
    geo_vol_r10_w365 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r10_w365,
        description="Sales volume within 10mi, annual (from feature store)",
    )
    geo_ppsf_r10_w365 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r10_w365,
        description="Median PPSF within 10mi, annual (from feature store)",
    )

    # --- Macro-Economic Features (FRED) ---
    mortgage_30y = Feature(dtype='numeric')
    fed_funds = Feature(dtype='numeric')
    cpi = Feature(dtype='numeric')
    unemployment = Feature(dtype='numeric')

    # --- Categorical Features ---
    building_style = Feature(dtype='categorical')
    exterior_wall = Feature(dtype='categorical')
    heat_type = Feature(dtype='categorical')
    central_air = Feature(dtype='categorical')
    basement_type = Feature(dtype='categorical')
    school_district = Feature(dtype='categorical')

    def fit(self, df: pd.DataFrame):
        import pandas as pd
        # Call base fit
        super().fit(df)

        # Save specific category structures for inference so unseen strings become safe NaNs
        self._fitted_categories = {}
        for feature in self.categorical_features:
            if feature.name in df.columns:
                cat_series = df[feature.name].fillna("Missing").astype(str).astype('category')
                self._fitted_categories[feature.name] = cat_series.cat.categories
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override transform to properly cast dtypes based on Feature definitions.

        The underlying geronimo FeatureSet.transform returns raw numpy arrays for features
        without transformers/encoders, stripping Pandas categorical types which XGBoost
        explicitly requires.
        """
        # Call base geronimo transform to run derivations/transformers
        result = super().transform(df)

        # Force strict typing based on our declarative schema
        for feature in self.categorical_features:
            if feature.name in result.columns:
                series = result[feature.name].fillna("Missing").astype(str)
                # If we've fitted categories, strictly enforce them so unseen labels -> NaN (safe for XGBoost)
                if hasattr(self, "_fitted_categories") and feature.name in self._fitted_categories:
                    import pandas as pd
                    cat_dtype = pd.CategoricalDtype(categories=self._fitted_categories[feature.name], ordered=False)
                    result[feature.name] = series.astype(cat_dtype)
                else:
                    result[feature.name] = series.astype('category')

        for feature in self.numeric_features:
            if feature.name in result.columns:
                result[feature.name] = pd.to_numeric(result[feature.name], errors='coerce')

        return result
