"""Feature definitions for comp-finder.

Defines the 28 features that make up the comp vector (physical + macro +
geographic velocity). The comp-finder does NOT redefine the regressor's
features — it imports the geo velocity derived functions from
`expected_transaction_price.sdk.features` and re-declares them here
so that `FeatureSet.transform()` produces the same columns.

Categorical features are excluded from the comp vector by design.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geronimo.features import FeatureSet, Feature

# ---------------------------------------------------------------------------
# Feature family definitions (mirrors expected-transaction-price/sdk/features.py)
# ---------------------------------------------------------------------------

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

MACRO_FEATURES = [
    "mortgage_30y",
    "fed_funds",
    "cpi",
    "unemployment",
]

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

# Categorical features excluded from the comp vector.
# Reason: these are high-cardinality nominal attributes (building_style,
# exterior_wall, heat_type, central_air, basement_type, school_district).
# Including them as one-hot encodings at full Euclidean weight would cause
# a single mismatched category to dominate the distance, drowning out the
# signal from physical, macro, and geographic features. Instead, they should
# be used as pre-filter matching criteria (e.g., "same school district" is a
# binary filter, not a distance contribution).
CATEGORICAL_FEATURES = [
    "building_style",
    "exterior_wall",
    "heat_type",
    "central_air",
    "basement_type",
    "school_district",
]

# All features used in the comp vector (physical + macro + geo velocity).
ALL_VECTOR_FEATURES = PHYSICAL_FEATURES + MACRO_FEATURES + GEO_VELOCITY_FEATURES

# Default per-family weight multipliers (applied after standardization).
DEFAULT_FAMILY_WEIGHTS = {
    "physical": 1.0,
    "macro": 1.0,
    "geographic": 1.0,
}

# ArtifactStore keys for the persisted scaler.
SCALER_ARTIFACT_NAME = "comp_vector_scaler"
SCALER_PROJECT = "comp-finder"
SCALER_VERSION = "1.0.0"


def get_feature_categories() -> dict[str, str]:
    """Return a mapping of excluded categorical features to their handling.

    Returns:
        Dict mapping each categorical feature name to a string explaining
        why it is excluded from the comp vector and how it should be used
        instead (as a matching filter).
    """
    return {
        "building_style": (
            "Excluded: high-cardinality nominal. Use as a binary match "
            "filter (same style = closer) rather than a distance component."
        ),
        "exterior_wall": (
            "Excluded: high-cardinality nominal. Use as a binary match filter."
        ),
        "heat_type": (
            "Excluded: high-cardinality nominal. Use as a binary match filter."
        ),
        "central_air": (
            "Excluded: binary categorical. Use as a binary match filter."
        ),
        "basement_type": (
            "Excluded: high-cardinality nominal. Use as a binary match filter."
        ),
        "school_district": (
            "Excluded: very high-cardinality nominal. Use as a hard match "
            "filter (same district) — distance-based comparison across "
            "district boundaries is meaningless."
        ),
    }


def get_comp_vector_info() -> dict:
    """Return metadata about the comp vector structure.

    Returns:
        Dict with feature lists, vector length, family weights,
        and categorical handling documentation.
    """
    return {
        "all_features": list(ALL_VECTOR_FEATURES),
        "physical_features": list(PHYSICAL_FEATURES),
        "macro_features": list(MACRO_FEATURES),
        "geo_velocity_features": list(GEO_VELOCITY_FEATURES),
        "excluded_categorical": list(CATEGORICAL_FEATURES),
        "vector_length": len(ALL_VECTOR_FEATURES),
        "family_weights": dict(DEFAULT_FAMILY_WEIGHTS),
        "categorical_handling": get_feature_categories(),
    }

# Import geo velocity derived feature functions from the existing
# expected-transaction-price feature set so the comp-finder uses the
# same H3-lookup logic.  Import lazily so the comp-finder can be
# used standalone (without the expected-transaction-price project)
# — the geo features will simply return NaN.
try:
    from expected_transaction_price.sdk.features import (
        _geo_vol_r1_w90,
        _geo_ppsf_r1_w90,
        _geo_vol_r1_w365,
        _geo_ppsf_r1_w365,
        _geo_vol_r5_w90,
        _geo_ppsf_r5_w90,
        _geo_vol_r5_w180,
        _geo_ppsf_r5_w180,
        _geo_vol_r10_w180,
        _geo_ppsf_r10_w180,
        _geo_vol_r10_w365,
        _geo_ppsf_r10_w365,
    )
except ImportError:
    _geo_vol_r1_w90 = None
    _geo_ppsf_r1_w90 = None
    _geo_vol_r1_w365 = None
    _geo_ppsf_r1_w365 = None
    _geo_vol_r5_w90 = None
    _geo_ppsf_r5_w90 = None
    _geo_vol_r5_w180 = None
    _geo_ppsf_r5_w180 = None
    _geo_vol_r10_w180 = None
    _geo_ppsf_r10_w180 = None
    _geo_vol_r10_w365 = None
    _geo_ppsf_r10_w365 = None


class CompFinderFeatures(FeatureSet):
    """Feature engineering for comp-finder.

    Declares the 28 features that make up the comp vector:
    - 12 physical features (numeric)
    - 4 macro-economic features (numeric)
    - 12 geographic velocity features (derived from feature store)

    Categorical features are excluded from the comp vector and should
    be used as pre-filter matching criteria instead.

    The geo velocity derived functions are imported from
    `expected_transaction_price.sdk.features` so that the comp-finder
    uses the exact same H3-lookup logic as the regressor.
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

    # --- Macro-Economic Features (FRED) ---
    mortgage_30y = Feature(dtype='numeric')
    fed_funds = Feature(dtype='numeric')
    cpi = Feature(dtype='numeric')
    unemployment = Feature(dtype='numeric')

    # --- Geographic Velocity (from Feature Store) ---
    # These are non-required so that the comp-finder can be tested
    # or used without the geographic feature store deployed.
    # When the store is unavailable, these features return NaN.
    geo_vol_r1_w90 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r1_w90,
        required=False,
        description="Sales volume within 1mi, 90-day (from feature store)",
    )
    geo_ppsf_r1_w90 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r1_w90,
        required=False,
        description="Median PPSF within 1mi, 90-day (from feature store)",
    )
    geo_vol_r1_w365 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r1_w365,
        required=False,
        description="Sales volume within 1mi, annual (from feature store)",
    )
    geo_ppsf_r1_w365 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r1_w365,
        required=False,
        description="Median PPSF within 1mi, annual (from feature store)",
    )
    geo_vol_r5_w90 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r5_w90,
        required=False,
        description="Sales volume within 5mi, 90-day (from feature store)",
    )
    geo_ppsf_r5_w90 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r5_w90,
        required=False,
        description="Median PPSF within 5mi, 90-day (from feature store)",
    )
    geo_vol_r5_w180 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r5_w180,
        required=False,
        description="Sales volume within 5mi, 180-day (from feature store)",
    )
    geo_ppsf_r5_w180 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r5_w180,
        required=False,
        description="Median PPSF within 5mi, 180-day (from feature store)",
    )
    geo_vol_r10_w180 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r10_w180,
        required=False,
        description="Sales volume within 10mi, 180-day (from feature store)",
    )
    geo_ppsf_r10_w180 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r10_w180,
        required=False,
        description="Median PPSF within 10mi, 180-day (from feature store)",
    )
    geo_vol_r10_w365 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_vol_r10_w365,
        required=False,
        description="Sales volume within 10mi, annual (from feature store)",
    )
    geo_ppsf_r10_w365 = Feature(
        dtype='derived',
        source_columns=['latitude', 'longitude'],
        derived_feature_fn=_geo_ppsf_r10_w365,
        required=False,
        description="Median PPSF within 10mi, annual (from feature store)",
    )

    # Internal state for comp vector.
    _comp_vector_scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame):
        """Fit features and optionally fit the comp vector scaler."""
        super().fit(df)

        # Fit the comp vector scaler on the feature subset.
        if self._comp_vector_scaler is not None:
            vector_df = df[ALL_VECTOR_FEATURES].dropna()
            if len(vector_df) > 0:
                self._comp_vector_scaler.fit(vector_df.values)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input data."""
        return super().transform(df)

    def build_comp_vector(
        self,
        df: pd.DataFrame,
        family_weights: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """Build a fixed-length standardized comp vector from a FeatureSet output.

        Takes the DataFrame produced by transform() and extracts the comp
        vector features, standardizes them (using the fitted scaler),
        applies per-family weights, and returns a 1-D numpy array.

        Args:
            df: DataFrame produced by FeatureSet.transform() containing
                columns from ALL_VECTOR_FEATURES.
            family_weights: Optional per-family weight dict with keys
                "physical", "macro", "geographic". Defaults to 1.0 for all.

        Returns:
            A 1-D numpy array of fixed length equal to len(ALL_VECTOR_FEATURES),
            suitable for distance-based nearest-neighbor search.

        Raises:
            ValueError: If required features are missing from df.
        """
        if family_weights is None:
            family_weights = dict(DEFAULT_FAMILY_WEIGHTS)

        missing = [f for f in ALL_VECTOR_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(
                f"build_comp_vector requires features {missing} "
                f"but they were not found in the input DataFrame. "
                f"Ensure the FeatureSet.transform() output includes all "
                f"expected feature columns."
            )

        if len(df) == 0:
            raise ValueError(
                "build_comp_vector received an empty DataFrame. "
                "Provide at least one row of features."
            )

        # Extract feature columns in the defined order.
        vector = df[ALL_VECTOR_FEATURES].values.astype(np.float64)

        # Standardize if a scaler is fitted.
        if self._comp_vector_scaler is not None:
            vector = self._comp_vector_scaler.transform(vector)

        # Squeeze to 1-D for single-row inputs.
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector.squeeze(axis=0)

        # Apply per-family weights.
        offset = 0
        for family_name, feature_names, weight in [
            ("physical", PHYSICAL_FEATURES, family_weights.get("physical", 1.0)),
            ("macro", MACRO_FEATURES, family_weights.get("macro", 1.0)),
            ("geographic", GEO_VELOCITY_FEATURES, family_weights.get("geographic", 1.0)),
        ]:
            n_features = len(feature_names)
            vector[offset : offset + n_features] *= weight
            offset += n_features

        return vector

    def set_scaler(self, scaler: StandardScaler):
        """Set a pre-fitted scaler for comp vector standardization.

        Call this after loading a scaler from ArtifactStore so that
        `build_comp_vector()` will standardize on transform.

        Args:
            scaler: A fitted sklearn StandardScaler.
        """
        self._comp_vector_scaler = scaler
