"""Model definition for comp-finder.

Two classes live here (the Geronimo SDK standardizes on model.py):

- ``CompFinderModel`` — the Geronimo Model.  Integrates with
  expected-transaction-price's feature set to produce a standardized
  comp vector; persists the comp vector scaler via ArtifactStore during
  ``save()`` and restores it during ``load()``.
- ``CompFinder`` — the query interface for the comp-vector
  nearest-neighbor index.  Loads the index / lookup / scaler artifacts
  from ArtifactStore and exposes ``find_comps(property_features, k)``.
"""

import math
from io import BytesIO
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd

from geronimo.models import Model, HyperParams
from geronimo.artifacts import ArtifactStore
from .features import (
    CompFinderFeatures,
    SCALER_ARTIFACT_NAME,
    SCALER_PROJECT,
    SCALER_VERSION,
    ALL_VECTOR_FEATURES,
    DEFAULT_FAMILY_WEIGHTS,
    comp_feature_template,
    ensure_feature_inputs,
    get_comp_vector_info,
)
from ..pipeline import (
    INDEX_ARTIFACT_NAME,
    INDEX_PROJECT,
    INDEX_VERSION,
    LOOKUP_ARTIFACT_NAME,
)
from sklearn.preprocessing import StandardScaler


class CompFinderModel(Model):
    """ML model for comp-finder.

    Uses declarative features for transformation and ArtifactStore for
    persistence. The comp vector scaler is saved alongside the model
    so that inference can standardize features identically to training.
    """

    name = "comp-finder"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self.estimator: Optional[Any] = None
        self.features: Optional[CompFinderFeatures] = None
        self._is_fitted = False

    def train(self) -> dict:
        """Train the model.

        Loads training data sources, joins them, fits features, and trains estimator.

        Returns:
            Training metrics dict
        """
        from .data_sources import training_sources

        if not training_sources:
            raise ValueError("No training_* DataSources defined in data_sources.py")

        # Load and join training data sources
        df = training_sources[0].load()
        for source in training_sources[1:]:
            source_df = source.load()
            if source.join_spec:
                df = df.merge(
                    source_df,
                    left_on=source.join_spec.left_on,
                    right_on=source.join_spec.right_on,
                    how=source.join_spec.how,
                )

        # Initialize and fit features (includes comp vector scaler)
        self.features = CompFinderFeatures()
        # X = self.features.fit_transform(df)

        # TODO: Initialize and train your estimator
        # self.estimator = ...
        # self.estimator.fit(X, y)

        self._is_fitted = True

        return {
            "n_samples": len(df),
        }

    def predict(self, X) -> np.ndarray:
        """Predict using the trained model.

        Args:
            X: Feature array or DataFrame

        Returns:
            Predictions array
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.features.feature_names)
        else:
            df = X

        # Transform using fitted features
        X_transformed = self.features.transform(df)
        return self.estimator.predict(X_transformed)

    def build_comp_vector(
        self,
        df: pd.DataFrame,
        family_weights: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """Build a standardized comp vector from a FeatureSet output.

        Convenience method that delegates to `self.features.build_comp_vector()`.
        The scaler is automatically loaded from ArtifactStore if not yet fitted.

        Args:
            df: DataFrame produced by FeatureSet.transform().
            family_weights: Optional per-family weight dict. Defaults to 1.0.

        Returns:
            A 1-D numpy array of fixed length equal to len(ALL_VECTOR_FEATURES).
        """
        if self.features is None:
            raise RuntimeError("Model not initialized. Call train() or load() first.")

        # Load scaler from store if not yet fitted.
        if self.features._comp_vector_scaler is None:
            try:
                store = ArtifactStore(project=SCALER_PROJECT, version=SCALER_VERSION)
                scaler = store.get(SCALER_ARTIFACT_NAME)
                self.features.set_scaler(scaler)
            except Exception:
                pass  # No scaler persisted yet; use raw values.

        return self.features.build_comp_vector(df, family_weights)

    def save(self, store: ArtifactStore) -> list[str]:
        """Save trained model and features to ArtifactStore.

        Also persists the comp vector scaler so that inference can
        standardize features identically to training.

        Args:
            store: ArtifactStore instance

        Returns:
            List of saved artifact paths
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Nothing to save.")

        paths = []

        # Save the trained estimator
        path = store.save(
            "estimator",
            self.estimator,
            artifact_type=type(self.estimator).__name__,
            tags={"model": self.name, "version": self.version},
        )
        paths.append(path)

        # Save the fitted features (includes transformers/scalers)
        path = store.save(
            "features",
            self.features,
            artifact_type="CompFinderFeatures",
            tags={"model": self.name, "version": self.version},
        )
        paths.append(path)

        # Save the comp vector scaler separately for standalone use.
        if self.features is not None and self.features._comp_vector_scaler is not None:
            comp_store = ArtifactStore(project=SCALER_PROJECT, version=SCALER_VERSION)
            comp_store.save(SCALER_ARTIFACT_NAME, self.features._comp_vector_scaler)
            paths.append(f"{SCALER_PROJECT}/{SCALER_VERSION}/{SCALER_ARTIFACT_NAME}")

        return paths

    def load(self, store: ArtifactStore) -> None:
        """Load trained model and features from ArtifactStore.

        Also loads the comp vector scaler so that `build_comp_vector()`
        will standardize on inference.

        Args:
            store: ArtifactStore instance
        """
        self.estimator = store.get("estimator")
        self.features = store.get("features")

        # Load the comp vector scaler from its dedicated store.
        try:
            comp_store = ArtifactStore(project=SCALER_PROJECT, version=SCALER_VERSION)
            scaler = comp_store.get(SCALER_ARTIFACT_NAME)
            self.features.set_scaler(scaler)
        except Exception:
            pass  # No scaler persisted yet; will use raw values.

        self._is_fitted = True

    @property
    def comp_vector_info(self) -> dict:
        """Return metadata about the comp vector structure."""
        return get_comp_vector_info()

    @property
    def is_fitted(self) -> bool:
        """Check if model is trained and ready for predictions."""
        return self._is_fitted


# ---------------------------------------------------------------------------
# CompFinder — query interface for the comp-vector nearest-neighbor index
# ---------------------------------------------------------------------------

#: Default number of comps to return from ``find_comps()``.
DEFAULT_K = 5


def _clean_scalar(value: Any) -> Any:
    """Convert a lookup-table cell to a JSON-friendly scalar.

    NaN/NaT become None; numpy scalars are unwrapped; everything else
    passes through unchanged.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class CompFinder:
    """Query the precomputed comp-vector index for nearest historical sales.

    On initialization, loads the BallTree index, the lookup table, and
    the fitted comp-vector scaler from the ``comp-finder`` ArtifactStore
    (project / version "1.0.0") — the same artifacts the index batch
    pipeline publishes and the same store the Ticket-1 scaler lives in.

    Args:
        store: Optional pre-built ArtifactStore.  When omitted, a store
            for project ``comp-finder`` version ``1.0.0`` is created
            (using ``base_path`` if given).
        base_path: Base directory for a local ArtifactStore (mainly
            useful in tests / local development).

    No HTTP or MCP surface is defined here — this is the query logic
    only, consumed directly (e.g. in tests or by a future endpoint
    ticket).
    """

    def __init__(
        self,
        store: Optional[ArtifactStore] = None,
        base_path: Optional[str] = None,
    ):
        if store is None:
            store = ArtifactStore(
                project=INDEX_PROJECT, version=INDEX_VERSION, base_path=base_path
            )
        self._store = store

        # The index artifact was saved as joblib-serialized bytes (see
        # CompSearchPipeline._publish); deserialize it here.
        index_bytes = self._store.get(INDEX_ARTIFACT_NAME)
        self._index = joblib.load(BytesIO(index_bytes))

        # The lookup table is stored as a DataFrame and comes back as one.
        self._lookup_df: pd.DataFrame = self._store.get(LOOKUP_ARTIFACT_NAME)

        # The fitted scaler from Ticket 1 (saved by CompFinderModel).
        scaler: StandardScaler = self._store.get(SCALER_ARTIFACT_NAME)
        self.features = CompFinderFeatures()
        # FeatureSet requires fit() before transform(); there are no
        # stateful transformers, so an empty frame declaring every
        # feature column is all that's needed to mark the set as fitted.
        self.features.fit(comp_feature_template())
        self.features.set_scaler(scaler)

    @property
    def population_size(self) -> int:
        """Number of historical sales in the index.

        ``find_comps(k)`` returns at most ``min(k, population_size)``
        results — a requested ``k`` larger than this is the clear reason
        for a shorter result list.
        """
        return len(self._lookup_df)

    def find_comps(
        self,
        property_features: Union[dict, pd.DataFrame],
        k: int = DEFAULT_K,
    ) -> list[dict]:
        """Find the k nearest historical comparable sales.

        The property's features are run through the same
        ``CompFinderFeatures`` transform + ``build_comp_vector()`` used
        to build the historical index, then the precomputed BallTree is
        queried for the k nearest historical sales.

        Args:
            property_features: A dict (one property) or a one-row
                DataFrame with the property's raw feature values — the
                same columns a historical sales record carries (physical
                attributes, macro indicators, latitude/longitude, and
                optionally the geo velocity features).
            k: Number of comps to return.  Defaults to ``DEFAULT_K``
                (5).  At most ``min(k, population_size)`` results are
                returned; requesting more than the historical population
                yields the full population, nearest first.

        Returns:
            A ranked list (nearest first) of dicts, each with:
            ``rank``, ``address``, ``city``, ``zip_code``,
            ``parcel_id``, ``print_key``, ``sale_price``,
            ``sale_date`` (ISO 8601 string), ``latitude``,
            ``longitude``, ``distance`` (raw euclidean distance in the
            standardized comp-vector space), and ``similarity``
            (``1 / (1 + distance)``, so 1.0 = identical, higher is
            more similar).

        Raises:
            ValueError: If ``k`` is not a positive integer, or required
                feature columns are missing from ``property_features``.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        if isinstance(property_features, dict):
            df = pd.DataFrame([property_features])
        elif isinstance(property_features, pd.DataFrame):
            df = property_features
        else:
            raise TypeError(
                "property_features must be a dict or a one-row DataFrame, "
                f"got {type(property_features).__name__}"
            )

        features_df = self.features.transform(ensure_feature_inputs(df))
        # build_comp_vector zero-fills any feature family that is entirely
        # NaN here (e.g. geographic when the feature store / regressor
        # package is unavailable), so an inactive family contributes nothing
        # to the distance instead of crashing BallTree with "Input contains
        # NaN".  The same policy is applied to the index at build time.
        vector = np.asarray(
            self.features.build_comp_vector(features_df), dtype=np.float64
        ).reshape(1, -1)

        # BallTree rejects k > n_samples; cap it.  The population size is
        # taken from the lookup table, which is parallel to the index.
        effective_k = min(k, self.population_size)
        if effective_k == 0:
            return []

        distances, indices = self._index.query(vector, k=effective_k)

        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            results.append(
                self._format_result(rank, float(distance), self._lookup_df.iloc[idx])
            )
        return results

    def _format_result(self, rank: int, distance: float, row: pd.Series) -> dict:
        """Build one result dict from a lookup-table row and its distance."""
        sale_date = _clean_scalar(row.get("sale_date"))
        if hasattr(sale_date, "isoformat"):
            sale_date = sale_date.isoformat()
        return {
            "rank": rank,
            "address": _clean_scalar(row.get("address")),
            "city": _clean_scalar(row.get("city")),
            "zip_code": _clean_scalar(row.get("zip_code")),
            "parcel_id": _clean_scalar(row.get("parcel_id")),
            "print_key": _clean_scalar(row.get("print_key")),
            "sale_price": _clean_scalar(row.get("sale_price")),
            "sale_date": sale_date,
            "latitude": _clean_scalar(row.get("latitude")),
            "longitude": _clean_scalar(row.get("longitude")),
            "distance": distance,
            "similarity": 1.0 / (1.0 + distance),
        }
