"""Model definition for the geographic feature store.

Maps onto Geronimo's Model pattern:

    train():
        1. Load sales data from data sources
        2. Fit the GeoVelocityFeatures — runs BallTree for each declared
           Feature (each radius×window grid)
        3. The fitted features ARE the model — no separate estimator

    predict(df):
        1. Maps input lat/lng → H3 cell
        2. Returns each declared Feature's precomputed value

    save(store) / load(store):
        Persist/restore fitted velocity grids via ArtifactStore

The features attribute provides a self-documenting catalog of what velocity
grids have been precomputed:

    model.features.sales_volume_r1_w30   # 1mi radius, 30-day volume
    model.features.median_ppsf_r5_w90    # 5mi radius, 90-day PPSF
    model.features.sales_volume_r10_w365 # 10mi radius, annual volume
    ...
"""

from typing import Optional

import pandas as pd

from geronimo.models import Model
from geronimo.artifacts import ArtifactStore
from .features import GeoVelocityFeatures, DEFAULT_H3_RESOLUTION
from .data_sources import training_sources


class GeoFeatureStoreModel(Model):
    """Model for precomputing and serving geographic velocity features.

    The "model" here is the fitted GeoVelocityFeatures. There is no separate
    ML estimator — training IS the velocity computation, and prediction IS
    the H3 cell lookup.

    The declared Features on the FeatureSet make it explicit which grids
    have been precomputed:

        >>> model = GeoFeatureStoreModel()
        >>> model.features.feature_names
        ['sales_volume_r1_w30', 'median_ppsf_r1_w30',
         'sales_volume_r1_w90', 'median_ppsf_r1_w90',
         ...
         'sales_volume_r10_w365', 'median_ppsf_r10_w365']

    Example:
        >>> model = GeoFeatureStoreModel()
        >>> metrics = model.train()               # loads sales, fits all grids
        >>> enriched = model.predict(properties)   # H3 cell lookup
        >>> enriched['sales_volume_r1_w30']        # access individual feature
    """

    name = "geographic-feature-store"
    version = "1.0.0"

    def __init__(self, h3_resolution: int = DEFAULT_H3_RESOLUTION):
        """Initialize model.

        Args:
            h3_resolution: H3 grid resolution (default 8, ~460m edge).
                           Higher = finer granularity, more cells.
        """
        super().__init__()
        self.features = GeoVelocityFeatures(h3_resolution=h3_resolution)

    def train(self) -> dict:
        """Train: load sales data and fit all declared velocity Features.

        Loads sales from all configured data sources (Buffalo + Rochester),
        then fits the GeoVelocityFeatures which computes BallTree velocity
        at every H3 grid cell for each declared (radius, window) Feature.

        Returns:
            Dict with training metrics.
        """
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

        if df.empty:
            raise ValueError("No sales data loaded from data sources.")

        # Fit features — runs velocity computation for every declared Feature
        self.features.fit(df)
        self._is_fitted = True

        return {
            "status": "success",
            "sales_records": len(df),
            "grid_cells": len(self.features.grid),
            "velocity_grids": len(self.features.velocity_data),
            "features": self.features.feature_names,
            "h3_resolution": self.features.h3_resolution,
            "reference_date": str(self.features.reference_date),
        }

    def predict(self, X) -> pd.DataFrame:
        """Predict: look up precomputed Feature values for input locations.

        Maps each row's lat/lng → H3 cell → retrieves all declared Feature
        values (sales_volume, median_ppsf for each radius×window combo).

        Args:
            X: DataFrame with 'latitude' and 'longitude' columns.

        Returns:
            DataFrame with one column per declared Feature.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")

        if isinstance(X, pd.DataFrame):
            return self.features.transform(X)

        raise TypeError(f"Expected DataFrame, got {type(X).__name__}")

    def save(self, store: ArtifactStore) -> list[str]:
        """Save fitted features to ArtifactStore.

        Each velocity grid is saved as a separate artifact, plus the
        H3 grid and config.

        Args:
            store: ArtifactStore instance.

        Returns:
            List of saved artifact names.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Nothing to save.")

        self.features.save(store)
        saved = list(self.features.velocity_data.keys()) + ["grid", "features_config"]
        return saved

    def load(self, store: ArtifactStore) -> None:
        """Load fitted features from ArtifactStore.

        Args:
            store: ArtifactStore instance.
        """
        self.features.load(store)
        self._is_fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
