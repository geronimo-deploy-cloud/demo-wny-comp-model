"""Model definition for comp-finder.

Integrates with expected-transaction-price's feature set to produce a
standardized comp vector suitable for nearest-neighbor comparison.

The comp vector scaler is persisted via ArtifactStore during `save()`
and restored during `load()`, following the same pattern used by
`expected-transaction-price` for its XGBoost estimator.
"""

from typing import Any, Optional
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
    get_comp_vector_info,
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
        if (
            self.features._comp_vector_scaler is None
            and self.features._comp_vector_scaler is not None
        ):
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
