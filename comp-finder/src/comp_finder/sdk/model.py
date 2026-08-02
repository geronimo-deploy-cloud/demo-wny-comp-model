"""Model definition for comp-finder.

This is the central file for your ML model. Implement:
- train(): Load data, fit features, train estimator
- predict(): Transform input and generate predictions
- save(): Persist estimator and features to ArtifactStore
- load(): Restore estimator and features from ArtifactStore
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from geronimo.models import Model, HyperParams
from geronimo.artifacts import ArtifactStore
from .features import CompFinderFeatures
from .data_sources import training_sources


class CompFinderModel(Model):
    """ML model for comp-finder.
    
    Uses declarative features for transformation and ArtifactStore for persistence.
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
        
        # TODO: Configure your target column if supervised learning
        # y = df["target"].values
        
        # Initialize and fit features
        self.features = CompFinderFeatures()
        # X = self.features.fit_transform(df)
        
        # TODO: Initialize and train your estimator
        # from sklearn.ensemble import RandomForestClassifier
        # params = HyperParams(n_estimators=100, max_depth=5, random_state=42)
        # self.estimator = RandomForestClassifier(**params.to_dict())
        # self.estimator.fit(X, y)
        
        self._is_fitted = True
        
        # TODO: Return training metrics
        return {
            "n_samples": len(df),
            # "accuracy": self.estimator.score(X, y),
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
    
    def save(self, store: ArtifactStore) -> list[str]:
        """Save trained model and features to ArtifactStore.
        
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
            tags={"model": self.name, "version": self.version}
        )
        paths.append(path)
        
        # Save the fitted features (includes transformers/scalers)
        path = store.save(
            "features",
            self.features,
            artifact_type="CompFinderFeatures",
            tags={"model": self.name, "version": self.version}
        )
        paths.append(path)
        
        return paths
    
    def load(self, store: ArtifactStore) -> None:
        """Load trained model and features from ArtifactStore.
        
        Args:
            store: ArtifactStore instance
        """
        self.estimator = store.get("estimator")
        self.features = store.get("features")
        self._is_fitted = True
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is trained and ready for predictions."""
        return self._is_fitted
