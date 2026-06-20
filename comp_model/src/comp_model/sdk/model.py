"""Model definition for comp_model.

Trains on Rochester data only with log-transformed target for better
heteroscedastic handling. Uses the geographic feature store for velocity
features (no inline BallTree).

Key changes from v1:
  - Rochester-only training (no Buffalo mixing)
  - log1p(sale_price) target → expm1 predictions
  - Outlier filtering ($20k–$1.5M)
  - Temporal train/val split with early stopping
  - Stronger XGBoost regularization
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from geronimo.models import Model, HyperParams
from geronimo.artifacts import ArtifactStore
from .features import CompModelFeatures
from .data_sources import training_rochester, training_macro


class CompModelModel(Model):
    """XGBoost comp model trained on Rochester property data.
    
    Uses declarative features for transformation and ArtifactStore for persistence.
    Target is log-transformed sale_price for better handling of price heteroscedasticity.
    """

    name = "comp_model"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.estimator: Optional[Any] = None
        self.features: Optional[CompModelFeatures] = None
        self._is_fitted = False
        self._log_target = True  # Train on log1p(sale_price)

    def train(self) -> dict:
        """Train the model on Rochester data.
        
        Steps:
          1. Load Rochester training data (with school districts)
          2. Filter outlier prices ($20k–$1.5M) and old sales (pre-2015)
          3. Log-transform target
          4. Fit features (including geo velocity from feature store)
          5. Random train/val split (80/20)
          6. Train XGBoost with early stopping on validation set

        Returns:
            Training metrics dict
        """
        import xgboost as xgb
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from sklearn.model_selection import train_test_split
        import os
        from dotenv import load_dotenv
        
        # Load local .env for FRED API Key if present
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

        # 1. Load Rochester data
        df = training_rochester.load()
        macro_df = training_macro.load()
        
        # Merge FRED macro data
        if not macro_df.empty:
            df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
            df = df.merge(macro_df, on="sale_date", how="left")
            
        df = df.dropna(subset=['sale_price']).copy()
        initial_count = len(df)

        # 2. Filter outliers and old data
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
        df = df.dropna(subset=["sale_date"])
        df = df[
            (df["sale_price"] >= 20_000) &
            (df["sale_price"] <= 1_500_000) &
            (df["sale_date"] >= "2015-01-01")  # Last ~10 years only
        ].copy()
        filtered_count = len(df)

        # 3. Target (log-transform)
        if self._log_target:
            y = np.log1p(df["sale_price"].values)
        else:
            y = df["sale_price"].values
        
        # 4. Fit features
        self.features = CompModelFeatures()
        X = self.features.fit_transform(df)
        
        # 5. Random train/val split (80/20, stratified by price decile)
        price_decile = pd.qcut(df["sale_price"], q=10, labels=False, duplicates="drop")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=price_decile,
        )
        
        # 6. XGBoost with regularized hyperparameters
        params = HyperParams(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=1.0,
            reg_lambda=5.0,
            gamma=0.1,
            random_state=42,
            enable_categorical=True,
            tree_method="hist",
        )
        
        self.estimator = xgb.XGBRegressor(**params.to_dict())
        self.estimator.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        self._is_fitted = True
        
        # 8. Compute metrics on validation set (in original price space)
        val_pred_log = self.estimator.predict(X_val)
        if self._log_target:
            val_pred = np.expm1(val_pred_log)
            val_actual = np.expm1(y_val)
        else:
            val_pred = val_pred_log
            val_actual = y_val

        r2 = float(r2_score(val_actual, val_pred))
        rmse = float(np.sqrt(mean_squared_error(val_actual, val_pred)))
        mae = float(mean_absolute_error(val_actual, val_pred))
        mape = float(np.mean(np.abs((val_pred - val_actual) / np.maximum(val_actual, 1))) * 100)
        bias = float(np.mean(val_pred - val_actual))

        # Also compute training metrics for overfitting detection
        train_pred_log = self.estimator.predict(X_train)
        if self._log_target:
            train_pred = np.expm1(train_pred_log)
            train_actual = np.expm1(y_train)
        else:
            train_pred = train_pred_log
            train_actual = y_train
        train_r2 = float(r2_score(train_actual, train_pred))

        best_iter = getattr(self.estimator, 'best_iteration', params.to_dict()['n_estimators'])

        return {
            "n_samples_raw": initial_count,
            "n_samples_filtered": filtered_count,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "log_target": self._log_target,
            "best_iteration": best_iter,
            "train_r2": train_r2,
            "val_r2": r2,
            "val_rmse": rmse,
            "val_mae": mae,
            "val_mape": mape,
            "val_bias": bias,
        }

    def predict(self, X) -> np.ndarray:
        """Predict sale prices.
        
        Args:
            X: Feature array or raw DataFrame
            
        Returns:
            Predictions in original price space (not log)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.features.feature_names)
        else:
            df = X
        
        # Transform using fitted features
        X_transformed = self.features.transform(df)
        predictions = self.estimator.predict(X_transformed)

        # Inverse log transform
        if self._log_target:
            predictions = np.expm1(predictions)

        return predictions
    
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
            artifact_type="CompModelFeatures",
            tags={"model": self.name, "version": self.version}
        )
        paths.append(path)

        # Save model config (log_target flag needed for inference)
        path = store.save(
            "model_config",
            {"log_target": self._log_target, "version": self.version},
            artifact_type="config",
            tags={"model": self.name}
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

        # Load config if available (backward compat with old saves)
        try:
            config = store.get("model_config")
            self._log_target = config.get("log_target", True)
        except Exception:
            self._log_target = True  # Default for new models

        self._is_fitted = True
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is trained and ready for predictions."""
        return self._is_fitted
