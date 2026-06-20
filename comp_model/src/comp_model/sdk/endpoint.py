"""Endpoint definition - handle incoming prediction requests."""

import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from geronimo.serving import Endpoint
from geronimo.artifacts import ArtifactStore
from .model import CompModelModel
from .data_sources import UNIFIED_COLUMNS, _assign_school_districts, training_macro
from .data_sources import lookup_rochester_assessment
from .scrapers.factory import ScraperFactory


class CompModelEndpoint(Endpoint):
    """Prediction endpoint for real-time serving.

    This is a working demo endpoint. Replace the preprocess/postprocess
    methods with your actual implementation once you have a trained model.

    To train a model:
        uv run python -m comp_model.train
    """

    model_class = CompModelModel

    def preprocess(self, request: dict):
        """Transform incoming request to model input.

        Args:
            request: JSON request body with "features" key

        Returns:
            Feature matrix ready for model.predict()
        """
        # Demo mode: just return the features dict
        # TODO: Replace with actual preprocessing once model is trained
        # import pandas as pd
        # df = pd.DataFrame([request["features"]])
        # return self.model.features.transform(df)
        return request.get("features", request)

    def postprocess(self, prediction):
        """Format model output for response.

        Args:
            prediction: Raw model output

        Returns:
            JSON-serializable response
        """
        # Demo mode: echo the input back
        # TODO: Replace with actual postprocessing once model is trained
        # return {"prediction": int(prediction[0]), "confidence": 0.95}
        return {"result": prediction, "status": "demo_mode"}

    def initialize(self, project=None, version=None):
        """Initialize endpoint.

        Demo mode: Skip model loading if artifacts don't exist.
        """
        try:
            super().initialize(project=project, version=version)
        except Exception:
            # Demo mode: continue without trained model
            self.model = None
            self._is_initialized = True

    def handle(self, request: dict) -> dict:
        """Handle prediction request.

        Demo mode: If no model loaded, echo the request back.
        """
        if self.model is None:
            # Demo mode
            features = self.preprocess(request)
            return self.postprocess(features)

        # Normal mode with trained model
        return super().handle(request)

    # =========================================================================
    # Prediction Pipeline
    # =========================================================================

    def _ensure_model(self):
        """Ensure the XGBoost model is loaded from ArtifactStore."""
        if self.model is None:
            store = ArtifactStore(project="comp_model", version="1.0.0")
            model = CompModelModel()
            model.load(store)
            return model
        return self.model

    def _build_dataframe(self, features: dict) -> pd.DataFrame:
        """Build a model-ready DataFrame from raw feature dict.

        Backfills any missing unified columns with NaN. Sets sale_date to now.
        """
        df = pd.DataFrame([features])
        for col in UNIFIED_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df['sale_date'] = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        return df

    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge macroeconomic indicators and assign school districts."""
        macro_df = training_macro.load()

        if not macro_df.empty:
            latest_macro_df = macro_df.loc[[macro_df['sale_date'].idxmax()]]
            latest_macro_df = latest_macro_df.drop(columns=['sale_date'])
            df = df.merge(latest_macro_df, how='cross')

        for col in ["mortgage_30y", "fed_funds", "cpi", "unemployment"]:
            if col not in df.columns:
                df[col] = np.nan

        df = _assign_school_districts(df)
        return df

    def _demo_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill in assessed_value from sale_price/2 if unavailable."""
        assessed = df.at[0, "assessed_value"]
        if pd.isna(assessed) or assessed in (0, 0.0):
            asking = df.at[0, "sale_price"] or 0
            print(f"WARNING: assessed_value unavailable; using sale_price/2 = ${asking / 2:,.0f} "
                  f"as a demo placeholder (prediction will be degraded).")
            df.at[0, "assessed_value"] = asking / 2.0
        return df

    def predict_features(self, features: dict) -> float:
        """Run the full prediction pipeline on a raw feature dict.

        Parameters:
            features: Dict of property attributes (beds, baths, year_built, etc.)

        Returns:
            Predicted sale price as float.
        """
        model = self._ensure_model()
        df = self._build_dataframe(features)
        df = self._enrich_dataframe(df)
        df = self._demo_fallback(df)
        X = model.features.transform(df)
        return model.predict(df)[0]

    def _scraper_to_features(self, url: str) -> dict:
        """Scrape a property URL and return a raw feature dict.

        Enriches with Rochester assessor data if the scraper missed
        assessment fields. Updates stories to 2 if second_floor_area > 0.
        """
        target_id = url
        scraper = ScraperFactory.get_scraper(url)
        if not scraper:
            raise ValueError("Unsupported URL domain. Scaffold a new Scraper for this site.")

        features = scraper.extract_raw_features()

        # Enrich with City of Rochester assessor data when needed
        if not features.get("assessed_value"):
            address = features.get("address")
            zip_code = features.get("zip_code")
            print(f"Assessment missing from scrape; querying City of Rochester for '{address}'...")
            enrichment = lookup_rochester_assessment(address, zip_code)
            if enrichment:
                filled = []
                for k, v in enrichment.items():
                    existing = features.get(k)
                    if existing in (None, 0, 0.0, "", "Unknown"):
                        features[k] = v
                        filled.append(k)
                print(f"   Enrichment filled: {filled}")
            else:
                print("   No Rochester record found. (Only City of Rochester properties "
                      "are covered - Monroe County suburbs aren't in this MapServer.)")

        # If scrubbing brought in a non-zero second_floor_area, property has 2+ stories
        second_floor = pd.to_numeric(features.get("second_floor_area"), errors="coerce")
        if pd.notna(second_floor) and second_floor > 0:
            features["stories"] = 2

        return features

    def predict_from_url(self, url: str) -> float:
        """Scrape a property URL and predict its sale price.

        Chains scraping → enrichment → feature transform → prediction.

        Parameters:
            url: A live Zillow or similar property listing URL.

        Returns:
            Predicted sale price as float.
        """
        features = self._scraper_to_features(url)
        return self.predict_features(features)
