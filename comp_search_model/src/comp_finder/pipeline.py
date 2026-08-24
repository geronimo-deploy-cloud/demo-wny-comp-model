"""Batch pipeline for building a comp-vector nearest-neighbor index.

This pipeline:

    1. Loads the full historical sales population (Buffalo + Rochester)
    2. Builds comp vectors for every sale using the same `build_comp_vector`
       logic as the realtime model, but with a scaler loaded from the
       comp-finder's ArtifactStore (version "1.0.0").
    3. Builds a BallTree (or NearestNeighbors) index over all vectors.
    4. Publishes the index and a parallel lookup table (address, price,
       date, etc.) to the ArtifactStore.

There is no realtime serving component — this is purely a scheduled
batch operation.

Usage:
    uv run python -m comp_finder.pipeline run
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from geronimo.artifacts import ArtifactStore
from geronimo.batch import BatchPipeline
from geronimo.batch.schedule import Schedule

from .sdk.features import (
    ALL_VECTOR_FEATURES,
    CompFinderFeatures,
    DEFAULT_FAMILY_WEIGHTS,
    SCALER_ARTIFACT_NAME,
    SCALER_PROJECT,
    SCALER_VERSION,
)
from .sdk.data_sources import _load_combined_training_data

logger = logging.getLogger(__name__)

# ArtifactStore keys for the index pipeline.
INDEX_PROJECT = "comp-finder"
INDEX_VERSION = "1.0.0"
INDEX_ARTIFACT_NAME = "comp_vector_index"
LOOKUP_ARTIFACT_NAME = "comp_vector_lookup"


class CompSearchPipeline(BatchPipeline):
    """Pipeline that builds and publishes a comp-vector nearest-neighbor index.

    The pipeline follows the same pattern as `geographic_feature_store`:
    a Metaflow flow wraps a Geronimo BatchPipeline, which calls
    initialize() → run() → save to ArtifactStore.

    Schedule: Weekly (Sunday 2 AM UTC)
    Rationale: Real estate sales data updates are incremental and
    slow-moving. A weekly refresh captures new sales while keeping
    compute costs reasonable. The full historical rebuild is cheap
    compared to incremental updates since the BallTree must be
    rebuilt from scratch anyway.
    """

    model_class = None  # No separate model class needed
    schedule = Schedule.weekly(day=0, hour=2)  # Sunday 2 AM UTC

    def __init__(self, h3_resolution: int = 8):
        """Initialize pipeline with tunable parameters.

        Args:
            h3_resolution: H3 grid resolution (default 8, ~460m edge).
        """
        super().__init__()
        self.h3_resolution = h3_resolution
        self._index: Optional[BallTree] = None
        self._lookup_df: Optional[pd.DataFrame] = None

    def initialize(self):
        """Initialize pipeline — create the features instance."""
        self.features = CompFinderFeatures()
        self._is_initialized = True
        logger.info("CompSearchPipeline initialized: h3_resolution=%d", self.h3_resolution)

    def run(self):
        """Execute the comp-vector index batch job.

        Steps:
            1. Load full historical sales data
            2. Transform to feature columns
            3. Load scaler from ArtifactStore
            4. Build comp vectors for every sale
            5. Build BallTree index
            6. Build lookup table
            7. Save index and lookup to ArtifactStore

        Returns:
            Dict with execution summary.
        """
        start_time = datetime.utcnow()

        if not getattr(self, '_is_initialized', False):
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # 1. Load full historical sales population
        logger.info("Loading full historical sales population...")
        df = _load_combined_training_data()

        if df.empty:
            raise ValueError("No sales data loaded from data sources.")

        logger.info("Loaded %d sales records", len(df))

        # 2. Transform to get feature columns
        logger.info("Transforming features...")
        features_df = self.features.transform(df)

        # 3. Load scaler from ArtifactStore
        logger.info("Loading comp vector scaler from ArtifactStore...")
        scaler = self._load_scaler()
        self.features.set_scaler(scaler)

        # 4. Build comp vectors for every sale
        logger.info("Building comp vectors for %d records...", len(df))
        comp_vectors = self.features.build_comp_vector(features_df)

        # 5. Build BallTree index
        logger.info("Building BallTree index...")
        self._index = BallTree(comp_vectors, metric="euclidean")

        # 6. Build lookup table (address, price, date, etc.)
        logger.info("Building lookup table...")
        self._lookup_df = df[UNIFIED_COLUMNS].copy()
        # Add comp_vector_index column for reverse lookup
        self._lookup_df["comp_vector_index"] = range(len(df))

        # 7. Save to ArtifactStore
        logger.info("Saving to ArtifactStore...")
        store = ArtifactStore(project=INDEX_PROJECT, version=INDEX_VERSION)
        saved = self._publish(store)

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        summary = {
            "status": "success",
            "sales_records": len(df),
            "comp_vector_length": len(comp_vectors[0]),
            "index_type": "BallTree",
            "features": list(ALL_VECTOR_FEATURES),
            "scaler_project": SCALER_PROJECT,
            "scaler_version": SCALER_VERSION,
            "artifacts_saved": len(saved),
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info(
            "Pipeline complete: %d artifacts in %.1f seconds",
            len(saved), elapsed,
        )
        return summary

    def _publish(self, store: ArtifactStore) -> list[str]:
        """Publish the index and lookup table to ArtifactStore.

        Args:
            store: ArtifactStore instance.

        Returns:
            List of saved artifact names.
        """
        saved = []

        # Save BallTree index
        logger.info("Saving BallTree index to ArtifactStore...")
        import joblib
        from io import BytesIO
        buf = BytesIO()
        joblib.dump(self._index, buf)
        store.save(
            name=INDEX_ARTIFACT_NAME,
            artifact=buf.getvalue(),
        )
        saved.append(INDEX_ARTIFACT_NAME)

        # Save lookup table
        logger.info("Saving lookup table to ArtifactStore...")
        store.save(
            name=LOOKUP_ARTIFACT_NAME,
            artifact=self._lookup_df,
        )
        saved.append(LOOKUP_ARTIFACT_NAME)

        return saved

    def load(self, store: ArtifactStore) -> None:
        """Load the index and lookup table from ArtifactStore.

        Args:
            store: ArtifactStore instance.
        """
        import joblib
        from io import BytesIO

        # Load BallTree index
        logger.info("Loading BallTree index from ArtifactStore...")
        index_bytes = store.get(
            name=INDEX_ARTIFACT_NAME,
        )
        self._index = joblib.load(BytesIO(index_bytes))

        # Load lookup table
        logger.info("Loading lookup table from ArtifactStore...")
        self._lookup_df = store.get(
            name=LOOKUP_ARTIFACT_NAME,
        )

    def find_comps(
        self,
        comp_vector: np.ndarray,
        k: int = 10,
    ) -> pd.DataFrame:
        """Find k nearest comps for a given comp vector.

        Args:
            comp_vector: 1-D numpy array from build_comp_vector().
            k: Number of nearest neighbors to return.

        Returns:
            DataFrame with the k nearest comps from the lookup table.
        """
        if self._index is None:
            raise RuntimeError("Pipeline not fitted. Call run() or load() first.")

        comp_vector = np.array(comp_vector).reshape(1, -1)
        distances, indices = self._index.query(comp_vector, k=k)

        return self._lookup_df.iloc[indices[0]].reset_index(drop=True)

    def _load_scaler(self) -> object:
        """Load the comp vector scaler from ArtifactStore."""
        store = ArtifactStore(project=SCALER_PROJECT, version=SCALER_VERSION)
        return store.get(artifact_name=SCALER_ARTIFACT_NAME)

    @property
    def is_fitted(self) -> bool:
        return self._index is not None and self._lookup_df is not None


# Unified schema columns for the lookup table (from expected-transaction-price)
UNIFIED_COLUMNS = [
    "source",
    "parcel_id",
    "print_key",
    "address",
    "city",
    "zip_code",
    "property_class",
    "property_class_desc",
    "sale_price",
    "sale_date",
    "assessed_value",
    "land_value",
    "year_built",
    "total_living_area",
    "first_floor_area",
    "second_floor_area",
    "beds",
    "baths",
    "half_baths",
    "kitchens",
    "stories",
    "fireplaces",
    "lot_frontage",
    "lot_depth",
    "lot_acres",
    "building_style",
    "overall_condition",
    "construction_grade",
    "exterior_wall",
    "heat_type",
    "central_air",
    "fuel_type",
    "basement_type",
    "latitude",
    "longitude",
    "school_district",
]
