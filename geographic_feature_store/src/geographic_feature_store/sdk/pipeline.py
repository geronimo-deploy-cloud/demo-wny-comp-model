"""Geographic Feature Store batch pipeline.

Precomputes geographic velocity features (sales volume and median PPSF) for H3
grid cells covering Buffalo and Rochester metro areas. Produces 12 independent
data products — one per (radius, lookback_window) combination — stored as
separate artifacts in the Geronimo ArtifactStore.

Data Products:
    geo_velocity_r1_w30    geo_velocity_r5_w30    geo_velocity_r10_w30
    geo_velocity_r1_w90    geo_velocity_r5_w90    geo_velocity_r10_w90
    geo_velocity_r1_w180   geo_velocity_r5_w180   geo_velocity_r10_w180
    geo_velocity_r1_w365   geo_velocity_r5_w365   geo_velocity_r10_w365

Schedule: Weekly (Sunday 4 AM)
"""

import logging
from datetime import datetime

from geronimo.batch import BatchPipeline
from geronimo.batch.schedule import Schedule
from geronimo.artifacts import ArtifactStore

from .model import GeoFeatureStoreModel
from .features import DEFAULT_H3_RESOLUTION

logger = logging.getLogger(__name__)

ARTIFACT_PROJECT = "geographic-feature-store"
ARTIFACT_VERSION = "1.0.0"


class GeoFeatureStorePipeline(BatchPipeline):
    """Weekly precomputation of geographic velocity features for WNY.

    Uses the standard Geronimo model lifecycle:
        1. model.train() — loads sales data, fits velocity features
        2. model.save(store) — persists 12 data products to ArtifactStore
    """

    model_class = GeoFeatureStoreModel
    schedule = Schedule.weekly(day=0, hour=4)  # Sunday 4 AM

    def __init__(self, h3_resolution: int = DEFAULT_H3_RESOLUTION):
        """Initialize pipeline with tunable parameters.

        Args:
            h3_resolution: H3 grid resolution (default 8, ~460m edge).
        """
        super().__init__()
        self.h3_resolution = h3_resolution

    def initialize(self):
        """Initialize pipeline — no pre-trained model to load.

        This pipeline trains (computes features) fresh each run, so we
        just create the model instance and mark as initialized.
        """
        self.model = GeoFeatureStoreModel(h3_resolution=self.h3_resolution)
        self._is_initialized = True
        logger.info("GeoFeatureStorePipeline initialized: %s", self.model.features)

    def execute(self):
        """Execute the pipeline."""
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.run()

    def run(self):
        """Execute the geographic feature store batch job.

        Steps:
            1. Train the model (loads sales data, fits velocity features)
            2. Save the fitted features to ArtifactStore

        Returns:
            Dict with execution summary.
        """
        start_time = datetime.utcnow()

        # 1. Train: load data and compute velocity at every grid cell
        logger.info("Step 1/2: Training model (fitting velocity features)...")
        metrics = self.model.train()
        logger.info("Training metrics: %s", metrics)

        # 2. Save: persist each data product as a separate artifact
        logger.info("Step 2/2: Saving to ArtifactStore...")
        store = ArtifactStore(project=ARTIFACT_PROJECT, version=ARTIFACT_VERSION)
        saved = self.model.save(store)
        logger.info("Saved %d artifacts", len(saved))

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        summary = {
            **metrics,
            "artifacts_saved": len(saved),
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info(
            "Pipeline complete: %d artifacts in %.1f seconds",
            len(saved), elapsed,
        )

        return summary
