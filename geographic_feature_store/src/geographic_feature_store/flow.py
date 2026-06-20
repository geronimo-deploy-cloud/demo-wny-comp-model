"""Metaflow flow — thin wrapper around GeoFeatureStorePipeline.

Run locally:
    python -m geographic_feature_store.flow run

Deploy to Step Functions:
    python -m geographic_feature_store.flow step-functions create
"""

from metaflow import FlowSpec, step, schedule
from geographic_feature_store.sdk.pipeline import GeoFeatureStorePipeline


@schedule(weekly=True)
class GeoFeatureStoreFlow(FlowSpec):
    """Batch flow for precomputing geographic velocity features."""

    @step
    def start(self):
        """Initialize the pipeline."""
        self.pipeline = GeoFeatureStorePipeline()
        self.pipeline.initialize()
        print(f"Initialized: {self.pipeline}")
        self.next(self.run_pipeline)

    @step
    def run_pipeline(self):
        """Execute the feature store pipeline."""
        self.result = self.pipeline.execute()
        print(f"Result: {self.result}")
        self.next(self.end)

    @step
    def end(self):
        """Flow complete."""
        print(f"Pipeline complete: {self.result}")


if __name__ == "__main__":
    GeoFeatureStoreFlow()
