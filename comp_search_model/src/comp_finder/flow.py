"""Metaflow flow for the comp-vector index pipeline.

This flow wraps the Geronimo `CompSearchPipeline` and provides
scheduled batch execution via Metaflow.

Usage:
    uv run python -m comp_finder.flow run              # local run
    uv run python -m comp_finder.flow step-functions create  # deploy to SF
"""

from metaflow import FlowSpec, step, card, project

from .pipeline import CompSearchPipeline, INDEX_PROJECT, INDEX_VERSION


@project(name="comp-finder-index", version="1.0.0")
class CompSearchFlow(FlowSpec):
    """Metaflow flow that builds and publishes a comp-vector nearest-neighbor index.

    This flow is the entry point for scheduled batch execution. It
    wraps the Geronimo `CompSearchPipeline` and exposes it through
    Metaflow's `@step` decorators.

    Scheduling decisions:
    - Default schedule: **weekly** (Sunday 2 AM UTC)
    - Rationale: Real estate sales data updates are incremental and
      slow-moving. A weekly refresh captures new sales while keeping
      compute costs reasonable. The full historical rebuild is cheap
      compared to incremental updates since the BallTree must be
      rebuilt from scratch anyway.
    - Alternative schedules considered:
      - Daily: Unnecessary overhead — sales data doesn't change enough
        to justify daily rebuilds.
      - Monthly: Too stale for market-sensitive pricing.
      - On-demand only: Would require manual triggering, which is
        error-prone for production use.

    The schedule can be overridden via Metaflow's `--schedule` flag
    or by modifying the `@schedule` decorator.
    """

    @step
    def start(self):
        """Entry point: initialize the pipeline."""
        self.pipeline = CompSearchPipeline(h3_resolution=8)
        self.next(self.train)

    @step
    def train(self):
        """Train: load sales, build comp vectors, build index."""
        metrics = self.pipeline.train()
        self.metrics = metrics
        self.next(self.publish)

    @step
    def publish(self):
        """Publish the index and lookup table to ArtifactStore."""
        from geronimo.artifacts import ArtifactStore

        store = ArtifactStore(project=INDEX_PROJECT, version=INDEX_VERSION)
        saved = self.pipeline.publish(store)
        self.saved_artifacts = saved
        self.next(self.end)

    @step
    def end(self):
        """Final step: log results."""
        self.summary = {
            "metrics": self.metrics,
            "saved_artifacts": self.saved_artifacts,
        }
        print(f"Pipeline complete. Saved: {self.saved_artifacts}")
        print(f"Metrics: {self.metrics}")


if __name__ == "__main__":
    CompSearchFlow()
