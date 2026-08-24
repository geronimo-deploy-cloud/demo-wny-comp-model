"""Training script for comp-vector index pipeline.

Usage:
    uv run python -m comp_finder.train
"""

from geronimo.artifacts import ArtifactStore
from comp_finder.pipeline import CompSearchPipeline, INDEX_PROJECT, INDEX_VERSION


def main():
    """Train and save the comp-vector index."""
    print("=" * 50)
    print("Comp-Vector Index Training")
    print("=" * 50)

    # 1. Initialize and train pipeline
    print("\n1. Training pipeline...")
    pipeline = CompSearchPipeline(h3_resolution=8)
    
    metrics = pipeline.train()
    print(f"   Training metrics: {metrics}")

    # 2. Save artifacts to ArtifactStore
    print("\n2. Saving artifacts...")
    
    store = ArtifactStore(
        project=INDEX_PROJECT,
        version=INDEX_VERSION,
    )
    
    paths = pipeline.publish(store)
    print(f"   Saved artifacts to {len(paths)} locations")
    print(f"   Backend: {store.backend}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
