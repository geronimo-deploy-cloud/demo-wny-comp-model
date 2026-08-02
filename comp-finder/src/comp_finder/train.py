"""Training script for comp-finder.

Usage:
    uv run python -m comp_finder.train
"""

from geronimo.artifacts import ArtifactStore
from comp_finder.sdk.model import CompFinderModel


def main():
    """Train and save the model."""
    print("=" * 50)
    print("Model Training")
    print("=" * 50)

    # 1. Initialize and train model
    # Data loading and feature engineering are handled by the model class
    print("\n1. Training model...")
    model = CompFinderModel()
    
    # Train (logic encapsulated in model.train())
    metrics = model.train()
    print(f"   Training metrics: {metrics}")

    # 2. Save model artifacts
    print("\n2. Saving artifacts...")
    
    # ArtifactStore uses your global config from ~/.geronimo/config.yaml
    store = ArtifactStore(
        project="comp-finder",
        version="1.0.0",
    )
    
    paths = model.save(store)
    print(f"   Saved artifacts to {len(paths)} locations")
    print(f"   Backend: {store.backend}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
