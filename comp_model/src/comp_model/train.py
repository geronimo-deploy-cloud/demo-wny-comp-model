"""Training script for comp_model.

Usage:
    python -m comp_model.train
"""

from geronimo.artifacts import ArtifactStore
from comp_model.sdk.model import CompModelModel


def main():
    """Train and save the model."""
    print("=" * 60)
    print("Comp Model Training — Rochester Only")
    print("=" * 60)

    # 1. Initialize and train model
    print("\n1. Training model...")
    model = CompModelModel()
    metrics = model.train()

    print("\n   Training Results:")
    print(f"   Samples:       {metrics['n_samples_filtered']:,} (filtered from {metrics['n_samples_raw']:,})")
    print(f"   Train/Val:     {metrics['n_train']:,} / {metrics['n_val']:,}")
    print(f"   Log Target:    {metrics['log_target']}")
    print(f"   Best Iter:     {metrics['best_iteration']}")
    print(f"   Train R²:      {metrics['train_r2']:.4f}")
    print(f"   Val R²:        {metrics['val_r2']:.4f}")
    print(f"   Val RMSE:      ${metrics['val_rmse']:,.0f}")
    print(f"   Val MAE:       ${metrics['val_mae']:,.0f}")
    print(f"   Val MAPE:      {metrics['val_mape']:.1f}%")
    print(f"   Val Bias:      ${metrics['val_bias']:+,.0f}")

    # 2. Save model artifacts
    print("\n2. Saving artifacts...")

    store = ArtifactStore(
        project="comp_model",
        version="1.0.0",
    )

    paths = model.save(store)
    print(f"   Saved {len(paths)} artifacts")
    print(f"   Backend: {store.backend}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
