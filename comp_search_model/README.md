# comp_search_model

Nearest-neighbor index for comparable sales. A weekly batch pipeline builds a BallTree over standardized 28-dimension "comp vectors" (physical + macro + geographic velocity features) of historical sales, and `CompFinder` queries it at inference time to retrieve ranked comparable sales.

**Names:** directory `comp_search_model/`, Python package `comp_finder`, ArtifactStore project `comp-finder` (hyphen), version `1.0.0`. Artifacts: `comp_vector_index`, `comp_vector_lookup`, `comp_vector_scaler`.

## Quick Start

```bash
# Install dependencies
uv sync

# Build and publish the index locally
uv run python -m comp_finder.flow run

# Query the index (see src/comp_finder/sdk/model.py — CompFinder)
#   CompFinder(base_path=...).find_comps(property_features, k=5)

# Run tests
uv run pytest
```

## Quality experiments

`src/comp_finder/experiments/comp_quality.py` empirically measures whether the
comp-vector index finds genuinely comparable sales:

```bash
# Offline sanity check: synthetic population (price = f(physical, geo)) plus
# a randomized-price negative control. No network access.
uv run python -m comp_finder.experiments.comp_quality synthetic

# Real Buffalo + Rochester sales (network; cached under /tmp afterwards)
uv run python -m comp_finder.experiments.comp_quality real [--limit N] [--json out.json]
```

- **Holdout price prediction (leave-one-out):** each query sale is removed
  from the index, the real `CompFinder` returns its top-k comps, and the
  median comp price is scored against baselines (population median,
  same-city median, random k) via Spearman, MAPE, and within-X% hit rates.
- **Family ablation:** same protocol under different physical/macro/
  geographic family weights to see which families carry price signal.

In a venv without `expected_transaction_price` the geographic family (and, on
real data, the macro family) is inactive and zeroed; the report says so.

## Project Structure

```
comp_search_model/
├── pyproject.toml
├── src/
│   └── comp_finder/
│       ├── sdk/
│       │   ├── FEATURES.md        # Feature ownership policy (read this first)
│       │   ├── features.py        # CompFinderFeatures — the 28-dim comp vector
│       │   ├── pipeline.py        # CompSearchPipeline — builds + publishes the index
│       │   ├── model.py           # CompFinderModel (saves scaler) + CompFinder (find_comps query)
│       │   ├── endpoint.py
│       │   └── data_sources.py
│       ├── flow.py                # Metaflow orchestration (weekly batch)
│       ├── app.py                 # FastAPI application
│       ├── agent.py               # MCP server
│       ├── train.py
│       └── monitoring/
└── tests/
```

## Feature Ownership

**`expected_transaction_price` is the arbiter of all feature names** across models.
See [sdk/FEATURES.md](src/comp_finder/sdk/FEATURES.md) for the full policy on how
to add features, handle exceptions, and avoid drift between models.

If `expected_transaction_price` is not installed in this project's venv, local
fallback copies of the feature lists are used and the 12 geographic velocity
features silently return NaN — see the comments in `sdk/features.py`.
