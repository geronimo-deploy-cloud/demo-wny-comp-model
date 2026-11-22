# comp_search_model

Nearest-neighbor index for comparable sales. A weekly batch pipeline builds a BallTree over standardized 28-dimension "comp vectors" (physical + macro + geographic velocity features) of historical sales, and `CompFinder` queries it at inference time to retrieve ranked comparable sales.

**Names:** directory `comp_search_model/`, Python package `comp_finder`, ArtifactStore project `comp-finder` (hyphen), version `1.0.0`. Artifacts: `comp_vector_index`, `comp_vector_lookup`, `comp_vector_scaler`.

## Quick Start

```bash
# Install dependencies (pulls the sibling expected_transaction_price
# project in as an editable path dependency — see Dependencies below)
uv sync

# Build and publish the index locally
uv run python -m comp_finder.flow run

# Query the index (see src/comp_finder/sdk/model.py — CompFinder)
#   CompFinder(base_path=...).find_comps(property_features, k=5)

# Run tests
uv run pytest
```

## Dependencies

`expected_transaction_price` is installed into this project's venv as an
**editable path dependency** on the sibling workspace directory
(`[tool.uv.sources]` → `../expected_transaction_price_model`). It is the
single source of truth for the canonical feature lists and the 12
geographic-velocity derived functions, so `uv sync` here must run from
inside the monorepo layout (the sibling checkout must exist at the
relative path).

### Geographic feature store prerequisite

The geo velocity features resolve H3 cells against the
`geographic-feature-store` ArtifactStore (project
`geographic-feature-store`, version `1.0.0`). The artifacts live in the
local store at `~/.geronimo/artifacts/geographic-feature-store/1.0.0/`
(the path is set in `~/.geronimo/config.yaml`). Produce them by running
the producer pipeline once:

```bash
cd ../geographic_feature_store
uv run python -m geographic_feature_store.flow run
```

If the artifacts are missing, every geo feature emits NaN and the
feature layer treats the geographic family as absent (the features are
declared `required=False`) — nothing crashes, but index and query
vectors lose the geographic dimensions.

## Quality experiments

`src/comp_finder/experiments/comp_quality.py` empirically measures whether the
comp-vector index finds genuinely comparable sales:

```bash
# Offline sanity check: synthetic population (price = f(physical, geo)) plus
# a randomized-price negative control. No network access.
uv run python -m comp_finder.experiments.comp_quality synthetic

# Real Buffalo + Rochester sales (network; cached under /tmp afterwards).
# Geo features come from the geographic feature store (see prerequisite
# above); macro features are joined from FRED's public CSV endpoint
# (no API key needed).
uv run python -m comp_finder.experiments.comp_quality real [--limit N] [--json out.json]

# Reproduce the Ticket-5 physical-only baseline (no store geo, no macro)
uv run python -m comp_finder.experiments.comp_quality real --physical-only
```

- **Holdout price prediction (leave-one-out):** each query sale is removed
  from the index, the real `CompFinder` returns its top-k comps, and the
  median comp price is scored against baselines (population median,
  same-city median, random k) via Spearman, MAPE, and within-X% hit rates.
- **Family ablation:** same protocol under different physical/macro/
  geographic family weights to see which families carry price signal.

With the regressor package installed (the default after `uv sync`), `real`
mode reports no INACTIVE families and runs all six ablation configurations.
Where `expected_transaction_price` is absent, or the feature-store artifacts
are missing, the geographic family falls back to zeroed/INACTIVE (and on
real data, macro likewise when the FRED fetch fails); the report says so and
nothing crashes. Before/after reports for the geo activation (Ticket 7a) are
committed under `reports/`.

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
├── tests/
└── reports/               # committed comp_quality JSON (before/after)
```

## Feature Ownership

**`expected_transaction_price` is the arbiter of all feature names** across models.
See [sdk/FEATURES.md](src/comp_finder/sdk/FEATURES.md) for the full policy on how
to add features, handle exceptions, and avoid drift between models.

If `expected_transaction_price` is not installed in this venv (i.e. the path
dependency above could not be resolved), local fallback copies of the feature
lists are used and the 12 geographic velocity features silently return NaN —
see the comments in `sdk/features.py`. With the package installed, geo features
return values wherever the geographic feature store covers the property's H3
cell; the `real` experiments report which families are active.
