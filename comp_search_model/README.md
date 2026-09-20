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

# Real mode additionally joins FRED macro indicators onto the population;
# set the API key to keep the macro family active:
FRED_API_KEY=... uv run python -m comp_finder.experiments.comp_quality real
```

- **Holdout price prediction (leave-one-out):** each query sale is removed
  from the index, the real `CompFinder` returns its top-k comps, and the
  median comp price is scored against baselines (population median,
  same-city median, random k) via Spearman, MAPE, and within-X% hit rates.
- **Family ablation:** same protocol under different physical/macro/
  geographic family weights to see which families carry price signal.

A family whose columns are entirely unresolvable (no `FRED_API_KEY` for
macro; no geographic-feature-store artifacts for the geo family) is zeroed
and reported as INACTIVE; see the module docstring. The Ticket 7b
geo-activation before/after evaluation is in
[reports/ticket7b_geo_activation.md](reports/ticket7b_geo_activation.md).

## Geographic feature store prerequisite

The 12 geo velocity features resolve against the
`geographic-feature-store` ArtifactStore (version `1.0.0`) at index-build
and query time. Publish the artifacts from the sibling project first (this
is the existing weekly batch command; no extra tooling):

```bash
cd ../geographic_feature_store
uv run python -m geographic_feature_store.flow run
```

With the default local ArtifactStore backend (`artifacts.base_path` in
`~/.geronimo/config.yaml`), the artifacts land under
`~/.geronimo/artifacts/geographic-feature-store/1.0.0/` — the six velocity
grids the comp-finder reads (`geo_velocity_r1_w90`, `geo_velocity_r1_w365`,
`geo_velocity_r5_w90`, `geo_velocity_r5_w180`, `geo_velocity_r10_w180`,
`geo_velocity_r10_w365`) plus `features_config` (H3 resolution). If the
store is missing, the geo features fall back to NaN and the family is
reported INACTIVE rather than crashing.

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

`expected_transaction_price` is an editable path dependency (see
`[tool.uv.sources]` in `pyproject.toml`), so `uv sync` installs it and the comp
vector uses the regressor's canonical feature lists and live geo velocity H3
lookups. The local fallback copies in `sdk/features.py` only engage when the
sibling checkout is absent (e.g. a standalone clone of this project alone), in
which case the 12 geographic velocity features return NaN and the geographic
family is reported INACTIVE.
