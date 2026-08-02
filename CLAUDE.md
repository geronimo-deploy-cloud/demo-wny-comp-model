# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a **multi-project workspace** with two sibling Geronimo-generated Python projects that are coupled at runtime via the Geronimo `ArtifactStore`. The top-level `pyproject.toml` is essentially a stub — real work happens inside the subdirectories, each of which has its own `.venv/`, `pyproject.toml`, `uv.lock`, and `geronimo.yaml`.

```
wny_real_estate_comp_model/
├── expected-transaction-price/  # realtime FastAPI + XGBoost regressor (consumer)
└── geographic_feature_store/    # weekly Metaflow batch pipeline (producer)
```

**Always `cd` into the project subdirectory** before running `uv` commands — each project has its own lockfile and virtualenv.

## Common commands

All commands assume you've `cd`'d into the relevant subdirectory.

```bash
# Install deps for either project
uv sync

# Add new deps for either project
uv add <package_name/>

# expected-transaction-price (realtime serving)
uv run uvicorn expected_transaction_price.app:app --reload          # API on :8000, MCP on /mcp
uv run python -m expected_transaction_price.train                    # train and save to ArtifactStore
uv run python -m expected_transaction_price.agent                    # MCP server via stdio
uv run pytest                                        # tests
uv run pytest tests/test_sdk.py::TestProjectModel    # single test class

# geographic_feature_store (batch)
uv run python -m geographic_feature_store.flow run                       # run pipeline locally
uv run python -m geographic_feature_store.train                          # one-shot train+save
uv run python -m geographic_feature_store.flow step-functions create     # deploy
uv run pytest
```

`expected-transaction-price` reads `FRED_API_KEY` from `expected-transaction-price/.env` for FRED macro data; without it, macro features come back empty (training still works).

## Architecture

### Producer/consumer coupling via ArtifactStore

`geographic_feature_store` is the **producer** and `expected-transaction-price` is the **consumer**. They never call each other directly — coordination happens entirely through Geronimo's `ArtifactStore`, keyed by project name and version:

- The batch pipeline writes 12 velocity grids (`geo_velocity_r{R}_w{W}` for R∈{1,5,10} miles, W∈{30,90,180,365} days) plus a `features_config` artifact under `project="geographic-feature-store", version="1.0.0"`.
- `expected-transaction-price`'s feature transform calls `ArtifactStore(project="geographic-feature-store", version="1.0.0").get(artifact_name)` and indexes the resulting DataFrame by `h3_index` for O(1) lookup at training and inference time. See `expected-transaction-price/src/expected_transaction_price/sdk/features.py` — `_load_geo_velocity_store()` and `_h3_lookup()`.

If you change the producer's artifact names, H3 resolution, or schema, the consumer breaks silently (NaN features). Coordinate both ends — the producer's `GEO_VELOCITY_CONFIGS` (in `geographic_feature_store/.../features.py`) and the consumer's `GEO_VELOCITY_CONFIGS` (in `expected-transaction-price/.../features.py`) must agree.

### Geronimo SDK conventions

Both projects follow the same Geronimo template:

```
src/<project>/sdk/
├── data_sources.py   # DataSource definitions, auto-collected by prefix
├── features.py       # FeatureSet declaring Feature(...) attributes
├── model.py          # Model subclass implementing train/predict/save/load
├── endpoint.py       # (expected-transaction-price only) Endpoint with preprocess/postprocess
└── pipeline.py       # (feature store only) BatchPipeline orchestrating train+save
```

**Data sources are auto-collected by name prefix** via `collect_data_sources(sys.modules[__name__], "training_")` and `"production_"` at the bottom of `data_sources.py`. Adding a new source means defining a module-level `training_<name> = DataSource(...)` — no registration needed. `JoinSpec` on a non-primary source describes how it merges onto the first source.

**Features are declarative.** `FeatureSet` classes list `Feature(dtype=...)` class attributes; `dtype='derived'` features supply a `derived_feature_fn` that takes the DataFrame and returns a Series. The `CompModelFeatures.transform()` override exists because Geronimo's base `transform` strips pandas categorical dtypes that XGBoost requires — preserve that behavior when modifying.

### expected-transaction-price specifics

- **Rochester-only training** with log1p(sale_price) target. Buffalo loaders exist but aren't used by `model.train()` — see the comment block at the top of `sdk/model.py`.
- `CompModelEndpoint` has a **demo-mode fallback**: if no artifact is found, `initialize()` silently sets `self.model = None` and `handle()` echoes the request back. Untrained deploys serve 200s, not 500s — be aware when debugging.
- Training filters to sale_price ∈ [$20k, $1.5M] and sale_date ≥ 2015-01-01, then does an 80/20 random split stratified by price decile with XGBoost early stopping.
- The FastAPI app mounts the MCP server at `/mcp` only when `geronimo.yaml`'s `model.mcp_enabled` is true.

### geographic_feature_store specifics

- The "model" is the fitted `GeoVelocityFeatures` itself — there's no separate estimator. `train()` runs BallTree computations over every H3 cell in the metro bounds, once per declared `(radius, window)` Feature.
- Metro bounds are hardcoded in `sdk/features.py` (`DEFAULT_METRO_BOUNDS` for Buffalo + Rochester).
- The Metaflow `flow.py` is a thin wrapper — real logic is in `sdk/pipeline.py`'s `GeoFeatureStorePipeline.run()`.

### Data source quirks

- **Buffalo**: Socrata SODA API, paginated by `$offset`, filters to property classes 210–250. Half baths are not split out (hardcoded to 0).
- **Rochester**: ArcGIS REST API, paginated by `resultOffset`. Polygon centroids come in **EPSG:3857** and must be reprojected to EPSG:4326 — `_load_rochester_training_data` does this via `pyproj.Transformer`. School districts are assigned via spatial join against the TIGER 2023 NYS UNSD shapefile, cached at `/tmp/tiger_school_districts`.
- The `UNIFIED_COLUMNS` list in `expected-transaction-price/sdk/data_sources.py` is the contract — every source loader normalizes to that schema before the feature layer sees it.


# Development Philosophy

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.