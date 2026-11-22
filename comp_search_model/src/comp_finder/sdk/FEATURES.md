# Feature Ownership — expected-transaction-price is the Source of Truth

## The Rule

**`expected_transaction_price.sdk.features` is the single source of truth for all feature names used across models in this workspace.**

Every other model (comp-finder, and any future model) must import the canonical feature lists from that project rather than declaring them locally.

## Canonical Lists

The following lists are exported from `expected_transaction_price.sdk.features`:

| Symbol | Contents | Description |
|---|---|---|
| `PHYSICAL_FEATURES` | 12 feature names | Numeric property measurements (area, beds, lot, year_built, etc.) |
| `MACRO_FEATURES` | 4 feature names | FRED macro-economic indicators (mortgage, fed_funds, cpi, unemployment) |
| `GEO_VELOCITY_FEATURES` | 12 feature names | Derived from geographic feature store (H3-lookup) |
| `CATEGORICAL_FEATURES` | 6 feature names | High-cardinality nominal attributes (excluded from comp vector) |
| `ALL_VECTOR_FEATURES` | 28 feature names | `PHYSICAL + MACRO + GEO_VELOCITY` |

Any model that consumes property data should import these lists:

```python
from expected_transaction_price.sdk.features import (
    PHYSICAL_FEATURES,
    MACRO_FEATURES,
    GEO_VELOCITY_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_VECTOR_FEATURES,
)
```

## When to Deviate

It is acceptable to define features **locally** in a consumer model when:

1. **The feature is unique to that model's use case** — no other model would ever use it. In this case, declare it directly in the consumer's `FeatureSet` class (as `Feature(dtype=...)`) and add a comment referencing the canonical lists so a future PR can move it up.

2. **The regressor project is not installed** (standalone testing, CI without the full workspace). The consumer uses local fallback copies that must match the canonical lists exactly. Tests that assert `len(ALL_VECTOR_FEATURES) == 28` will catch drift.

**Never** add a new feature to a consumer's local copy without also adding it to `expected_transaction_price/sdk/features.py` first. The canonical lists are the arbiter — consumers are consumers.

## How It Works

The import chain uses a try/except pattern:

```python
try:
    from expected_transaction_price.sdk.features import (
        PHYSICAL_FEATURES,
        MACRO_FEATURES,
        GEO_VELOCITY_FEATURES,
        CATEGORICAL_FEATURES,
        ALL_VECTOR_FEATURES,
    )
except ImportError:
    # Fallback for standalone testing — must match canonical lists exactly.
    PHYSICAL_FEATURES = [...]  # must stay in sync
    ...
```

If someone adds a feature to the canonical lists but forgets to update the fallback in a consumer, the length-assertion tests will fail, flagging the drift.

## Adding a New Feature

When a new feature is needed:

1. **Add it to `expected_transaction_price/sdk/features.py`** — either to the appropriate list (e.g., `PHYSICAL_FEATURES.append("new_col")`) or to the `FeatureSet` class as a `Feature(dtype=...)` declaration.

2. **Verify consumers pick it up** — run the consumer's tests. The `test_vector_length_matches_feature_count` test will fail if the canonical list grew but the consumer's fallback didn't.

3. **Update the fallback** — if the fallback lists in the consumer don't already include the new feature, add them. The fallback must always match the canonical lists exactly.

4. **Document the exception** — if the feature is only used by one consumer (not the regressor), add a comment in the consumer's `FeatureSet` class noting the feature and why it's local.

## Future Models

Any new model added to this workspace should follow the same pattern:

- Import canonical feature lists from `expected_transaction_price.sdk.features`.
- Declare model-specific features locally in its own `FeatureSet`.
- Add a test that asserts the consumer's fallback lists match the canonical lists (by length and by feature name).
- Update this document if the model has unique feature conventions.

## Fail-Safe NaN Policy (Inactive Feature Families)

The geographic velocity features are derived functions imported from `expected_transaction_price` — in a venv without that package (or whenever the geographic feature store is down) they emit NaN. The macro features are NaN when no FRED data was joined. Raw NaN reaches `BallTree` as a hard crash ("Input contains NaN") at both index build and query time, so the comp-finder ships a fail-safe policy (Ticket 6a, options 1 + 2):

1. **Inactive families are zero-filled.** A family (`physical` / `macro` / `geographic`) whose comp-vector columns are *entirely* NaN is declared INACTIVE and zero-filled before standardization, so it contributes (a constant that) nothing to pairwise distances. This is applied identically at index build (`CompSearchPipeline.run`) and at query time (`CompFinder.find_comps`), matching the semantics `experiments/comp_quality.py` uses. The build logs the zeroed families and reports them in its run summary under `inactive_families`.
2. **Per-row NaNs are median-imputed.** NaNs inside otherwise-active families (e.g. one sale with a missing `year_built`) are imputed with per-column population medians. Medians are computed at build time and persisted alongside the index as the `comp_vector_imputation_medians` artifact; queries loaded from that store apply the same medians so a query row is treated exactly like its population counterpart. Indexes built before this policy have no medians artifact — `CompFinder` then falls back to build-time medians from the query matrix (0.0 for all-NaN columns).

When no NaNs are present anywhere the policy is a no-op: vectors are byte-identical to pre-policy behavior. `CompFinderFeatures.build_comp_vector()` itself stays NaN-transparent (it is a pure vectorizer); the policy lives in `sanitize_comp_features()` in `sdk/features.py`, called by both the build and query paths. A companion `ensure_store_feature_columns()` materializes missing macro/geo input columns as NaN before `transform()` (which requires every declared column), so inputs that never joined FRED/geo data reach the inactive-family logic instead of erroring; missing *physical* columns still raise the normal required-feature error.

**Known limitation:** a query whose geo columns are zero-filled against an index built with *live* geo values (or vice versa) compares different environments — the zero-filled columns carry a shared constant offset rather than a true zero contribution. Build and query must therefore come from the same store; mixing a full-environment index with bare-venv queries degrades (but does not crash) ranking quality.
