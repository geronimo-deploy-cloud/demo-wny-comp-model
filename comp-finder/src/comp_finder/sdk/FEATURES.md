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
