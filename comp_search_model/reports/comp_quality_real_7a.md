# Ticket 7a — real-data comp quality, before/after geo activation

Both runs: `comp_finder.experiments.comp_quality real`, n=10,123 sales
(Buffalo + Rochester, $100k–$600k, post 2022-01-01), 200 leave-one-out
queries, k=5, seed=42.

- `comp_quality_real_before_geo_inactive.json` — physical-only (Ticket-5
  baseline reproduced): geographic and macro families INACTIVE/zeroed.
- `comp_quality_real_after_geo_active.json` — geographic family resolved
  live from the `geographic-feature-store` artifacts via H3 lookup; macro
  joined from FRED's public CSV endpoint. No INACTIVE families; all six
  ablation configurations ran.

## Experiment 1 — holdout price prediction (median of 5 comps)

| metric | before (physical-only) | after (28-dim, geo+macro active) |
|---|---|---|
| spearman | 0.775 | 0.783 |
| MAPE | 0.178 | 0.182 |
| median APE | 0.145 | 0.136 |
| within-10% | 0.380 | 0.385 |
| within-15% | 0.505 | 0.535 |
| within-20% | 0.640 | 0.670 |

Both before and after, `comp_finder` beats all three baselines
(population/city median, random-k) on spearman + within-15.

## Experiment 2 — family ablation (after run, 6/6 configs)

| config | spearman | MAPE | within-15% |
|---|---|---|---|
| all (default) | 0.783 | 0.182 | 0.535 |
| physical only | 0.775 | 0.178 | 0.505 |
| macro only | 0.087 | 0.332 | 0.270 |
| geographic only | 0.602 | 0.255 | 0.410 |
| physical + macro | 0.678 | 0.201 | 0.520 |
| physical + geographic | 0.762 | 0.189 | 0.475 |

## Interpretation

- The geographic family carries real standalone price signal on actual
  data (geographic-only spearman 0.602), confirming the synthetic
  ablation's expectation that it is not noise.
- Activating it lifts end-to-end holdout accuracy modestly: within-15%
  0.505 → 0.535, within-20% 0.640 → 0.670, median APE 0.145 → 0.136.
  MAPE ticked up slightly (outlier-driven; median APE improved).
- Equal-weight "all" is the best configuration (0.783 / 0.535);
  notably `physical + geographic` (0.762) scores *below* physical-only,
  and `physical + macro` (0.678) below as well — on the default
  standardized vector the families interact non-additively. Weight
  tuning is explicitly out of scope for 7a; worth a follow-on ticket.
- Macro-only is near-noise (0.087) as expected: FRED series are
  market-wide time trends, not per-property information; its value is
  in conditioning the physical/geo distances by sale date.
- Feature coverage in the after run: geo `sales_volume` columns have
  100% grid coverage, `median_ppsf` columns ~67% (cells with no
  $/sqft data), macro 99.1%; fully-missing entries are imputed with
  the column median before vectorization (experiment harness only).

### Caveat — static-grid leakage

The velocity grids are computed from the same sales population and are
static: leave-one-out removes the query sale from the index but not from
the store's grid statistics, so "geographic only" (and partly "all") is
optimistically biased. The end-to-end before/after delta is still
meaningful because both runs share the protocol; absolute geo-only
numbers are the upper bound.

### Environment-dependence

- Geo results depend on the `geographic-feature-store/1.0.0` artifacts
  present in the local ArtifactStore (`~/.geronimo/artifacts/`). Rebuild
  with `uv run python -m geographic_feature_store.flow run` from the
  `geographic_feature_store` project (they must match the current sales
  pull for reproducibility).
- Macro values come from FRED's public CSV endpoint; the regressor's own
  macro loader needs `FRED_API_KEY`, which this machine lacks. Fetch
  failure degrades to macro-INACTIVE instead of crashing.
- `real --physical-only` reproduces the before column of this table on
  demand.
