# Ticket 7b — Activating the geographic velocity family: before/after

Two `comp_quality real` runs on the identical protocol (population = 10,123
Buffalo + Rochester sales in $100k–$600k, sale_date > 2022-01-01; 200
leave-one-out queries; k = 5; seed 42):

- `ticket7b_real_before.json` — comp venv **without** the
  `expected_transaction_price` dependency (geographic + macro families
  INACTIVE; physical-only 12-dim vector). Matches the Ticket 5 baseline
  (spearman 0.68–0.77, within-15% 0.54–0.57).
- `ticket7b_real_after.json` — venv **with** the path dependency and the
  `geographic-feature-store` 1.0.0 artifacts resolving live (geographic
  ACTIVE; macro still INACTIVE — no `FRED_API_KEY` in this environment).

## Experiment 1 — holdout price prediction (real `CompFinder`)

| run | spearman | MAPE | median APE | within-10% | within-15% | within-20% |
|---|---|---|---|---|---|---|
| before (physical-only) | 0.775 | 0.178 | 0.145 | 0.380 | 0.505 | 0.640 |
| after (physical + geographic) | 0.762 | 0.189 | 0.156 | 0.340 | 0.475 | 0.635 |

## Experiment 2 — family ablation (after)

| config | spearman | MAPE | within-15% |
|---|---|---|---|
| physical only | 0.775 | 0.178 | 0.505 |
| geographic only | 0.602 | 0.255 | 0.410 |
| physical + geographic | 0.762 | 0.189 | 0.475 |
| all (default), macro only, physical + macro | skipped — macro INACTIVE (needs `FRED_API_KEY`) | | |

## Interpretation

- The wiring works: `comp_finder.sdk.features._geo_vol_r1_w90` resolves to
  the regressor's H3 lookup, the geographic family is ACTIVE on real data
  (volume columns resolve for ~100% of rows, price/sqft columns for ~67%,
  misses median-imputed), and all 92 comp-finder tests pass.
- The real-data headline is **negative**: adding the geographic family at
  the default unit weight slightly *degrades* holdout accuracy on both
  cities (Rochester within-15% 0.427 → 0.399; Buffalo 0.702 → 0.667).
  The synthetic result that motivated this ticket (physical + geographic
  strongest) does not transfer as-is.
- Geographic features alone still carry real signal (spearman 0.602,
  within-15% 0.410), but that number is likely optimistic: the grid
  medians are computed from realized sale prices that *include the query
  sale's own window* (and the store snapshot, `reference_date`
  2026-04-21, lags the population). Distance in the comp vector does not
  de-leak the query row the way the leave-one-out index rebuild does.
- Follow-up (explicitly out of scope here): family weight tuning /
  geo-feature leakage hardening, and joining FRED once a `FRED_API_KEY`
  is available to run the remaining three ablation configs.
- The store-missing fallback holds for the experiments: with the
  geographic-feature-store artifacts removed, `comp_quality real` reports
  `geographic` INACTIVE and completes (no crash). Raw `CompFinder`
  queries are **not** yet fail-safe on unresolvable geo cells
  (`ValueError: Input contains NaN`) — that is open Ticket 6 (#17/#18),
  which this ticket flagged as a dependency.
