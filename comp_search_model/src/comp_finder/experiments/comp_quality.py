"""Empirical quality experiments for the CompFinder comp-vector index.

Answers the question: *is the comp finder actually finding comparable
sales, or would a coin flip do as well?*

Two subcommands:

    synthetic   Self-contained sanity check.  Generates a synthetic sales
                population whose price is a known function of physical +
                geographic attributes, runs the experiments on it, and
                re-runs them on the same population with randomized prices
                (negative control).  No network access.

    real        Loads the combined Buffalo + Rochester sales population
                (network; cached as a pickle), and runs the same
                experiments against the real CompFinder.

Experiments
-----------
1. Holdout price prediction (leave-one-out).
   For each sampled query sale, rebuild the index *excluding* that sale,
   publish it to a scratch ArtifactStore, ask the real ``CompFinder`` for
   its top-k comps, and predict the query's price as the median comp
   price.  Compare against three baselines — population median,
   same-city median, random-k median — using Spearman correlation,
   MAPE, and within-X% hit rates.  A worthwhile finder must clearly
   beat all three.

2. Feature-family ablation.
   Same leave-one-out protocol, computed directly in vector space (no
   per-query store round-trips), under different family weights
   (physical / macro / geographic) to see which families carry signal.

Caveats (surfaced in the output):
- The 12 geographic velocity features are derived functions resolved
  from ``expected_transaction_price`` against the geographic-feature-
  store ArtifactStore.  In ``real`` mode the store lookup is used
  directly (geo family ACTIVE) whenever the regressor package is
  installed.  Where the package or its artifacts are unavailable the
  features emit NaN and this script treats the geographic family as
  INACTIVE: NaNs are replaced with 0.0 pre-standardization (zero
  contribution) and ablation configs that use the family are skipped.
  Synthetic mode injects synthetic geo values via the same identity-
  passthrough mechanism the tests use (see the ``geo_passthrough``
  fixture in tests/test_finder.py).
- Macro features are joined in ``real`` mode from the public FRED CSV
  endpoint (no API key needed; see ``fetch_macro_daily``).  They are
  zeroed when the fetch fails or when absent from the input.
- Query rows with missing feature values are imputed with the same
  per-column population medians used when building the index, so a
  query vector always matches how its population row was treated.

Usage:
    uv run python -m comp_finder.experiments.comp_quality synthetic
    uv run python -m comp_finder.experiments.comp_quality real --limit 2000
    uv run python -m comp_finder.experiments.comp_quality real --physical-only
    uv run python -m comp_finder.experiments.comp_quality synthetic --json out.json
"""

import argparse
import contextlib
import json
import logging
import math
import sys
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from geronimo.artifacts import ArtifactStore

from ..pipeline import (
    INDEX_ARTIFACT_NAME,
    INDEX_PROJECT,
    INDEX_VERSION,
    LOOKUP_ARTIFACT_NAME,
    UNIFIED_COLUMNS,
)
from ..sdk import features as features_module
from ..sdk.features import (
    ALL_VECTOR_FEATURES,
    CompFinderFeatures,
    GEO_VELOCITY_FEATURES,
    MACRO_FEATURES,
    PHYSICAL_FEATURES,
    SCALER_ARTIFACT_NAME,
)
from ..sdk.model import CompFinder

logger = logging.getLogger(__name__)

FAMILY_COLUMNS = {
    "physical": list(PHYSICAL_FEATURES),
    "macro": list(MACRO_FEATURES),
    "geographic": list(GEO_VELOCITY_FEATURES),
}

#: Where ``real`` mode caches the combined population between runs.
DEFAULT_CACHE_PATH = Path("/tmp/comp_quality_cache/wny_combined_sales.pkl")

#: Family-weight configurations for the ablation experiment.
ABLATION_CONFIGS = [
    ("all (default)", {"physical": 1.0, "macro": 1.0, "geographic": 1.0}),
    ("physical only", {"physical": 1.0, "macro": 0.0, "geographic": 0.0}),
    ("macro only", {"physical": 0.0, "macro": 1.0, "geographic": 0.0}),
    ("geographic only", {"physical": 0.0, "macro": 0.0, "geographic": 1.0}),
    ("physical + macro", {"physical": 1.0, "macro": 1.0, "geographic": 0.0}),
    ("physical + geographic", {"physical": 1.0, "macro": 0.0, "geographic": 1.0}),
]


# ---------------------------------------------------------------------------
# Geo passthrough (mirrors the geo_passthrough fixture in test_finder.py)
# ---------------------------------------------------------------------------

def install_geo_passthrough():
    """Install identity passthroughs on the geo derived features.

    The real derived functions resolve H3 cells against the geographic
    feature store (an optional dependency); when it is unavailable they
    emit NaN, which crashes BallTree.  Passthroughs let an input column
    flow through unchanged — synthetic mode fills those columns, and
    ``real --physical-only`` ends up with NaN -> 0.0 (see
    prepare_population).  See ``geo_mode`` for when this is installed.

    Returns:
        A zero-argument restore callable that puts the original
        functions back (important in tests, where the Feature objects
        are class attributes shared across the whole session).
    """
    originals = {}
    for name in GEO_VELOCITY_FEATURES:
        feature = CompFinderFeatures.__dict__[name]
        originals[name] = feature._derived_feature_fn
        feature._derived_feature_fn = (lambda col: (lambda df: df[col]))(name)

    def restore():
        for name, fn in originals.items():
            CompFinderFeatures.__dict__[name]._derived_feature_fn = fn

    return restore


#: Geo derived feature functions as imported into this package's feature
#: module — None when the regressor package is not installed.
_GEO_FNS = [
    getattr(features_module, f"_{name}", None) for name in GEO_VELOCITY_FEATURES
]

#: Real-mode macro source: FRED public CSV endpoint (no API key required).
#: Mirrors the series mapped by the regressor's ``_load_macro_economic_data``
#: (which needs FRED_API_KEY); the values are the same published series.
FRED_PUBLIC_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MACRO_SERIES_IDS = {
    "mortgage_30y": "MORTGAGE30US",
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
}


def resolve_geo_mode(use_store: bool) -> str:
    """Decide how the geo velocity features get their values.

    "store"      — the real H3-lookup derived functions run against the
                   geographic-feature-store (geographic family active,
                   provided the store artifacts exist; if they don't the
                   features emit NaN and are treated as INACTIVE, which
                   is the graceful-fallback path).
    "passthrough" — identity passthroughs (synthetic mode, and real mode
                   with ``--physical-only`` or without the regressor
                   package installed).
    """
    if not use_store:
        return "passthrough"
    if all(fn is not None for fn in _GEO_FNS):
        return "store"
    logger.warning(
        "Geo features requested from the feature store but "
        "expected_transaction_price is not installed — falling back to "
        "passthrough (geographic family will be INACTIVE)."
    )
    return "passthrough"


@contextlib.contextmanager
def geo_mode(use_store: bool = False):
    """Context manager choosing how the geo derived features get values.

    "store" mode wraps the real feature-store lookups with the
    input-column fallback (``install_store_geo_fallback``); otherwise the
    identity passthrough is installed (the pre-store behavior).
    """
    mode = resolve_geo_mode(use_store)
    restore = install_store_geo_fallback() if mode == "store" else install_geo_passthrough()
    try:
        yield mode
    finally:
        restore()


def install_store_geo_fallback():
    """Wrap the real geo store lookups with an input-column fallback.

    ``CompFinder.find_comps`` re-runs the derived functions on every
    query row, and the store returns NaN for properties outside grid
    coverage — a NaN comp vector crashes BallTree.  The wrapper keeps
    the live lookup but falls back to the query row's pre-imputed value
    (see ``_query_row``) wherever the lookup is NaN, and to 0.0-
    /median-imputed columns on population rows (see prepare_population).

    Returns a restore callable, mirroring install_geo_passthrough.
    """
    originals = {}
    for name in GEO_VELOCITY_FEATURES:
        feature = CompFinderFeatures.__dict__[name]
        originals[name] = feature._derived_feature_fn
        real_fn = getattr(features_module, f"_{name}")

        def fn(df, real_fn=real_fn, name=name):
            vals = real_fn(df)
            if name in df.columns:
                vals = vals.where(
                    ~pd.isna(vals), pd.to_numeric(df[name], errors="coerce")
                )
            return vals

        feature._derived_feature_fn = fn

    def restore():
        for name, fn in originals.items():
            CompFinderFeatures.__dict__[name]._derived_feature_fn = fn

    return restore


def fetch_macro_daily(start: str = "2021-06-01") -> Optional[pd.DataFrame]:
    """Fetch the 4 macro series from FRED's public CSV endpoint.

    Weekly/monthly observations are forward-filled to daily so they can
    be mapped onto sale dates (mirrors the regressor's macro loader).

    Returns:
        Daily-indexed DataFrame with the MACRO_FEATURES columns, or None
        if no series could be fetched.
    """
    frames = []
    for col, series_id in MACRO_SERIES_IDS.items():
        try:
            resp = requests.get(
                FRED_PUBLIC_CSV_URL,
                params={"id": series_id, "cosd": start},
                timeout=60,
            )
            resp.raise_for_status()
            raw = pd.read_csv(BytesIO(resp.content), na_values=["."])
            values = pd.to_numeric(raw[raw.columns[-1]], errors="coerce")
            values.index = pd.to_datetime(raw[raw.columns[0]])
            values = values[~values.index.duplicated(keep="last")].sort_index()
            frames.append(
                pd.DataFrame({col: values.resample("D").ffill()})
            )
        except Exception as e:
            logger.warning("Failed to fetch FRED %s: %s", series_id, e)
    if not frames:
        return None
    macro = pd.concat(frames, axis=1, sort=True)
    logger.info("Fetched FRED macro data: %d daily rows", len(macro))
    return macro


# ---------------------------------------------------------------------------
# Population preparation / index publishing
# ---------------------------------------------------------------------------

def _fit_template() -> pd.DataFrame:
    """An empty frame declaring every feature column (same trick
    CompFinder uses: FeatureSet.fit() is only a fitted-marker here)."""
    columns = (
        list(PHYSICAL_FEATURES) + list(MACRO_FEATURES)
        + list(ALL_VECTOR_FEATURES) + ["latitude", "longitude"]
    )
    return pd.DataFrame(columns=dict.fromkeys(columns, np.nan))


def prepare_population(df: pd.DataFrame):
    """Run the CompFinderFeatures transform and sanitize the output.

    Missing input columns are added as NaN before transform.  After
    transform, a feature family whose columns are entirely NaN is
    declared INACTIVE and zeroed (zero contribution after
    standardization); remaining per-row NaNs are imputed with the
    column median, and a column that is entirely missing within an
    active family (store grid coverage varies by radius/window) is
    imputed with 0.0.

    Returns:
        (features, transformed_df, inactive_family_names, medians) where
        ``medians`` maps each comp-vector feature to the imputation
        median used (0.0 for inactive columns).
    """
    prep = df.copy()
    for col in list(ALL_VECTOR_FEATURES) + ["latitude", "longitude"]:
        if col not in prep.columns:
            prep[col] = np.nan

    features = CompFinderFeatures()
    features.fit(_fit_template())
    transformed = features.transform(prep)

    inactive = []
    for family, cols in FAMILY_COLUMNS.items():
        if transformed[cols].isna().all(axis=None):
            transformed[cols] = 0.0
            inactive.append(family)

    matrix = transformed[ALL_VECTOR_FEATURES].astype(np.float64)
    medians = matrix.median()
    n_nans = int(matrix.isna().sum().sum())
    if n_nans:
        logger.info("Imputing %d remaining NaN feature values with column medians", n_nans)
        matrix = matrix.fillna(medians)
    # Columns whose median is itself NaN (entirely missing within an
    # active family — sparse grid coverage) fall back to 0.0.
    matrix = matrix.fillna(0.0)
    transformed[ALL_VECTOR_FEATURES] = matrix

    return features, transformed, inactive, medians.fillna(0.0).to_dict()


def publish_index_artifacts(population: pd.DataFrame, store_dir: str,
                            family_weights: Optional[dict] = None) -> list[str]:
    """Build the index for ``population`` and publish all three artifacts
    (BallTree, lookup table, scaler) to a scratch local ArtifactStore.

    Mirrors CompSearchPipeline._publish + CompFinderModel's scaler save,
    but into ``store_dir`` instead of the production store.

    Returns:
        (inactive_family_names, imputation_medians)
    """
    features, transformed, inactive, medians = prepare_population(population)

    scaler = StandardScaler().fit(transformed[ALL_VECTOR_FEATURES].values)
    features.set_scaler(scaler)
    vectors = np.atleast_2d(features.build_comp_vector(transformed, family_weights))
    tree = BallTree(vectors, metric="euclidean")

    lookup_cols = [c for c in UNIFIED_COLUMNS if c in population.columns]
    lookup = population[lookup_cols].copy()

    store = ArtifactStore(project=INDEX_PROJECT, version=INDEX_VERSION, base_path=store_dir)
    buf = BytesIO()
    joblib.dump(tree, buf)
    store.save(INDEX_ARTIFACT_NAME, buf.getvalue())
    store.save(LOOKUP_ARTIFACT_NAME, lookup)
    store.save(SCALER_ARTIFACT_NAME, scaler)
    return inactive, medians


def _query_row(population: pd.DataFrame, pos: int, medians: dict) -> dict:
    """One query as a dict for CompFinder.find_comps.

    NaN feature values are imputed with the same medians used when the
    index was built (0.0 for inactive families), so the query vector is
    treated exactly like its population row.
    """
    row = population.iloc[pos].to_dict()
    for col in ALL_VECTOR_FEATURES:
        if pd.isna(row.get(col, np.nan)):
            row[col] = float(medians.get(col, 0.0))
    return row


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def price_metrics(actual, predicted) -> dict:
    """Prediction-quality metrics for a price predictor."""
    # Positional numpy alignment — callers pass parallel sequences and
    # pandas index alignment would silently misalign them.
    a_all = np.asarray(actual, dtype=float)
    p_all = np.asarray(predicted, dtype=float)
    mask = ~np.isnan(a_all) & ~np.isnan(p_all) & (a_all > 0)
    a, p = a_all[mask], p_all[mask]
    if len(a) < 2:
        return {"n": int(len(a))}
    pct_err = np.abs(p - a) / a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant input → NaN correlation
        spearman = pd.Series(a).corr(pd.Series(p), method="spearman")
    return {
        "n": int(len(a)),
        "spearman": float(spearman),
        "mape": float(pct_err.mean()),
        "median_ape": float(np.median(pct_err)),
        "within_10": float((pct_err <= 0.10).mean()),
        "within_15": float((pct_err <= 0.15).mean()),
        "within_20": float((pct_err <= 0.20).mean()),
    }


# ---------------------------------------------------------------------------
# Experiment 1 — holdout price prediction through the real CompFinder
# ---------------------------------------------------------------------------

def run_holdout(population: pd.DataFrame, query_positions: list[int], k: int,
                seed: int, use_store_geo: bool = False) -> dict:
    """Leave-one-out price prediction through the real CompFinder.

    For each query sale: rebuild + republish the index without that
    sale, run ``CompFinder.find_comps`` on it, and predict its price as
    the median comp price.  Also computes three baseline predictions.
    """
    rng = np.random.default_rng(seed + 1)
    actual, comp_pred, mean_sim, sources = [], [], [], []
    pop_median, city_median, random_k = [], [], []

    # Constant market baselines (full population — they don't depend on
    # the query).  A leave-one-out median would vary by ~1/n and produce
    # a spurious rank correlation.
    prices_all = population["sale_price"].astype(float)
    overall_median = float(prices_all.median())
    city_medians = (
        population.groupby("city")["sale_price"].median().to_dict()
        if "city" in population.columns else {}
    )

    with geo_mode(use_store_geo):
        with tempfile.TemporaryDirectory(prefix="comp_quality_") as tmp:
            for j, pos in enumerate(query_positions):
                if j % 10 == 0:
                    logger.info("holdout query %d/%d", j, len(query_positions))
                inactive, medians = publish_index_artifacts(population.drop(index=pos), tmp)
                finder = CompFinder(base_path=tmp)
                comps = finder.find_comps(_query_row(population, pos, medians), k=k)
                comp_prices = [c["sale_price"] for c in comps if c["sale_price"] is not None]
                comp_pred.append(float(np.median(comp_prices)) if comp_prices else np.nan)
                mean_sim.append(float(np.mean([c["similarity"] for c in comps])) if comps else np.nan)

                price = float(population.iloc[pos]["sale_price"])
                actual.append(price)
                sources.append(population.iloc[pos].get("source"))

                pop_median.append(overall_median)
                city = population.iloc[pos].get("city")
                city_median.append(float(city_medians.get(city, overall_median)))
                others_prices = population.drop(index=pos)["sale_price"].astype(float)
                random_k.append(float(others_prices.sample(
                    min(k, len(others_prices)),
                    random_state=int(rng.integers(2**31))).median()))

    results = pd.DataFrame({
        "actual": actual, "comp_finder": comp_pred, "mean_similarity": mean_sim,
        "source": sources, "population_median": pop_median,
        "city_median": city_median, "random_k": random_k,
    })
    # Group by city when available — the real data's `source` column is
    # unreliable (Buffalo rows carry NaN there).
    group_col = "city" if results["source"].isna().any() and "city" in population.columns else "source"
    results[group_col] = population.iloc[query_positions][group_col].to_numpy()

    summary = {method: price_metrics(results["actual"], results[method])
               for method in ["comp_finder", "population_median", "city_median", "random_k"]}

    # Sanity signal: closer comps (higher mean similarity) should have
    # smaller price errors.  Expect a negative correlation.
    err = ((results["comp_finder"] - results["actual"]).abs() / results["actual"])
    summary["similarity_vs_error_spearman"] = float(
        results["mean_similarity"].corr(err, method="spearman"))

    summary["per_source"] = {
        str(src): price_metrics(g["actual"].to_numpy(), g["comp_finder"].to_numpy())
        for src, g in results.groupby(group_col)
    }
    return summary


# ---------------------------------------------------------------------------
# Experiment 2 — feature-family ablation (direct vector space)
# ---------------------------------------------------------------------------

def run_ablation(population: pd.DataFrame, query_positions: list[int], k: int,
                 use_store_geo: bool = False) -> tuple[list, list]:
    """Leave-one-out price prediction under different family weights.

    Same protocol as the holdout experiment but computed directly on the
    comp vectors (one tree per config; self-match excluded by querying
    k+1 and dropping the query itself) — no per-query store round-trips.

    Returns:
        (rows, inactive_families) — one metrics dict per config, with
        {"config": name, "skipped": True} for configs that need an
        inactive family.
    """
    with geo_mode(use_store_geo):
        features, transformed, inactive, _ = prepare_population(population)
        scaler = StandardScaler().fit(transformed[ALL_VECTOR_FEATURES].values)
        features.set_scaler(scaler)
        prices = population["sale_price"].astype(float)

        rows = []
        for name, weights in ABLATION_CONFIGS:
            if any(weights.get(fam, 0.0) > 0 and fam in inactive for fam in FAMILY_COLUMNS):
                rows.append({"config": name, "skipped": True})
                continue
            vectors = np.atleast_2d(features.build_comp_vector(transformed, weights))
            tree = BallTree(vectors, metric="euclidean")

            preds, d1s = [], []
            for pos in query_positions:
                dists, idxs = tree.query(vectors[pos : pos + 1], k=min(k + 1, len(vectors)))
                kept = [(float(d), int(i)) for d, i in zip(dists[0], idxs[0]) if i != pos][:k]
                preds.append(float(prices.iloc[[i for _, i in kept]].median()))
                d1s.append(kept[0][0] if kept else np.nan)

            metrics = price_metrics(prices.iloc[query_positions].to_numpy(), preds)
            metrics.update({"config": name, "mean_top1_distance": float(np.mean(d1s))})
            rows.append(metrics)
        return rows, inactive


# ---------------------------------------------------------------------------
# Synthetic population generator
# ---------------------------------------------------------------------------

#: (lat, lon, city, source, price premium, base $/sqft, sales activity)
_SYNTH_CLUSTERS = [
    (42.90, -78.88, "Buffalo", "buffalo", 0.85, 75.0, 30.0),
    (42.95, -78.75, "Buffalo", "buffalo", 1.10, 105.0, 45.0),
    (43.13, -77.65, "Rochester", "rochester", 1.00, 95.0, 40.0),
    (43.19, -77.52, "Rochester", "rochester", 1.45, 140.0, 25.0),
]


def generate_synthetic_population(n: int = 800, seed: int = 42,
                                  random_price: bool = False) -> pd.DataFrame:
    """Generate a synthetic sales population in the UNIFIED_COLUMNS schema.

    Price is a noisy function of living area, beds, and a per-cluster
    premium; the geo ``*_ppsf`` input columns carry the cluster's price
    level and the macro columns vary with sale date but do NOT affect
    price.  With ``random_price=True`` the price is drawn uniformly at
    random instead (negative control: every non-price column is
    identical to the signal population for the same seed).
    """
    rng = np.random.default_rng(seed)
    n_clusters = len(_SYNTH_CLUSTERS)
    cluster = rng.integers(0, n_clusters, n)
    c = np.array(_SYNTH_CLUSTERS, dtype=object)[cluster]
    premium = np.array([row[4] for row in _SYNTH_CLUSTERS], dtype=float)[cluster]
    ppsf_base = np.array([row[5] for row in _SYNTH_CLUSTERS], dtype=float)[cluster]
    activity = np.array([row[6] for row in _SYNTH_CLUSTERS], dtype=float)[cluster]

    # Physical attributes.  tla noise is kept modest so that physically
    # similar comps genuinely imply similar prices (the effect under test).
    tla = rng.lognormal(mean=7.2, sigma=0.20, size=n)
    beds = np.clip(np.round(tla / 550 + rng.normal(0, 0.6, n)), 1, 6)
    stories = rng.choice([1.0, 1.5, 2.0], size=n, p=[0.4, 0.2, 0.4])
    year_built = rng.integers(1900, 2015, n)
    lot_frontage = rng.uniform(25, 60, n)
    lot_depth = rng.uniform(80, 150, n)
    lot_acres = rng.uniform(0.05, 0.4, n)

    # Sale dates 2022-01-01 .. 2025-06-30.
    start = pd.Timestamp("2022-01-01")
    span_days = (pd.Timestamp("2025-06-30") - start).days
    sale_date = start + pd.to_timedelta(rng.integers(0, span_days, n), unit="D")
    t_frac = rng.integers(0, span_days, n) / span_days

    # Macro features: date-driven, deliberately UNRELATED to price.
    mortgage_30y = 3.5 + 3.5 * t_frac + rng.normal(0, 0.15, n)
    fed_funds = 0.5 + 4.5 * t_frac + rng.normal(0, 0.2, n)
    cpi = 280 + 35 * t_frac + rng.normal(0, 2, n)
    unemployment = 4.0 - 0.8 * t_frac + rng.normal(0, 0.2, n)

    # Geo velocity input columns: ppsf carries cluster price signal,
    # volume carries cluster activity.
    geo_cols = {}
    for col in GEO_VELOCITY_FEATURES:
        if "ppsf" in col:
            geo_cols[col] = ppsf_base * np.exp(rng.normal(0, 0.10, n))
        else:
            geo_cols[col] = np.clip(activity + rng.normal(0, 5, n), 0, None)

    # Price — the known ground-truth function (or the random control).
    # Both branches draw BOTH variates so the rng stream stays aligned:
    # every non-price column is identical between signal and control
    # for the same seed.
    fundamental = premium * 110.0 * tla * (1 + 0.08 * (beds - 3))
    signal_noise = np.exp(rng.normal(0, 0.10, n))
    random_draw = rng.uniform(80_000, 600_000, n)
    price = random_draw if random_price else fundamental * signal_noise

    # Assessment tracks the fundamental value, not the realized price —
    # in the random-price control this keeps assessment independent of
    # price (otherwise the assessment features would leak the control's
    # random price through the comp vector).
    assessed_value = fundamental * rng.uniform(0.4, 0.6, n)
    land_value = fundamental * rng.uniform(0.10, 0.25, n)

    df = pd.DataFrame({
        "source": [row[3] for row in c],
        "parcel_id": np.arange(1000, 1000 + n),
        "print_key": [f"PK-{i}" for i in range(n)],
        "address": [f"{i + 1} Synth St" for i in range(n)],
        "city": [row[2] for row in c],
        "zip_code": [f"14{i % 100:02d}01" for i in range(n)],
        "property_class": 210,
        "property_class_desc": "One Family Year-Round Residence",
        "sale_price": price,
        "sale_date": sale_date,
        "assessed_value": assessed_value,
        "land_value": land_value,
        "year_built": year_built,
        "total_living_area": tla,
        "first_floor_area": tla / stories,
        "second_floor_area": np.maximum(tla - tla / stories, 0.0),
        "beds": beds,
        "baths": rng.integers(1, 4, n),
        "half_baths": 0,
        "kitchens": 1,
        "stories": stories,
        "fireplaces": rng.integers(0, 3, n),
        "lot_frontage": lot_frontage,
        "lot_depth": lot_depth,
        "lot_acres": lot_acres,
        "building_style": "Colonial",
        "overall_condition": rng.choice(["Average", "Good"], size=n),
        "construction_grade": rng.choice(["Standard", "Above Average"], size=n),
        "exterior_wall": "Vinyl",
        "heat_type": "Gas",
        "central_air": rng.choice(["Y", "N"], size=n),
        "fuel_type": "Gas",
        "basement_type": "Full",
        "latitude": np.array([row[0] for row in c], dtype=float) + rng.normal(0, 0.02, n),
        "longitude": np.array([row[1] for row in c], dtype=float) + rng.normal(0, 0.02, n),
        "school_district": [f"District {row[2]}" for row in c],
        "mortgage_30y": mortgage_30y,
        "fed_funds": fed_funds,
        "cpi": cpi,
        "unemployment": unemployment,
        **geo_cols,
    })
    return df


# ---------------------------------------------------------------------------
# Real population loader
# ---------------------------------------------------------------------------

def load_real_population(limit: int = 0, cache_path: Path = DEFAULT_CACHE_PATH,
                         join_macro: bool = True) -> pd.DataFrame:
    """Load (or read from cache) the combined Buffalo + Rochester sales.

    Applies the documented comp-finder population filter: sale_price in
    [$100k, $600k] and sale_date after 2022-01-01.  When ``join_macro``
    is set, the 4 macro series are joined from FRED's public endpoint
    onto each sale's date (the combined data loader, like the one in
    the regressor project, does not carry macro columns).
    """
    if cache_path.exists():
        logger.info("Reading cached population from %s", cache_path)
        df = pd.read_pickle(cache_path)
    else:
        from ..sdk.data_sources import _load_combined_training_data
        df = _load_combined_training_data()
        if df.empty:
            raise SystemExit("No sales data loaded from data sources.")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        logger.info("Cached %d sales to %s", len(df), cache_path)

    before = len(df)
    sale_date = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df[
        df["sale_price"].notna()
        & (df["sale_price"] >= 100_000) & (df["sale_price"] <= 600_000)
        & sale_date.notna() & (sale_date > pd.Timestamp("2022-01-01"))
    ]
    logger.info("Population filter ($100k-$600k, >2022-01-01): %d -> %d sales", before, len(df))

    if limit and limit < len(df):
        df = df.sample(limit, random_state=0)
    df = df.reset_index(drop=True)

    if join_macro:
        macro = fetch_macro_daily()
        if macro is None:
            logger.warning(
                "Macro data unavailable — macro family will be INACTIVE."
            )
        else:
            sale_days = pd.to_datetime(df["sale_date"], errors="coerce").dt.normalize()
            aligned = macro.reindex(sale_days.to_numpy())
            aligned.index = df.index
            for col in MACRO_FEATURES:
                if col in aligned.columns:
                    df[col] = aligned[col].to_numpy()
            logger.info(
                "Macro join: %.1f%% of sales have values",
                100.0 * df[MACRO_FEATURES].notna().any(axis=1).mean(),
            )
    return df


# ---------------------------------------------------------------------------
# Driver + reporting
# ---------------------------------------------------------------------------

def run_all(population: pd.DataFrame, n_queries: int, k: int, seed: int,
            use_store_geo: bool = False) -> dict:
    """Run both experiments on one population and return the raw report.

    ``use_store_geo`` (real mode): resolve the geographic features
    through the geographic feature store instead of the synthetic
    passthrough.  Falls back to the passthrough when the regressor
    package is not installed.
    """
    population = population[
        population["sale_price"].notna() & (population["sale_price"].astype(float) > 0)
    ].reset_index(drop=True)

    if len(population) <= k + 1:
        raise SystemExit(f"Not enough query rows ({len(population)}).")
    rng = np.random.default_rng(seed)
    query_positions = sorted(
        int(p) for p in rng.choice(len(population), size=min(n_queries, len(population)), replace=False)
    )

    geo_mode_resolved = resolve_geo_mode(use_store_geo)
    holdout = run_holdout(population, query_positions, k, seed, use_store_geo)
    ablation, inactive = run_ablation(population, query_positions, k, use_store_geo)

    return {
        "population_size": int(len(population)),
        "n_queries": len(query_positions),
        "k": k,
        "seed": seed,
        "geo_mode": geo_mode_resolved,
        "inactive_families": inactive,
        "holdout": holdout,
        "ablation": ablation,
    }


def _metrics_table(rows: list[dict], label_col: str) -> str:
    cols = [label_col, "spearman", "mape", "median_ape", "within_10", "within_15", "within_20"]
    df = pd.DataFrame(rows)
    cols = [c for c in cols if c in df.columns]
    if "mean_top1_distance" in df.columns:
        cols.append("mean_top1_distance")
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:9.3f}")


def print_report(report: dict, mode: str, control: Optional[dict] = None) -> None:
    sep = "=" * 72
    print(f"\n{sep}\n CompFinder quality report — {mode}")
    print(f" population: {report['population_size']} sales | queries: "
          f"{report['n_queries']} | k: {report['k']} | seed: {report['seed']}")
    if report.get("geo_mode"):
        print(f" geo features: {report['geo_mode']}")
    if report["inactive_families"]:
        print(f" INACTIVE families (zeroed): {', '.join(report['inactive_families'])}")
    print(sep)

    h = report["holdout"]
    print("\nEXPERIMENT 1 — holdout price prediction (leave-one-out, real CompFinder)")
    print(" predicted price = median of the k comps' sale prices\n")
    rows = [{"method": f"comp_finder (k={report['k']})", **h["comp_finder"]},
            {"method": "baseline: population median", **h["population_median"]},
            {"method": "baseline: same-city median", **h["city_median"]},
            {"method": "baseline: random k sales", **h["random_k"]}]
    print(_metrics_table(rows, "method"))
    print(f"\n similarity vs price-error spearman: {h['similarity_vs_error_spearman']:.3f}"
          f"   (negative = closer comps have smaller errors)")
    if h["per_source"]:
        print("\n comp_finder per city/source:")
        src_rows = [{"group": s, **m} for s, m in h["per_source"].items() if "spearman" in m]
        print(_metrics_table(src_rows, "group"))

    wins = 0
    cf = h["comp_finder"]
    for b in ["population_median", "city_median", "random_k"]:
        bm = h[b]
        # A constant baseline has undefined rank correlation (NaN);
        # count the rank criterion as satisfied in that case.
        rank_ok = math.isnan(bm.get("spearman", float("nan"))) or cf["spearman"] > bm["spearman"]
        wins += rank_ok and cf["within_15"] >= bm["within_15"]
    print(f"\n verdict: comp_finder beats {wins}/3 baselines on spearman+within-15"
          f"\n (constant baselines have undefined spearman → shown as nan)")

    print("\nEXPERIMENT 2 — feature-family ablation (same queries, vector space)")
    ab_rows = []
    for row in report["ablation"]:
        if row.get("skipped"):
            ab_rows.append({"config": row["config"] + "  (skipped: family inactive)"})
        else:
            ab_rows.append({"config": row["config"], **row})
    print(_metrics_table(ab_rows, "config"))

    if control is not None:
        print(f"\n{sep}\n NEGATIVE CONTROL — same population, randomized prices")
        print(sep)
        c, s = control["holdout"]["comp_finder"], report["holdout"]["comp_finder"]
        print(f"\n signal   : spearman {s['spearman']:.3f}  within-15 {s['within_15']:.2f}")
        print(f" control  : spearman {c['spearman']:.3f}  within-15 {c['within_15']:.2f}")
        sensitive = (s["spearman"] > c["spearman"] + 0.2
                     and s["within_15"] > c["within_15"])
        print(f" verdict: measurement {'IS' if sensitive else 'is NOT'} sensitive to "
              f"real price structure (control should score near zero)\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-queries", type=int, default=200)
    common.add_argument("--k", type=int, default=5)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--json", type=Path, default=None, help="also write the raw report JSON here")

    syn = sub.add_parser("synthetic", parents=[common], help="offline sanity check + negative control")
    syn.add_argument("--n", type=int, default=800, help="synthetic population size")

    real = sub.add_parser("real", parents=[common], help="combined Buffalo + Rochester sales")
    real.add_argument("--limit", type=int, default=0, help="sample down to this many sales (0 = all)")
    real.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    real.add_argument(
        "--physical-only", action="store_true",
        help="skip the geographic feature store and the FRED macro join "
             "(reproduces the Ticket-5 physical-only baseline)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mode == "synthetic":
        population = generate_synthetic_population(args.n, args.seed)
        control_pop = generate_synthetic_population(args.n, args.seed, random_price=True)
        report = run_all(population, args.n_queries, args.k, args.seed)
        control = run_all(control_pop, args.n_queries, args.k, args.seed)
        print_report(report, "synthetic (signal)", control=control)
        payload = {"signal": report, "control": control}
    else:
        population = load_real_population(
            args.limit, args.cache, join_macro=not args.physical_only,
        )
        report = run_all(
            population, args.n_queries, args.k, args.seed,
            use_store_geo=not args.physical_only,
        )
        print_report(report, "real")
        payload = {"real": report}

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, default=float))
        print(f"raw report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
