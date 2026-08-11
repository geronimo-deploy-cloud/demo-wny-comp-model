"""Bulk historical model analysis for expected-transaction-price.

Loads a previously trained model, scores recent Rochester data, and produces:
  1. XGBoost native feature importances (gain, weight, cover)
  2. SHAP values — beeswarm + bar summary
  3. Price-band sensitivity analysis (MSE, MAE, MAPE by $50k quantile)
  4. Lift chart (predicted vs actual, sorted by actual)
  5. Automated recommendations for model improvement

Usage:
    python -m expected_transaction_price.bulk_historical_tests
    python -m expected_transaction_price.bulk_historical_tests --output-dir ./analysis_output
    python -m expected_transaction_price.bulk_historical_tests --max-samples 500
"""

import argparse
import json
import logging
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from expected_transaction_price.sdk.model import ExpectedTransactionPriceModel
from expected_transaction_price.sdk.data_sources import training_rochester
from geronimo.artifacts import ArtifactStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# =============================================================================
# 1. Data loading & scoring
# =============================================================================


def load_and_score(max_samples: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the trained model, fetch Rochester data, transform, and score.

    Returns:
        (raw_df, X_transformed, y_actual, y_predicted)
    """
    logger.info("Loading trained model from ArtifactStore...")
    store = ArtifactStore(project="expected-transaction-price", version="1.0.0")
    model = ExpectedTransactionPriceModel()
    model.load(store)
    logger.info("Model loaded: %s", model)

    logger.info("Fetching Rochester training data...")
    raw_df = training_rochester.load()
    raw_df = raw_df.dropna(subset=["sale_price"]).copy()

    # Load and merge macro features
    from expected_transaction_price.sdk.data_sources import training_macro
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    
    macro_df = training_macro.load()
    if not macro_df.empty:
        raw_df["sale_date"] = pd.to_datetime(raw_df["sale_date"], errors="coerce")
        raw_df = raw_df.merge(macro_df, on="sale_date", how="left")

    # Filter to recent data — last 2 years for relevance
    raw_df["sale_date"] = pd.to_datetime(raw_df["sale_date"], errors="coerce")
    two_years_ago = datetime.now() - pd.Timedelta(days=730)
    raw_df = raw_df[raw_df["sale_date"] >= two_years_ago]

    # Match training outlier filter ($100k–$600k)
    raw_df = raw_df[
        (raw_df["sale_price"] >= 100_000) & (raw_df["sale_price"] <= 600_000)
    ]

    if max_samples and len(raw_df) > max_samples:
        raw_df = raw_df.sample(n=max_samples, random_state=42)

    logger.info("Scoring %d Rochester properties...", len(raw_df))
    y_actual = raw_df["sale_price"].values

    # Use model.predict() which handles log-transform inverse automatically
    X_transformed = model.features.transform(raw_df)
    y_predicted = model.predict(raw_df)

    logger.info(
        "Scoring complete. Actual range: $%s–$%s, Predicted range: $%s–$%s",
        f"{y_actual.min():,.0f}", f"{y_actual.max():,.0f}",
        f"{y_predicted.min():,.0f}", f"{y_predicted.max():,.0f}",
    )

    return raw_df, X_transformed, y_actual, y_predicted


# =============================================================================
# 2. Feature Importance (XGBoost native)
# =============================================================================


def plot_feature_importance(model: ExpectedTransactionPriceModel, output_dir: Path) -> pd.DataFrame:
    """Generate XGBoost native feature importance charts (gain, weight, cover).

    Returns:
        DataFrame of feature importances sorted by gain.
    """
    logger.info("Computing XGBoost feature importances...")

    importance_types = {
        "gain": "Average Gain — how much each feature reduces loss when used",
        "weight": "Weight — how many times each feature appears in splits",
        "cover": "Cover — average number of samples affected by splits on this feature",
    }

    dfs = []
    for imp_type, desc in importance_types.items():
        scores = model.estimator.get_booster().get_score(importance_type=imp_type)
        df = pd.DataFrame(
            {"feature": list(scores.keys()), imp_type: list(scores.values())}
        )
        dfs.append(df)

    # Merge all importance types
    importance_df = dfs[0]
    for df in dfs[1:]:
        importance_df = importance_df.merge(df, on="feature", how="outer")
    importance_df = importance_df.fillna(0).sort_values("gain", ascending=False)

    # Plot top 25 by gain
    top_n = min(25, len(importance_df))
    top_df = importance_df.head(top_n).sort_values("gain", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_df["feature"], x=top_df["gain"],
        orientation="h", name="Gain",
        marker_color="#636EFA",
    ))
    fig.update_layout(
        title=f"Top {top_n} Features by Average Gain",
        xaxis_title="Average Gain (loss reduction per split)",
        yaxis_title="Feature",
        height=max(400, top_n * 28),
        margin=dict(l=200),
        template="plotly_white",
    )
    fig.write_html(str(output_dir / "feature_importance_gain.html"))

    # Multi-metric horizontal bar comparison (normalized)
    fig2 = go.Figure()
    for col, color in [("gain", "#636EFA"), ("weight", "#EF553B"), ("cover", "#00CC96")]:
        normed = top_df[col] / top_df[col].max() if top_df[col].max() > 0 else top_df[col]
        fig2.add_trace(go.Bar(
            y=top_df["feature"], x=normed,
            orientation="h", name=col.title(),
            marker_color=color,
        ))
    fig2.update_layout(
        title=f"Top {top_n} Features — Normalized Importance (Gain vs Weight vs Cover)",
        xaxis_title="Normalized Score",
        yaxis_title="Feature",
        barmode="group",
        height=max(500, top_n * 32),
        margin=dict(l=200),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig2.write_html(str(output_dir / "feature_importance_comparison.html"))

    importance_df.to_csv(str(output_dir / "feature_importance.csv"), index=False)
    logger.info("Feature importance saved to %s", output_dir)

    return importance_df


# =============================================================================
# 3. SHAP Values
# =============================================================================


def compute_shap_values(
    model: ExpectedTransactionPriceModel,
    X_transformed: pd.DataFrame,
    output_dir: Path,
    max_shap_samples: int = 500,
) -> None:
    """Compute SHAP values and generate beeswarm + bar charts.

    Uses TreeExplainer for XGBoost — O(TLD) per prediction, fast.
    """
    import shap

    logger.info("Computing SHAP values (TreeExplainer)...")

    # Subsample for SHAP if dataset is large
    if len(X_transformed) > max_shap_samples:
        X_shap = X_transformed.sample(n=max_shap_samples, random_state=42)
        logger.info("  Subsampled to %d rows for SHAP computation", max_shap_samples)
    else:
        X_shap = X_transformed

    explainer = shap.TreeExplainer(model.estimator)
    shap_values = explainer(X_shap)

    # --- SHAP Bar Summary (mean |SHAP|) ---
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    top_n = min(25, len(shap_df))
    top_shap = shap_df.head(top_n).sort_values("mean_abs_shap", ascending=True)

    fig_bar = go.Figure(go.Bar(
        y=top_shap["feature"], x=top_shap["mean_abs_shap"],
        orientation="h",
        marker_color="#AB63FA",
    ))
    fig_bar.update_layout(
        title=f"Top {top_n} Features by Mean |SHAP Value|",
        xaxis_title="Mean |SHAP Value| ($)",
        yaxis_title="Feature",
        height=max(400, top_n * 28),
        margin=dict(l=200),
        template="plotly_white",
    )
    fig_bar.write_html(str(output_dir / "shap_bar_summary.html"))

    # --- SHAP Beeswarm (scatter) ---
    # Plotly beeswarm: for each of the top features, scatter SHAP value vs
    # feature value (color = feature value magnitude).
    # Only include numeric features — categorical columns contain strings
    # that can't be color-scaled. Also skip columns that are entirely NaN.
    top_candidates = shap_df.head(min(25, len(shap_df)))["feature"].tolist()
    top_features = []
    for feat in top_candidates:
        if pd.api.types.is_numeric_dtype(X_shap[feat]):
            vals = pd.to_numeric(X_shap[feat], errors="coerce")
            if vals.notna().sum() > 0:
                top_features.append(feat)
        if len(top_features) >= 15:
            break

    beeswarm_data = []
    for feat in top_features:
        feat_idx = list(X_shap.columns).index(feat)
        feat_vals = pd.to_numeric(X_shap[feat], errors="coerce").values
        shap_vals = shap_values.values[:, feat_idx]

        # Normalize feature values for coloring (0-1)
        finite_mask = np.isfinite(feat_vals)
        if finite_mask.any():
            v_min = np.nanmin(feat_vals[finite_mask])
            v_max = np.nanmax(feat_vals[finite_mask])
        else:
            v_min, v_max = 0.0, 0.0

        if v_max > v_min:
            normed = (np.nan_to_num(feat_vals, nan=v_min) - v_min) / (v_max - v_min)
        else:
            normed = np.zeros(len(feat_vals))

        for i in range(len(shap_vals)):
            beeswarm_data.append({
                "feature": feat,
                "shap_value": float(shap_vals[i]),
                "feature_value": float(feat_vals[i]) if np.isfinite(feat_vals[i]) else 0.0,
                "normalized_value": float(normed[i]),
            })

    beeswarm_df = pd.DataFrame(beeswarm_data)

    # Build beeswarm with go.Scatter (px.strip doesn't support color_continuous_scale)
    fig_bee = go.Figure()
    for feat in reversed(top_features):  # reversed so top feature is at the top
        subset = beeswarm_df[beeswarm_df["feature"] == feat]
        fig_bee.add_trace(go.Scatter(
            x=subset["shap_value"],
            y=subset["feature"],
            mode="markers",
            marker=dict(
                size=4,
                color=subset["normalized_value"],
                colorscale="RdBu_r",
                cmin=0, cmax=1,
                showscale=(feat == top_features[0]),  # show colorbar once
                colorbar=dict(title="Feature<br>Value") if feat == top_features[0] else None,
                opacity=0.6,
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{feat}</b><br>"
                "SHAP: $%{x:,.0f}<br>"
                "Value: %{customdata[0]:.2f}<extra></extra>"
            ),
            customdata=subset[["feature_value"]].values,
        ))
    fig_bee.update_layout(
        title=f"SHAP Beeswarm — Top {len(top_features)} Numeric Features",
        xaxis_title="SHAP Value ($)",
        yaxis_title="Feature",
        height=max(500, len(top_features) * 45),
        margin=dict(l=200),
        template="plotly_white",
    )
    fig_bee.write_html(str(output_dir / "shap_beeswarm.html"))

    shap_df.to_csv(str(output_dir / "shap_summary.csv"), index=False)
    logger.info("SHAP analysis saved to %s", output_dir)


# =============================================================================
# 4. Price-Band Sensitivity Analysis
# =============================================================================


def price_band_analysis(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    output_dir: Path,
    band_size: int = 50_000,
) -> pd.DataFrame:
    """Compute MSE, MAE, and MAPE by $50k price bands.

    Returns:
        DataFrame with per-band metrics.
    """
    logger.info("Computing price-band sensitivity (band=$%dk)...", band_size // 1000)

    # Create bins
    max_price = int(np.ceil(y_actual.max() / band_size) * band_size)
    bins = list(range(0, max_price + band_size, band_size))
    labels = [f"${b // 1000}k–${(b + band_size) // 1000}k" for b in bins[:-1]]

    band_idx = np.digitize(y_actual, bins) - 1
    band_idx = np.clip(band_idx, 0, len(labels) - 1)

    records = []
    for i, label in enumerate(labels):
        mask = band_idx == i
        n = mask.sum()
        if n < 3:
            continue

        actual_band = y_actual[mask]
        pred_band = y_predicted[mask]
        errors = pred_band - actual_band
        abs_errors = np.abs(errors)
        pct_errors = abs_errors / np.maximum(actual_band, 1)

        records.append({
            "price_band": label,
            "n_samples": int(n),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(abs_errors)),
            "mape": float(np.mean(pct_errors) * 100),
            "median_error": float(np.median(errors)),
            "mean_actual": float(np.mean(actual_band)),
            "mean_predicted": float(np.mean(pred_band)),
            "bias": float(np.mean(errors)),
        })

    band_df = pd.DataFrame(records)

    # --- Bar chart: RMSE by price band ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["RMSE by Price Band", "MAPE (%) by Price Band"],
        shared_xaxes=True,
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(
            x=band_df["price_band"], y=band_df["rmse"],
            name="RMSE ($)",
            marker_color="#636EFA",
            text=[f"n={n}" for n in band_df["n_samples"]],
            textposition="outside",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=band_df["price_band"], y=band_df["mape"],
            name="MAPE (%)",
            marker_color="#EF553B",
            text=[f"{m:.1f}%" for m in band_df["mape"]],
            textposition="outside",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title="Model Sensitivity by Price Band ($50k quantiles)",
        height=700,
        template="plotly_white",
        showlegend=False,
    )
    fig.update_yaxes(title_text="RMSE ($)", row=1, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
    fig.update_xaxes(tickangle=45)
    fig.write_html(str(output_dir / "price_band_sensitivity.html"))

    # --- Bias chart: systematic over/under-prediction ---
    fig_bias = go.Figure()
    fig_bias.add_trace(go.Bar(
        x=band_df["price_band"], y=band_df["bias"],
        marker_color=np.where(band_df["bias"] > 0, "#00CC96", "#EF553B"),
        text=[f"${b:+,.0f}" for b in band_df["bias"]],
        textposition="outside",
    ))
    fig_bias.update_layout(
        title="Prediction Bias by Price Band (positive = over-predicting)",
        xaxis_title="Price Band",
        yaxis_title="Mean Bias ($)",
        height=450,
        template="plotly_white",
    )
    fig_bias.update_xaxes(tickangle=45)
    fig_bias.write_html(str(output_dir / "price_band_bias.html"))

    band_df.to_csv(str(output_dir / "price_band_metrics.csv"), index=False)
    logger.info("Price band analysis saved to %s", output_dir)

    return band_df


# =============================================================================
# 5. Lift Chart (predicted vs actual)
# =============================================================================


def lift_chart(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate a lift chart: predicted vs actual sorted by actual price."""
    logger.info("Generating lift chart...")

    # Sort by actual
    sort_idx = np.argsort(y_actual)
    actual_sorted = y_actual[sort_idx]
    pred_sorted = y_predicted[sort_idx]

    # Smooth with rolling average for readability
    window = max(1, len(actual_sorted) // 50)
    actual_smooth = pd.Series(actual_sorted).rolling(window, center=True).mean().values
    pred_smooth = pd.Series(pred_sorted).rolling(window, center=True).mean().values

    fig = go.Figure()

    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=list(range(len(actual_sorted))),
        y=actual_smooth,
        mode="lines",
        name="Actual (smoothed)",
        line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(pred_sorted))),
        y=pred_smooth,
        mode="lines",
        name="Predicted (smoothed)",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=f"Lift Chart — Predicted vs Actual (n={len(y_actual):,}, smoothed window={window})",
        xaxis_title="Properties (sorted by actual price)",
        yaxis_title="Sale Price ($)",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.write_html(str(output_dir / "lift_chart.html"))

    # Scatter: actual vs predicted (identity line)
    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(
        x=y_actual, y=y_predicted,
        mode="markers",
        marker=dict(size=3, color="#636EFA", opacity=0.4),
        name="Predictions",
    ))
    price_range = [min(y_actual.min(), y_predicted.min()), max(y_actual.max(), y_predicted.max())]
    fig2.add_trace(go.Scatter(
        x=price_range, y=price_range,
        mode="lines",
        line=dict(color="red", width=1, dash="dash"),
        name="Perfect prediction",
    ))
    fig2.update_layout(
        title="Actual vs Predicted — Scatter",
        xaxis_title="Actual Sale Price ($)",
        yaxis_title="Predicted Sale Price ($)",
        height=600,
        width=600,
        template="plotly_white",
    )
    fig2.write_html(str(output_dir / "actual_vs_predicted.html"))

    logger.info("Lift chart saved to %s", output_dir)


# =============================================================================
# 6. Residual Distribution
# =============================================================================


def residual_analysis(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    output_dir: Path,
) -> dict:
    """Analyze residual distribution and plot diagnostics.

    Returns:
        Summary statistics dict.
    """
    logger.info("Computing residual diagnostics...")
    residuals = y_predicted - y_actual
    pct_errors = (residuals / np.maximum(y_actual, 1)) * 100

    stats = {
        "n_samples": int(len(residuals)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "mape": float(np.mean(np.abs(pct_errors))),
        "median_error": float(np.median(residuals)),
        "mean_bias": float(np.mean(residuals)),
        "std_error": float(np.std(residuals)),
        "within_10pct": float(np.mean(np.abs(pct_errors) <= 10) * 100),
        "within_20pct": float(np.mean(np.abs(pct_errors) <= 20) * 100),
        "within_50pct": float(np.mean(np.abs(pct_errors) <= 50) * 100),
        "r2": float(1 - np.sum(residuals ** 2) / np.sum((y_actual - y_actual.mean()) ** 2)),
    }

    # Residual histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=80,
        marker_color="#636EFA",
        opacity=0.7,
        name="Residual ($)",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1)
    fig.add_vline(x=stats["mean_bias"], line_dash="dot", line_color="orange",
                  annotation_text=f"Mean bias: ${stats['mean_bias']:,.0f}")
    fig.update_layout(
        title=f"Residual Distribution (n={stats['n_samples']:,}, RMSE=${stats['rmse']:,.0f})",
        xaxis_title="Residual (Predicted − Actual, $)",
        yaxis_title="Count",
        height=400,
        template="plotly_white",
    )
    fig.write_html(str(output_dir / "residual_histogram.html"))

    # Percentage error histogram
    fig2 = go.Figure()
    clipped_pct = np.clip(pct_errors, -100, 100)
    fig2.add_trace(go.Histogram(
        x=clipped_pct,
        nbinsx=80,
        marker_color="#AB63FA",
        opacity=0.7,
    ))
    fig2.add_vline(x=0, line_dash="dash", line_color="red", line_width=1)
    fig2.update_layout(
        title=f"Percentage Error Distribution (within ±10%: {stats['within_10pct']:.1f}%)",
        xaxis_title="Error (%)",
        yaxis_title="Count",
        height=400,
        template="plotly_white",
    )
    fig2.write_html(str(output_dir / "pct_error_histogram.html"))

    logger.info("Residual analysis saved to %s", output_dir)
    return stats


# =============================================================================
# 7. Automated Recommendations
# =============================================================================


def generate_recommendations(
    importance_df: pd.DataFrame,
    band_df: pd.DataFrame,
    overall_stats: dict,
    output_dir: Path,
) -> str:
    """Analyze results and generate actionable recommendations."""
    logger.info("Generating recommendations...")

    rec = []
    rec.append("=" * 70)
    rec.append("MODEL IMPROVEMENT RECOMMENDATIONS")
    rec.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    rec.append("=" * 70)

    # --- Overall performance ---
    rec.append("\n## Overall Performance")
    rec.append(f"  R²:             {overall_stats['r2']:.4f}")
    rec.append(f"  RMSE:           ${overall_stats['rmse']:,.0f}")
    rec.append(f"  MAE:            ${overall_stats['mae']:,.0f}")
    rec.append(f"  MAPE:           {overall_stats['mape']:.1f}%")
    rec.append(f"  Mean Bias:      ${overall_stats['mean_bias']:+,.0f}")
    rec.append(f"  Within ±10%:    {overall_stats['within_10pct']:.1f}%")
    rec.append(f"  Within ±20%:    {overall_stats['within_20pct']:.1f}%")

    if overall_stats["r2"] < 0.7:
        rec.append("\n  ⚠️  R² < 0.70 — model is explaining less than 70% of price variance.")
        rec.append("     RECOMMENDATION: Consider adding more features (condition scores,")
        rec.append("     renovation year, proximity features) or switching to a more")
        rec.append("     expressive model (deeper XGBoost, LightGBM).")

    if abs(overall_stats["mean_bias"]) > 10_000:
        direction = "OVER" if overall_stats["mean_bias"] > 0 else "UNDER"
        rec.append(f"\n  ⚠️  Systematic {direction}-prediction bias of ${abs(overall_stats['mean_bias']):,.0f}.")
        rec.append("     RECOMMENDATION: Check if training data distribution differs from")
        rec.append("     the Rochester test set. Consider stratified sampling or a city-specific")
        rec.append("     intercept term.")

    # --- Feature importance insights ---
    rec.append("\n## Feature Importance Insights")
    if len(importance_df) > 0:
        top5 = importance_df.head(5)["feature"].tolist()
        rec.append(f"  Top 5 features by gain: {', '.join(top5)}")

        # Check for geographic velocity dominance
        velocity_feats = [f for f in importance_df["feature"].tolist() if "volume" in f or "ppsf" in f]
        non_velocity = [f for f in importance_df["feature"].tolist() if f not in velocity_feats]

        if velocity_feats:
            vel_gain = importance_df[importance_df["feature"].isin(velocity_feats)]["gain"].sum()
            total_gain = importance_df["gain"].sum()
            vel_pct = vel_gain / total_gain * 100 if total_gain > 0 else 0

            if vel_pct > 50:
                rec.append(f"\n  ⚠️  Geographic velocity features account for {vel_pct:.1f}% of total gain.")
                rec.append("     RECOMMENDATION: The model is leaning heavily on neighborhood signals.")
                rec.append("     This may cause poor generalization to areas with sparse sales data.")
                rec.append("     Consider regularizing velocity features or adding property-intrinsic")
                rec.append("     features to balance the model.")
            elif vel_pct < 5:
                rec.append(f"\n  ℹ️  Geographic velocity features contribute only {vel_pct:.1f}% of gain.")
                rec.append("     RECOMMENDATION: Velocity features may need recalibration.")
                rec.append("     Check if the lookback windows match Rochester's market tempo.")

        # Check for low-importance features (candidates for removal)
        if len(importance_df) > 10:
            bottom = importance_df.tail(5)
            if bottom["gain"].max() < importance_df["gain"].median() * 0.05:
                low_feats = bottom["feature"].tolist()
                rec.append(f"\n  ℹ️  Near-zero importance features: {', '.join(low_feats)}")
                rec.append("     RECOMMENDATION: Consider dropping these to reduce noise.")

    # --- Price band analysis ---
    rec.append("\n## Price Band Sensitivity")
    if not band_df.empty:
        worst_band = band_df.loc[band_df["mape"].idxmax()]
        best_band = band_df.loc[band_df["mape"].idxmin()]

        rec.append(f"  Best band:   {best_band['price_band']} — MAPE {best_band['mape']:.1f}%, n={int(best_band['n_samples'])}")
        rec.append(f"  Worst band:  {worst_band['price_band']} — MAPE {worst_band['mape']:.1f}%, n={int(worst_band['n_samples'])}")

        # High-end performance
        high_bands = band_df[band_df["mean_actual"] > 300_000]
        if not high_bands.empty:
            high_mape = high_bands["mape"].mean()
            if high_mape > 25:
                rec.append(f"\n  ⚠️  High-end properties (>$300k) have {high_mape:.1f}% MAPE.")
                rec.append("     RECOMMENDATION: The model struggles with luxury properties.")
                rec.append("     Consider: (1) separate model for >$300k, (2) log-transform target,")
                rec.append("     (3) add luxury-specific features (pool, waterfront, lot size class).")

        # Low-end performance
        low_bands = band_df[band_df["mean_actual"] < 100_000]
        if not low_bands.empty:
            low_mape = low_bands["mape"].mean()
            if low_mape > 30:
                rec.append(f"\n  ⚠️  Low-end properties (<$100k) have {low_mape:.1f}% MAPE.")
                rec.append("     RECOMMENDATION: Low-price properties are often distressed sales,")
                rec.append("     foreclosures, or investor flips. Consider: (1) adding a")
                rec.append("     'distressed_sale' flag, (2) filtering outlier prices during training,")
                rec.append("     (3) predicting log(price) to reduce heteroscedasticity.")

        # Bias direction
        biased_bands = band_df[band_df["bias"].abs() > 15_000]
        if not biased_bands.empty:
            rec.append("\n  ⚠️  Bands with >$15k systematic bias:")
            for _, row in biased_bands.iterrows():
                direction = "over" if row["bias"] > 0 else "under"
                rec.append(f"     {row['price_band']}: ${row['bias']:+,.0f} ({direction}-predicting)")

    # --- Actionable next steps ---
    rec.append("\n## Recommended Next Steps")
    rec.append("  1. Review SHAP beeswarm for non-monotone feature effects")
    rec.append("  2. Log-transform sale_price target to reduce heteroscedasticity")
    rec.append("  3. Add interaction features (total_living_area × school_district)")
    rec.append("  4. Hyperparameter tune: max_depth, min_child_weight, subsample")
    rec.append("  5. Train city-specific models if cross-city bias is significant")
    rec.append("  6. Evaluate geographic feature store velocity vs inline computation")

    report = "\n".join(rec)
    report_path = output_dir / "recommendations.txt"
    report_path.write_text(report)
    print(report)

    return report


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Bulk historical analysis of expected-transaction-price on Rochester data."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./analysis_output",
        help="Directory for HTML charts and CSV exports (default: ./analysis_output)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max number of properties to score (default: all available)",
    )
    parser.add_argument(
        "--max-shap-samples", type=int, default=500,
        help="Max samples for SHAP computation (default: 500)",
    )
    parser.add_argument(
        "--band-size", type=int, default=50_000,
        help="Price band width in dollars (default: 50000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMP MODEL — BULK HISTORICAL ANALYSIS (Rochester)")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 70)

    # 1. Load and score
    raw_df, X_transformed, y_actual, y_predicted = load_and_score(
        max_samples=args.max_samples,
    )

    # 2. Load model for introspection
    store = ArtifactStore(project="expected-transaction-price", version="1.0.0")
    model = ExpectedTransactionPriceModel()
    model.load(store)

    # 3. Feature importance
    importance_df = plot_feature_importance(model, output_dir)

    # 4. SHAP values
    compute_shap_values(model, X_transformed, output_dir, max_shap_samples=args.max_shap_samples)

    # 5. Price band sensitivity
    band_df = price_band_analysis(y_actual, y_predicted, output_dir, band_size=args.band_size)

    # 6. Lift chart
    lift_chart(y_actual, y_predicted, output_dir)

    # 7. Residual analysis
    overall_stats = residual_analysis(y_actual, y_predicted, output_dir)

    # 8. Recommendations
    generate_recommendations(importance_df, band_df, overall_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE — Output files:")
    print("=" * 70)
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:<40} {size:>10,} bytes")

    # Save run metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(y_actual)),
        "output_dir": str(output_dir.resolve()),
        **overall_stats,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nOpen the HTML files in a browser to explore interactive charts.")
    print(f"Key file: {output_dir / 'recommendations.txt'}")


if __name__ == "__main__":
    main()
