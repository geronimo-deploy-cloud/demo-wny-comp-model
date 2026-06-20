"""Exploratory Data Analysis (EDA) utilities for comp_model feature engineering."""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values in the dataset."""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_pct
    }).sort_values('missing_percent', ascending=False)
    return missing_df[missing_df['missing_count'] > 0]

def analyze_numerical_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze distribution properties of numerical columns."""
    numerics = df.select_dtypes(include=[np.number])
    return pd.DataFrame({
        'count': numerics.count(),
        'mean': numerics.mean(),
        'std': numerics.std(),
        'min': numerics.min(),
        'p25': numerics.quantile(0.25),
        'median': numerics.median(),
        'p75': numerics.quantile(0.75),
        'max': numerics.max(),
        'skew': numerics.skew(),
        'kurtosis': numerics.kurtosis()
    })

def analyze_categorical_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze cardinality and mode of categorical columns."""
    categoricals = df.select_dtypes(exclude=[np.number, 'datetime64[ns]'])
    stats = []
    for col in categoricals.columns:
        if df[col].dropna().empty:
            continue
        stats.append({
            'column': col,
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
            'top_freq': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0,
            'top_freq_pct': 100 * df[col].value_counts().iloc[0] / len(df) if not df[col].value_counts().empty else 0
        })
    return pd.DataFrame(stats).set_index('column') if stats else pd.DataFrame()

def analyze_correlations_with_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Calculate correlation of numerical features with the target variable."""
    numerics = df.select_dtypes(include=[np.number])
    if target_col not in numerics.columns:
        raise ValueError(f"Target column '{target_col}' must be numeric.")
    corr = numerics.corr()[target_col].sort_values(ascending=False)
    return corr.drop(target_col)

def run_full_eda(df: pd.DataFrame, target_col: str = "sale_price") -> Dict[str, Any]:
    """Run all EDA functions and return a comprehensive analysis dictionary."""
    logger.info(f"Running EDA on dataframe with shape {df.shape}")
    results = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing': analyze_missing_values(df),
        'numerical': analyze_numerical_distributions(df),
        'categorical': analyze_categorical_cardinality(df),
    }
    if target_col in df.columns:
        results['target_correlations'] = analyze_correlations_with_target(df, target_col)
    return results
