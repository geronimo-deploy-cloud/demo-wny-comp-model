"""Feature definitions for comp-finder.

DEVELOPMENT WORKFLOW:
1. Review training and production data sources for column consistency
2. Perform exploratory data analysis (EDA) to identify:
   - Missing values → consider imputation strategies
   - Outliers → consider clipping or winsorization  
   - Skewed distributions → consider log/power transforms
   - Categorical cardinality → consider encoding strategies
3. Define Features with appropriate transformers

Each Feature describes a column with its type, transformer, and encoder.
The FeatureSet handles fit_transform (training) and transform (inference).
"""

from typing import Optional
from geronimo.features import FeatureSet, Feature
# from sklearn.preprocessing import StandardScaler, OneHotEncoder


class CompFinderFeatures(FeatureSet):
    """Feature engineering for comp-finder.
    
    Define your features here. Each Feature describes a column transformation.
    
    Example:
        age = Feature(dtype='numeric', transformer=StandardScaler())
        income = Feature(dtype='numeric', transformer=StandardScaler())
        category = Feature(dtype='categorical', encoder=OneHotEncoder())
        
        # Derived feature from multiple columns
        age_income_ratio = Feature(
            dtype='numeric',
            derived_feature_fn=lambda df: df['age'] / df['income']
        )
    """
    
    # TODO: Define your features based on EDA results
    # feature_1 = Feature(dtype='numeric')
    # feature_2 = Feature(dtype='categorical')
    pass
