"""Data source definitions for comp-finder.

NAMING CONVENTIONS:
- training_* : DataSources used for model training (e.g., training_customers, training_transactions)
- production_* : DataSources used for production inference/batch scoring

JOIN BEHAVIOR:
- The FIRST DataSource in each group is the primary source
- Subsequent DataSources are joined to the primary using their join_spec
- All DataSources in a group should share a common primary key

This module is imported by model.py and pipeline.py to load training/scoring data.
"""

import sys
from geronimo.data_sources import DataSource, JoinSpec, Query, collect_data_sources


# =============================================================================
# Training Data Sources
# =============================================================================

# Primary training data source
# training_customers = DataSource(
#     name="training_customers",
#     query=Query("SELECT * FROM customers WHERE split = 'train'"),
#     connection_type="snowflake",
# )

# Additional training data (joined to primary)
# training_transactions = DataSource(
#     name="training_transactions",
#     query=Query("SELECT * FROM transactions WHERE split = 'train'"),
#     connection_type="snowflake",
#     join_spec=JoinSpec(
#         left_on="customer_id",
#         right_on="customer_id",
#         how="left"
#     ),
# )


# =============================================================================
# Production Data Sources
# =============================================================================

# Production data sources for inference/scoring
# production_customers = DataSource(
#     name="production_customers",
#     query=Query("SELECT * FROM customers WHERE is_active = true"),
#     connection_type="snowflake",
# )


# =============================================================================
# Auto-collect sources
# =============================================================================

# Automatically collect training and production sources from module
training_sources = collect_data_sources(sys.modules[__name__], "training_")
production_sources = collect_data_sources(sys.modules[__name__], "production_") 

