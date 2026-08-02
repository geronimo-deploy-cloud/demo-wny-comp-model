"""Data drift detection for ML models.

Wraps Evidently AI to calculate drift between reference (training) 
data and current production data.
"""

import logging
from typing import Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from evidently import Report, DataDefinition, Dataset
    from evidently.presets import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not installed. Drift detection will be disabled.")


class DriftDetector:
    """Detector for data and prediction drift."""

    def __init__(
        self,
        reference_data: pd.DataFrame | None = None,
        categorical_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
        target_column: str | None = None,
    ) -> None:
        """Initialize drift detector.

        Args:
            reference_data: Training data to compare against.
            categorical_features: List of categorical column names.
            numerical_features: List of numerical column names.
            target_column: Name of the target/prediction column.
        """
        self.data_definition = DataDefinition()
        
        if categorical_features:
            self.data_definition.categorical_columns = categorical_features
        if numerical_features:
            self.data_definition.numerical_columns = numerical_features
        if target_column:
            self.data_definition.target_column = target_column

        self.reference_data = Dataset.from_pandas(reference_data, data_definition=self.data_definition)

    def calculate_drift(
        self,
        current_data: pd.DataFrame,
        report_path: str | None = None
    ) -> dict[str, Any]:
        """Calculate drift metrics.

        Args:
            current_data: Batch of recent production data.
            report_path: If provided, save HTML report to this path.

        Returns:
            Dictionary of drift metrics and status.
        """
        if not EVIDENTLY_AVAILABLE:
            return {"error": "Evidently not installed"}

        if self.reference_data is None:
            return {"error": "No reference data provided"}

        # apply column mapping to current data
        current_data = Dataset.from_pandas(current_data, data_definition=self.data_definition)

        # configure report
        report = Report(metrics=[
            DataDriftPreset()
        ])

        try:
            snapshot = report.run(
                reference_data=self.reference_data,
                current_data=current_data,
            )
            
            # Save HTML report
            if report_path:
                report.save_html(report_path)

            # Extract key metrics
            result = snapshot.dict()
            
            drift_share = result["metrics"][0]["value"]["share"]
            drift_count = result["metrics"][0]["value"]["count"]
            
            return {
                "drift_count": drift_count,
                "drift_share": drift_share,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": result["metrics"]
            }
            
        except Exception as e:
            logger.error(f"Drift calculation failed: {e}")
            return {"error": str(e)}
