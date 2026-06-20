"""Batch monitoring configuration - drift detection and alerts.

Batch jobs focus on:
- Drift detection (data drift between training and scoring data)
- Alerts for drift/failures (Slack/email notifications)

Unlike realtime endpoints, batch jobs don't need latency tracking.
"""

import os
import pandas as pd


# =============================================================================
# Drift Thresholds - customize these values
# =============================================================================

# Feature drift threshold (percentage of features drifted)
FEATURE_DRIFT_THRESHOLD = 0.3     # Alert if >30% of features show drift

# Dataset drift threshold (PSI/KS statistic)
DATASET_DRIFT_THRESHOLD = 0.1     # Alert if dataset drift score > 0.1

# Prediction drift threshold
PREDICTION_DRIFT_THRESHOLD = 0.2  # Alert if prediction distribution shifts


# =============================================================================
# Alert Configuration
# =============================================================================

def create_alert_manager():
    """Create alert manager for batch job notifications.
    
    To enable Slack alerts:
        export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    """
    from geographic_feature_store.monitoring.alerts import AlertManager
    
    alerts = AlertManager(cooldown_seconds=0)  # No cooldown for batch (runs periodically)
    
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        alerts.add_slack(
            webhook_url=slack_webhook,
            channel=os.getenv("SLACK_CHANNEL"),
        )
    
    return alerts


# =============================================================================
# Drift Detection - integrate into your pipeline.run() method
# =============================================================================

def create_drift_detector(reference_data: pd.DataFrame = None):
    """Create drift detector for batch scoring.
    
    Args:
        reference_data: Training data sample to compare against.
                       Load from your artifact store or data warehouse.
    
    Usage in pipeline.py:
        from .monitoring_config import create_drift_detector, check_drift
        
        def run(self):
            # Load reference data (training sample)
            reference = pd.read_parquet("data/training_sample.parquet")
            detector = create_drift_detector(reference_data=reference)
            
            # Load scoring data
            scoring_data = self.data_source.load()
            
            # Check for drift before scoring
            drift_result = check_drift(detector, scoring_data)
            if drift_result["has_drift"]:
                # Log warning or alert
                pass
            
            # Continue with scoring...
    """
    from geographic_feature_store.monitoring.drift import DriftDetector
    
    return DriftDetector(
        reference_data=reference_data,
        # TODO: Configure feature types for your model
        # categorical_features=["category", "region"],
        # numerical_features=["feature_1", "feature_2"],
        # target_column="prediction",
    )


def check_drift(detector, current_data: pd.DataFrame, alert_manager=None) -> dict:
    """Check for drift and optionally send alerts.
    
    Args:
        detector: DriftDetector instance with reference data
        current_data: Current batch data to check for drift
        alert_manager: Optional AlertManager for notifications
    
    Returns:
        Dict with drift results and alert status
    """
    from geographic_feature_store.monitoring.alerts import AlertSeverity
    
    result = detector.calculate_drift(current_data)
    
    has_drift = False
    if "drift_share" in result:
        has_drift = result["drift_share"] > FEATURE_DRIFT_THRESHOLD
        
        if has_drift and alert_manager:
            alert_manager.alert(
                title="Data Drift Detected",
                message=f"{result['drift_share']*100:.1f}% of features show drift (threshold: {FEATURE_DRIFT_THRESHOLD*100}%)",
                severity=AlertSeverity.WARNING,
                metadata={
                    "drift_share": result["drift_share"],
                    "threshold": FEATURE_DRIFT_THRESHOLD,
                },
            )
    
    return {
        "has_drift": has_drift,
        "drift_result": result,
    }


def send_pipeline_completion_alert(alert_manager, result: dict, success: bool = True):
    """Send alert when pipeline completes.
    
    Usage in pipeline.py:
        from .monitoring_config import create_alert_manager, send_pipeline_completion_alert
        
        def run(self):
            alerts = create_alert_manager()
            result = {"samples_scored": 1000}
            send_pipeline_completion_alert(alerts, result)
    """
    from geographic_feature_store.monitoring.alerts import AlertSeverity
    
    if success:
        alert_manager.alert(
            title="Batch Pipeline Complete",
            message=f"Successfully processed {result.get('samples_scored', 'N/A')} samples",
            severity=AlertSeverity.INFO,
            metadata=result,
        )
    else:
        alert_manager.alert(
            title="Batch Pipeline Failed",
            message=f"Pipeline failed: {result.get('error', 'Unknown error')}",
            severity=AlertSeverity.CRITICAL,
            metadata=result,
        )
