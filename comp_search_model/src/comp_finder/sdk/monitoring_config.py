"""Realtime monitoring configuration - metrics thresholds and alerts.

Realtime endpoints focus on:
- Latency monitoring (p50, p99 thresholds)
- Error rate tracking
- Alerts for threshold breaches (Slack/email)

NOTE: For drift detection in realtime, consider using batch monitoring
to periodically analyze sampled request data. Full realtime drift is
planned for a future release.

TODO: Future enhancement - add realtime drift detection by:
- Sampling incoming requests to a buffer
- Periodically comparing buffer against reference data
- See: https://docs.evidentlyai.com/user-guide/installation/monitor_with_evidently
"""

import os
from comp_finder.monitoring.alerts import AlertManager, SlackAlert, AlertSeverity
from comp_finder.monitoring.metrics import MetricsCollector


# =============================================================================
# Latency Thresholds - customize these values
# =============================================================================

# Latency thresholds (milliseconds)
LATENCY_P50_WARNING = 100.0  # Alert if p50 latency exceeds this
LATENCY_P99_WARNING = 500.0  # Alert if p99 latency exceeds this

# Error rate thresholds (percentage)
ERROR_RATE_WARNING = 1.0     # Alert if error rate exceeds 1%
ERROR_RATE_CRITICAL = 5.0    # Critical alert if error rate exceeds 5%

# TODO: Future - drift thresholds for realtime (requires request sampling)
# DRIFT_THRESHOLD = 0.3      # Alert if drift score exceeds 30%


# =============================================================================
# Alert Configuration
# =============================================================================

def create_alert_manager() -> AlertManager:
    """Create and configure the alert manager.
    
    To enable Slack alerts:
    1. Create a Slack incoming webhook
    2. Set SLACK_WEBHOOK_URL environment variable
    
    Example:
        export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    """
    alerts = AlertManager(cooldown_seconds=300)  # 5 min between duplicate alerts
    
    # Add Slack if configured
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        alerts.add_slack(
            webhook_url=slack_webhook,
            channel=os.getenv("SLACK_CHANNEL"),  # Optional channel override
        )
    
    return alerts


# =============================================================================
# Drift Detection Configuration (Optional - requires Evidently)
# =============================================================================

def create_drift_detector(reference_data=None):
    """Create and configure the drift detector.
    
    Args:
        reference_data: Training data to compare production data against.
                       Load this from your data warehouse or artifact store.
    
    Example:
        import pandas as pd
        training_df = pd.read_parquet("data/training_sample.parquet")
        detector = create_drift_detector(reference_data=training_df)
    """
    from comp_finder.monitoring.drift import DriftDetector
    
    return DriftDetector(
        reference_data=reference_data,
        # TODO: Specify your feature types
        # categorical_features=["category", "region"],
        # numerical_features=["age", "income", "score"],
        # target_column="prediction",
    )


# =============================================================================
# Threshold Monitoring (call periodically or after each request)
# =============================================================================

def check_thresholds(metrics: MetricsCollector, alerts: AlertManager) -> None:
    """Check metrics against thresholds and send alerts if breached.
    
    Call this periodically (e.g., every minute) or after batch processing.
    
    Example:
        # In app.py or a background task
        from comp_finder.sdk.monitoring_config import (
            create_alert_manager, check_thresholds
        )
        from comp_finder.monitoring.metrics import metrics
        
        alerts = create_alert_manager()
        check_thresholds(metrics, alerts)
    """
    # Check latency
    p50 = metrics.get_latency_p50()
    if p50 > LATENCY_P50_WARNING:
        alerts.alert(
            title="High P50 Latency",
            message=f"P50 latency is {p50:.1f}ms (threshold: {LATENCY_P50_WARNING}ms)",
            severity=AlertSeverity.WARNING,
            metadata={"current": p50, "threshold": LATENCY_P50_WARNING},
        )
    
    p99 = metrics.get_latency_p99()
    if p99 > LATENCY_P99_WARNING:
        alerts.alert(
            title="High P99 Latency",
            message=f"P99 latency is {p99:.1f}ms (threshold: {LATENCY_P99_WARNING}ms)",
            severity=AlertSeverity.WARNING,
            metadata={"current": p99, "threshold": LATENCY_P99_WARNING},
        )
    
    # Check error rate
    error_count = metrics.get_error_count()
    request_count = metrics.get_request_count()
    if request_count > 0:
        error_rate = (error_count / request_count) * 100
        if error_rate > ERROR_RATE_CRITICAL:
            alerts.alert(
                title="Critical Error Rate",
                message=f"Error rate is {error_rate:.1f}% (threshold: {ERROR_RATE_CRITICAL}%)",
                severity=AlertSeverity.CRITICAL,
                metadata={"error_rate": error_rate, "errors": error_count, "requests": request_count},
            )
        elif error_rate > ERROR_RATE_WARNING:
            alerts.alert(
                title="Elevated Error Rate",
                message=f"Error rate is {error_rate:.1f}% (threshold: {ERROR_RATE_WARNING}%)",
                severity=AlertSeverity.WARNING,
                metadata={"error_rate": error_rate},
            )
