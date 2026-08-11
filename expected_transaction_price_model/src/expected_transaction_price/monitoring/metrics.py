"""Metrics collection and schema for model monitoring.

Defines the metrics schema and provides collectors for pushing
metrics to CloudWatch or other backends.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MetricType(str, Enum):
    """Types of metrics collected for ML models."""

    # Latency metrics
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    LATENCY_MEAN = "latency_mean"

    # Throughput metrics
    REQUEST_COUNT = "request_count"
    REQUESTS_PER_SECOND = "requests_per_second"

    # Error metrics
    ERROR_COUNT = "error_count"
    ERROR_RATE = "error_rate"

    # Model-specific metrics
    PREDICTION_COUNT = "prediction_count"
    PREDICTION_MEAN = "prediction_mean"
    PREDICTION_STD = "prediction_std"

    # Drift metrics
    DATA_DRIFT_SCORE = "data_drift_score"
    PREDICTION_DRIFT_SCORE = "prediction_drift_score"

    # Resource metrics
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"


@dataclass
class MetricValue:
    """A single metric measurement."""

    name: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    dimensions: dict[str, str] = field(default_factory=dict)
    unit: str = "None"


@dataclass
class MetricsBuffer:
    """Buffer for collecting metrics before publishing."""

    metrics: list[MetricValue] = field(default_factory=list)
    max_size: int = 100

    def add(self, metric: MetricValue) -> None:
        """Add a metric to the buffer."""
        self.metrics.append(metric)
        if len(self.metrics) > self.max_size:
            self.metrics = self.metrics[-self.max_size :]

    def flush(self) -> list[MetricValue]:
        """Get and clear all buffered metrics."""
        metrics = self.metrics.copy()
        self.metrics.clear()
        return metrics


class MetricsCollector:
    """Collects and publishes metrics for ML model monitoring.

    Supports multiple backends (CloudWatch, local file, etc.).
    """

    def __init__(
        self,
        project_name: str,
        environment: str = "dev",
        backend: str = "cloudwatch",
        buffer_size: int = 100,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            project_name: Name of the project (used as namespace).
            environment: Deployment environment.
            backend: Metrics backend ('cloudwatch', 'local', 'none').
            buffer_size: Max metrics to buffer before auto-flush.
        """
        self.project_name = project_name
        self.environment = environment
        self.backend = backend
        self._buffer = MetricsBuffer(max_size=buffer_size)
        self._default_dimensions = {
            "Project": project_name,
            "Environment": environment,
        }

    def record(
        self,
        metric_type: MetricType,
        value: float,
        unit: str = "None",
        dimensions: dict[str, str] | None = None,
    ) -> None:
        """Record a metric value.

        Args:
            metric_type: Type of metric.
            value: Metric value.
            unit: CloudWatch unit (Count, Seconds, Percent, etc.).
            dimensions: Additional dimensions.
        """
        all_dimensions = {**self._default_dimensions}
        if dimensions:
            all_dimensions.update(dimensions)

        metric = MetricValue(
            name=metric_type,
            value=value,
            unit=unit,
            dimensions=all_dimensions,
        )
        self._buffer.add(metric)

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement in milliseconds."""
        self.record(MetricType.LATENCY_MEAN, latency_ms, unit="Milliseconds")

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.record(MetricType.ERROR_COUNT, 1.0, unit="Count")

    def record_prediction(self, prediction: float | int) -> None:
        """Record a prediction value for distribution tracking."""
        self.record(MetricType.PREDICTION_COUNT, 1.0, unit="Count")
        self.record(MetricType.PREDICTION_MEAN, float(prediction), unit="None")

    def flush(self) -> int:
        """Flush buffered metrics to the backend.

        Returns:
            Number of metrics flushed.
        """
        metrics = self._buffer.flush()

        if not metrics:
            return 0

        if self.backend == "cloudwatch":
            self._publish_to_cloudwatch(metrics)
        elif self.backend == "local":
            self._publish_to_local(metrics)
        # 'none' backend discards metrics (useful for testing)

        return len(metrics)

    def _publish_to_cloudwatch(self, metrics: list[MetricValue]) -> None:
        """Publish metrics to CloudWatch.

        Note: Requires boto3 and AWS credentials.
        """
        try:
            import boto3

            client = boto3.client("cloudwatch")

            # Convert to CloudWatch format
            metric_data = []
            for m in metrics:
                metric_data.append(
                    {
                        "MetricName": m.name.value,
                        "Value": m.value,
                        "Unit": m.unit,
                        "Timestamp": m.timestamp,
                        "Dimensions": [
                            {"Name": k, "Value": v}
                            for k, v in m.dimensions.items()
                        ],
                    }
                )

            # CloudWatch accepts max 1000 metrics per call
            for i in range(0, len(metric_data), 1000):
                batch = metric_data[i : i + 1000]
                client.put_metric_data(
                    Namespace=f"Geronimo/{self.project_name}",
                    MetricData=batch,
                )

        except ImportError:
            # boto3 not installed, skip CloudWatch publishing
            pass
        except Exception as e:
            # Log but don't fail on metrics publishing errors
            import logging

            logging.warning(f"Failed to publish metrics to CloudWatch: {e}")

    def _publish_to_local(self, metrics: list[MetricValue]) -> None:
        """Log metrics locally for debugging."""
        import logging

        logger = logging.getLogger("geronimo.metrics")
        for m in metrics:
            logger.info(f"METRIC: {m.name.value}={m.value} {m.unit} {m.dimensions}")


@dataclass
class LatencyStats:
    """Aggregated latency statistics."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    values: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        """Record a latency value."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.values.append(latency_ms)

    @property
    def mean(self) -> float:
        """Calculate mean latency."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        """Calculate the p-th percentile."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        return self.percentile(50)

    @property
    def p95(self) -> float:
        """95th percentile."""
        return self.percentile(95)

    @property
    def p99(self) -> float:
        """99th percentile."""
        return self.percentile(99)
