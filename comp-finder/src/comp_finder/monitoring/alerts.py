"""Alert management for model monitoring.

Provides alerting capabilities via Slack webhooks and
future ADO Work Item integration.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert to be sent to one or more destinations."""

    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.WARNING
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "geronimo"


class AlertDestination(ABC):
    """Abstract base for alert destinations."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert to this destination.

        Returns:
            True if alert was sent successfully.
        """
        pass


class SlackAlert(AlertDestination):
    """Send alerts to Slack via webhook."""

    # Severity to Slack color mapping
    SEVERITY_COLORS = {
        AlertSeverity.INFO: "#36a64f",  # Green
        AlertSeverity.WARNING: "#ffcc00",  # Yellow
        AlertSeverity.ERROR: "#ff6600",  # Orange
        AlertSeverity.CRITICAL: "#ff0000",  # Red
    }

    # Severity to emoji mapping
    SEVERITY_EMOJI = {
        AlertSeverity.INFO: ":information_source:",
        AlertSeverity.WARNING: ":warning:",
        AlertSeverity.ERROR: ":x:",
        AlertSeverity.CRITICAL: ":rotating_light:",
    }

    def __init__(self, webhook_url: str, channel: str | None = None) -> None:
        """Initialize Slack alerter.

        Args:
            webhook_url: Slack incoming webhook URL.
            channel: Override channel (optional).
        """
        self.webhook_url = webhook_url
        self.channel = channel

    def send(self, alert: Alert) -> bool:
        """Send an alert to Slack."""
        payload = self._build_payload(alert)

        try:
            data = json.dumps(payload).encode("utf-8")
            request = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urlopen(request, timeout=10)
            return True
        except URLError as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Slack alert: {e}")
            return False

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """Build Slack message payload."""
        emoji = self.SEVERITY_EMOJI.get(alert.severity, ":bell:")
        color = self.SEVERITY_COLORS.get(alert.severity, "#808080")

        payload: dict[str, Any] = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} {alert.title}",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": alert.message,
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Source:* {alert.source} | *Time:* {alert.timestamp.isoformat()}",
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        # Add metadata fields if present
        if alert.metadata:
            fields = []
            for key, value in alert.metadata.items():
                fields.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*{key}:* {value}",
                    }
                )
            payload["attachments"][0]["blocks"].insert(
                2,
                {
                    "type": "section",
                    "fields": fields[:10],  # Slack limits to 10 fields
                },
            )

        if self.channel:
            payload["channel"] = self.channel

        return payload


class AlertManager:
    """Manages alert routing to multiple destinations.

    Supports thresholds and rate limiting to avoid alert fatigue.
    """

    def __init__(
        self,
        cooldown_seconds: int = 300,
    ) -> None:
        """Initialize the alert manager.

        Args:
            cooldown_seconds: Minimum time between duplicate alerts.
        """
        self._destinations: list[AlertDestination] = []
        self._cooldown_seconds = cooldown_seconds
        self._last_alerts: dict[str, datetime] = {}

    def add_destination(self, destination: AlertDestination) -> None:
        """Add an alert destination."""
        self._destinations.append(destination)

    def add_slack(self, webhook_url: str, channel: str | None = None) -> None:
        """Convenience method to add Slack destination."""
        self.add_destination(SlackAlert(webhook_url, channel))

    def alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        metadata: dict[str, Any] | None = None,
        source: str = "geronimo",
        bypass_cooldown: bool = False,
    ) -> int:
        """Send an alert to all destinations.

        Args:
            title: Alert title.
            message: Alert message body.
            severity: Alert severity level.
            metadata: Additional context.
            source: Alert source identifier.
            bypass_cooldown: If True, ignore rate limiting.

        Returns:
            Number of destinations that received the alert.
        """
        # Check cooldown
        alert_key = f"{title}:{severity.value}"
        now = datetime.utcnow()

        if not bypass_cooldown and alert_key in self._last_alerts:
            last_time = self._last_alerts[alert_key]
            elapsed = (now - last_time).total_seconds()
            if elapsed < self._cooldown_seconds:
                logger.debug(f"Alert '{title}' skipped due to cooldown")
                return 0

        # Create alert
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            timestamp=now,
            metadata=metadata or {},
            source=source,
        )

        # Send to all destinations
        sent_count = 0
        for dest in self._destinations:
            if dest.send(alert):
                sent_count += 1

        # Update cooldown tracking
        self._last_alerts[alert_key] = now

        return sent_count

    def alert_threshold(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        comparison: str = "gt",
        **alert_kwargs: Any,
    ) -> bool:
        """Send an alert if a threshold is breached.

        Args:
            metric_name: Name of the metric.
            current_value: Current metric value.
            threshold: Threshold value.
            comparison: 'gt' (greater than) or 'lt' (less than).
            **alert_kwargs: Additional arguments for alert().

        Returns:
            True if alert was sent.
        """
        is_breached = (
            current_value > threshold
            if comparison == "gt"
            else current_value < threshold
        )

        if is_breached:
            direction = "above" if comparison == "gt" else "below"
            title = alert_kwargs.pop("title", f"{metric_name} threshold breached")
            message = alert_kwargs.pop(
                "message",
                f"{metric_name} is {direction} threshold.\n"
                f"Current: {current_value:.2f}, Threshold: {threshold:.2f}",
            )
            self.alert(title=title, message=message, **alert_kwargs)
            return True

        return False
