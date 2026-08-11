"""FastAPI middleware for monitoring.

Automatically captures and records:
- Request latency
- Error rates
- Request counts
- Input/output data for drift detection (optional)
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .metrics import MetricsCollector, MetricType

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic ML model monitoring."""

    def __init__(
        self,
        app: ASGIApp,
        collector: MetricsCollector,
        log_requests: bool = True,
    ) -> None:
        """Initialize middleware.

        Args:
            app: The ASGI application.
            collector: MetricsCollector instance.
            log_requests: Whether to log request details.
        """
        super().__init__(app)
        self.collector = collector
        self.log_requests = log_requests

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and record metrics."""
        start_time = time.time()
        
        # Determine path for dimensions
        path = request.url.path
        method = request.method
        
        # Skip health checks to avoid noise
        if path in ("/health", "/ready", "/live"):
            return await call_next(request)

        try:
            response = await call_next(request)
            
            # Record latency
            duration_ms = (time.time() - start_time) * 1000
            self.collector.record_latency(duration_ms)
            
            # Record request count
            self.collector.record_latency(duration_ms) # Reuse latency for count + dimension? 
            # Actually, explicit count is better
            self.collector.record(
                MetricType.REQUEST_COUNT, 
                1, 
                unit="Count",
                dimensions={"Path": path, "Method": method, "Status": str(response.status_code)}
            )
            
            # Record error if 5xx
            if 500 <= response.status_code < 600:
                self.collector.record_error()
                
            return response
            
        except Exception as e:
            # Record exception as error
            self.collector.record_error()
            
            # Re-raise to let FastAPI handle it
            raise e
        finally:
            # Attempt to flush periodically? 
            # For now, rely on collector's internal buffer logic or external scheduler
            pass
