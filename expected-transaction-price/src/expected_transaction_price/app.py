"""FastAPI application - thin wrapper around SDK endpoint.

This app integrates:
- SDK endpoint for predictions
- Monitoring middleware for latency/error tracking
- Metrics collector for CloudWatch/custom backends
- MCP server for AI agent integration (at /mcp)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from geronimo.config.loader import load_config
from expected_transaction_price.sdk.endpoint import ExpectedTransactionPriceEndpoint
from expected_transaction_price.monitoring.middleware import MonitoringMiddleware
from expected_transaction_price.monitoring.metrics import MetricsCollector


# =============================================================================
# Configuration - loaded from geronimo.yaml
# =============================================================================

def _find_config() -> Path:
    """Find geronimo.yaml in current or parent directories."""
    current = Path.cwd()
    for _ in range(5):  # Search up to 5 levels
        config_path = current / "geronimo.yaml"
        if config_path.exists():
            return config_path
        current = current.parent
    return Path("geronimo.yaml")

_config_path = _find_config()
_config = load_config(_config_path) if _config_path.exists() else None

PROJECT_NAME = _config.project.name if _config else "expected-transaction-price"

# Metrics backend: "cloudwatch", "local", or custom
METRICS_BACKEND = "local"  # TODO: Change to "cloudwatch" for production

# MCP agent integration - reads from geronimo.yaml model.mcp_enabled
ENABLE_MCP = _config.model.mcp_enabled if _config else True


# =============================================================================
# Initialize components
# =============================================================================

# Initialize metrics collector
# For CloudWatch: MetricsCollector(project_name=PROJECT_NAME, namespace="MLModels")
metrics = MetricsCollector(project_name=PROJECT_NAME)

# Lazy-load endpoint
_endpoint = None


def get_endpoint():
    global _endpoint
    if _endpoint is None:
        _endpoint = ExpectedTransactionPriceEndpoint()
        _endpoint.initialize()
    return _endpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle - load model on startup."""
    # Startup: pre-load model for faster first request
    get_endpoint()
    yield
    # Shutdown: cleanup if needed


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title=PROJECT_NAME,
    description="ML model serving API with monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - customize origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring middleware - tracks latency, errors, request counts
app.add_middleware(MonitoringMiddleware, collector=metrics)


# =============================================================================
# MCP Agent Integration (AI agents can call your model via /mcp)
# =============================================================================

if ENABLE_MCP:
    try:
        from expected_transaction_price.agent import mcp
        app.mount("/mcp", mcp.http_app())
    except ImportError:
        pass  # MCP dependencies not installed


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    """Prediction request schema (feature dict input)."""
    features: dict[str, Any]


class PredictUrlRequest(BaseModel):
    """Prediction request schema (Zillow URL input)."""
    url: str


class PredictResponse(BaseModel):
    """Unified prediction response schema."""
    prediction: float
    status: str = "ok"


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "mcp_enabled": ENABLE_MCP}


@app.get("/metrics")
def get_metrics():
    """Get current metrics summary.
    
    Returns latency percentiles, request counts, and error rates.
    """
    return {
        "latency_p50_ms": metrics.get_latency_p50(),
        "latency_p99_ms": metrics.get_latency_p99(),
        "request_count": metrics.get_request_count(),
        "error_count": metrics.get_error_count(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_features(request: PredictRequest):
    """Generate prediction from a feature dict.

    Expects: {"features": {"beds": 3, "baths": 2, ...}}
    Returns: {"prediction": 285000.50, "status": "ok"}
    """
    try:
        endpoint = get_endpoint()
        prediction = endpoint.predict_features(request.features)
        return PredictResponse(prediction=prediction)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-url", response_model=PredictResponse)
def predict_url(request: PredictUrlRequest):
    """Generate prediction from a Zillow property URL.

    Scrapes the listing, enriches features if needed, and runs prediction.
    Expects: {"url": "https://www.zillow.com/homedetails/..."}
    Returns: {"prediction": 285000.50, "status": "ok"}
    """
    try:
        endpoint = get_endpoint()
        prediction = endpoint.predict_from_url(request.url)
        return PredictResponse(prediction=prediction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
