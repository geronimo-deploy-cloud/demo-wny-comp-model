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
from comp_finder.sdk.endpoint import CompFinderEndpoint
from comp_finder.monitoring.middleware import MonitoringMiddleware
from comp_finder.monitoring.metrics import MetricsCollector


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

PROJECT_NAME = _config.project.name if _config else "comp-finder"

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
        _endpoint = CompFinderEndpoint()
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
        from comp_finder.agent import mcp
        app.mount("/mcp", mcp.http_app())
    except ImportError:
        pass  # MCP dependencies not installed


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    """Prediction request schema."""
    features: dict[str, Any]


class PredictResponse(BaseModel):
    """Prediction response schema."""
    prediction: Any


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
def predict(request: PredictRequest):
    """Generate prediction from model.
    
    The endpoint handles:
    1. preprocess() - transform request to features
    2. model.predict() - generate prediction
    3. postprocess() - format response
    
    Latency and errors are automatically tracked by MonitoringMiddleware.
    """
    try:
        endpoint = get_endpoint()
        result = endpoint.handle(request.model_dump())
        return PredictResponse(prediction=result)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
