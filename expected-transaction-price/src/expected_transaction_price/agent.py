"""MCP Server for expected-transaction-price - AI Agent Integration.

This module provides an MCP (Model Context Protocol) server that exposes
your ML model as a tool for AI agents like Claude.

Transports:
- stdio: For local desktop agents (Claude Desktop)
- HTTP: Mounted at /mcp on your FastAPI app for remote agents

Usage (stdio):
    uv run python -m expected_transaction_price.agent
"""

import os
from typing import Any

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "FastMCP not installed. Install with: pip install fastmcp"
    )

from expected_transaction_price.sdk.endpoint import ExpectedTransactionPriceEndpoint


# Initialize MCP server
mcp = FastMCP("expected-transaction-price")

# Lazy-load endpoint
_endpoint = None


def get_endpoint() -> ExpectedTransactionPriceEndpoint:
    """Get or initialize the prediction endpoint."""
    global _endpoint
    if _endpoint is None:
        _endpoint = ExpectedTransactionPriceEndpoint()
        _endpoint.initialize()
    return _endpoint


@mcp.tool()
async def predict(features: dict[str, Any]) -> str:
    """Make a prediction using the expected-transaction-price ML model.
    
    Args:
        features: Dictionary of input features for the model.
        
    Returns:
        JSON string containing the prediction result.
    """
    endpoint = get_endpoint()
    try:
        result = endpoint.handle({"features": features})
        return str(result)
    except NotImplementedError:
        return "Model not trained. Run training first."
    except Exception as e:
        return f"Prediction error: {e}"


@mcp.tool()
async def predict_from_url(url: str) -> str:
    """Make a prediction from a property listing URL.
    
    Args:
        url: A live property listing URL (e.g., Zillow).
        
    Returns:
        JSON string containing the prediction result.
    """
    endpoint = get_endpoint()
    try:
        result = endpoint.predict_from_url(url)
        return str(result)
    except ValueError as e:
        return f"Prediction error: {e}"
    except Exception as e:
        return f"Prediction error: {e}"


@mcp.tool()
async def get_model_info() -> str:
    """Get information about the deployed model.
    
    Returns:
        Model name, version, and status.
    """
    endpoint = get_endpoint()
    if endpoint.model is None:
        return "Model not loaded (demo mode)"
    
    return f"Model: {endpoint.model.name} v{endpoint.model.version}"


def run_stdio():
    """Run the MCP server using stdio transport.
    
    This is used for local desktop agents like Claude Desktop.
    Configure in claude_desktop_config.json:
    
    {
        "mcpServers": {
            "expected-transaction-price": {
                "command": "uv",
                "args": ["run", "python", "-m", "expected_transaction_price.agent"],
                "cwd": "/path/to/expected-transaction-price"
            }
        }
    }
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_stdio()
