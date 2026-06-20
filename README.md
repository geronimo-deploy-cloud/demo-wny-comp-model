# Western NY Real Estate Comp Model
-----------------------------------
This model is trained on recent residential real estate sales data in the Western NY Region and predicts home sale price based on physical attributes, macro-economic data, and geographic spatiotemporal market data. 

Two producer/consumer Python projects communicate via the Geronimo `ArtifactStore`:

1. **`geographic_feature_store`** — weekly batch pipeline that precomputes spatiotemporal velocity grids
2. **`comp_model`** — realtime XGBoost model serving with a REST endpoint and an MCP server for AI agents

## Geographic Feature Store

The geographic feature store is a weekly Metaflow batch pipeline that converts raw sales transactions into precomputed H3 spatial grids, making them available for near-realtime model inference at inference time.

### How it works

Each weekly run ingests property sales data for the Buffalo and Rochester metro areas, then computes a BallTree over every sale's coordinates to calculate two metrics — `sales_volume` and `median_ppsf` (price per square foot) — for every H3 cell in the metro bounding boxes. 

### Spatial resolution

The grid uses **H3 resolution 8**, which produces hexagonal cells with an edge length of approximately 0.46 km (~0.29 mi). Each level-8 hex covers roughly 0.74 square km and has 7 parent hexagons at resolution 7. This resolution was chosen as a sweet spot between spatial detail and coverage: fine enough to capture neighborhood-level price variation, but coarse enough that nearby sales reliably fall into adjacent cells for robust velocity aggregation.

### Velocity grid configuration

The pipeline outputs **12 velocity artifacts**, one for each (radius, lookback window) combination:

| Radius | Lookback Windows |
| -------- | ---------------- |
| 1 mile   | 30, 90, 365 days |
| 5 miles  | 90, 180, 365 days |
| 10 miles | 180, 365 days    |

At inference time, `comp_model` loads the 12 artifacts from the `ArtifactStore` (project: `geographic-feature-store`, version: `1.0.0`), indexes them by `h3_index`, and performs O(1) lookups per property coordinate. This decouples the expensive BallTree computation (run weekly in batch) from the low-latency real-time prediction path.

## Comp Model

The comp model is a realtime model serving layer that loads trained XGBoost artifacts and precomputed velocity grids from the geographic feature store, then serves predictions via REST API and MCP (Model Context Protocol) for AI agent integration.

### Model overview

Trained on Rochester residential sales with a log-transformed target (`log1p(sale_price)`), the model uses three feature families:

- **Physical** — `beds`, `baths`, `year_built`, `total_living_area`, `lot_acres`, etc.
- **Macro-economic** — 30-year mortgage rates, federal funds rate, CPI, and unemployment rate loaded from FRED
- **Geographic velocity** — precomputed H3 grid lookups from the feature store (six radius/window combos, two metrics each)

### Endpoints

| Route | Method | Description |
| ------- | ---------- | ------------- |
| `POST /predict` | JSON body | Predict from raw property features dict |
| `POST /predict-url` | JSON body | Predict from a Zillow listing URL (scrapes → enriches → predicts) |
| `GET /health` | — | Service health check |
| `GET /metrics` | — | Latency percentiles, request counts, error rates |
| `POST /mcp` | — | MCP endpoint for AI agent tool calls |
| `GET /mcp` | — | MCP server HTTP transport |

Both `/predict` and `/predict-url` return the same structured response:

```json
{
  "prediction": 285000.50,
  "status": "ok"
}
```

The `/predict-url` endpoint is a convenience wrapper: it scrapes a property page (currently Zillow via `__NEXT_DATA__` JSON extraction), enriches missing fields via the City of Rochester ArcGIS assessor API, assigns school districts via shapefile spatial join, and then runs the full feature transform and prediction pipeline.

### MCP Server

The MCP server is mounted at `/mcp` when `geronimo.yaml` sets `model.mcp_enabled: true`. It exposes two tools:

| Tool | Purpose |
| -------- | ---------- |
| `predict(features)` | Make a model prediction from a features dict |
| `get_model_info()` | Return model name, version, and training status |

The server supports two transports:

- **HTTP** — mounted at `http://localhost:8000/mcp` for remote agent access
- **stdio** — run via `uv run python -m comp_model.agent` for local desktop agents (e.g. Claude Desktop)
