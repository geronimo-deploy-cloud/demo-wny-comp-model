"""Data sources for geographic feature store.

Standalone copies of the Buffalo (Socrata) and Rochester (ArcGIS) data loaders.
Only fetches the columns needed for geographic velocity computation:
  latitude, longitude, sale_date, sale_price, total_living_area, parcel_id, source

These loaders are intentionally decoupled from comp_model so the feature store
can evolve independently without a cross-project runtime dependency.
"""

import logging
import math
import sys
from typing import Optional

import numpy as np
import pandas as pd
import requests

from geronimo.data_sources import DataSource, collect_data_sources

logger = logging.getLogger(__name__)


# =============================================================================
# Minimal schema for velocity computation
# =============================================================================

VELOCITY_COLUMNS = [
    "source",
    "parcel_id",
    "latitude",
    "longitude",
    "sale_date",
    "sale_price",
    "total_living_area",
]


# =============================================================================
# Buffalo (Socrata SODA API)
# =============================================================================

BUFFALO_ASSESSMENT_URL = "https://data.buffalony.gov/resource/4t8s-9yih.json"
BUFFALO_SALES_URL = "https://data.buffalony.gov/resource/tndx-c4xz.json"
BUFFALO_SOCRATA_LIMIT = 50_000


def _fetch_socrata(
    url: str, where: Optional[str] = None, limit: int = BUFFALO_SOCRATA_LIMIT
) -> pd.DataFrame:
    """Fetch all records from a Socrata SODA API endpoint with pagination.

    Args:
        url: Socrata resource URL.
        where: Optional SoQL WHERE clause.
        limit: Page size for pagination.

    Returns:
        DataFrame with all records from the endpoint.
    """
    all_records = []
    offset = 0

    while True:
        params = {
            "$limit": limit,
            "$offset": offset,
            "$order": ":id",
        }
        if where:
            params["$where"] = where

        logger.info("Fetching %s offset=%d limit=%d", url, offset, limit)
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()

        batch = resp.json()
        if not batch:
            break

        all_records.extend(batch)
        offset += len(batch)

        if len(batch) < limit:
            break

    logger.info("Fetched %d total records from %s", len(all_records), url)
    return pd.DataFrame(all_records)


def _load_buffalo_sales() -> pd.DataFrame:
    """Fetch and normalize Buffalo sales data for velocity computation.

    Returns:
        DataFrame with VELOCITY_COLUMNS.
    """
    # Fetch assessment roll (has lat/lng, living area)
    assessment_df = _fetch_socrata(
        BUFFALO_ASSESSMENT_URL,
        where="property_class_code IN ('210', '220', '230', '240', '250')",
    )
    if assessment_df.empty:
        logger.warning("Buffalo assessment data returned empty")
        return pd.DataFrame(columns=VELOCITY_COLUMNS)

    # Fetch sales data
    sales_df = _fetch_socrata(BUFFALO_SALES_URL, where="sale_price > '0'")
    if sales_df.empty:
        logger.warning("Buffalo sales data returned empty")
        return pd.DataFrame(columns=VELOCITY_COLUMNS)

    # Join on print_key
    merged = sales_df.merge(
        assessment_df,
        on="print_key",
        how="inner",
        suffixes=("_sale", "_assess"),
    )

    logger.info(
        "Buffalo: %d assessments × %d sales → %d joined records",
        len(assessment_df),
        len(sales_df),
        len(merged),
    )

    _safe_float = lambda s: pd.to_numeric(s, errors="coerce")

    df = pd.DataFrame()
    df["source"] = "buffalo"
    df["parcel_id"] = merged.get("sbl", merged.get("print_key"))
    df["latitude"] = _safe_float(merged.get("latitude", pd.Series(dtype="float")))
    df["longitude"] = _safe_float(merged.get("longitude", pd.Series(dtype="float")))
    df["sale_date"] = pd.to_datetime(merged["sale_date"], errors="coerce", format="ISO8601")
    df["sale_price"] = _safe_float(merged["sale_price_sale"])
    df["total_living_area"] = _safe_float(
        merged.get("total_living_area", pd.Series(dtype="float"))
    )

    # Filter invalid records
    df = df.dropna(subset=["sale_price", "total_living_area", "latitude", "longitude"])
    df = df[(df["sale_price"] > 0) & (df["total_living_area"] > 0)]

    logger.info("Buffalo: %d clean velocity records after filtering", len(df))
    return df[VELOCITY_COLUMNS]


# =============================================================================
# Rochester (ArcGIS REST API)
# =============================================================================

ROCHESTER_BASE_URL = (
    "https://maps.cityofrochester.gov/arcgis/rest/services/"
    "App_PropertyInformation/ROC_Parcel_Query_RPS_Merged/MapServer"
)
ROCHESTER_PARCEL_LAYER = 0
ROCHESTER_RESIDENTIAL_TABLE = 2
ROCHESTER_MAX_RECORDS = 200_000


def _fetch_arcgis(
    layer_id: int,
    where: str = "1=1",
    out_fields: str = "*",
    return_geometry: bool = False,
) -> pd.DataFrame:
    """Fetch all records from a Rochester ArcGIS REST API layer/table.

    Args:
        layer_id: Layer or table ID within the MapServer.
        where: SQL WHERE clause for filtering.
        out_fields: Comma-separated field names or "*" for all.
        return_geometry: Whether to include geometry in response.

    Returns:
        DataFrame with all matching records.
    """
    query_url = f"{ROCHESTER_BASE_URL}/{layer_id}/query"

    # Get total record count
    count_params = {"where": where, "returnCountOnly": "true", "f": "json"}
    resp = requests.get(query_url, params=count_params, timeout=60)
    resp.raise_for_status()
    total_count = resp.json().get("count", 0)
    logger.info(
        "ArcGIS layer %d: %d total records matching '%s'",
        layer_id,
        total_count,
        where[:60],
    )

    if total_count == 0:
        return pd.DataFrame()

    # Paginate through results
    all_records = []
    page_size = min(ROCHESTER_MAX_RECORDS, total_count)
    num_pages = math.ceil(total_count / page_size)

    for page in range(num_pages):
        params = {
            "where": where,
            "outFields": out_fields,
            "f": "json",
            "returnGeometry": "true" if return_geometry else "false",
            "resultRecordCount": page_size,
            "resultOffset": page * page_size,
        }

        logger.info("Fetching ArcGIS layer %d page %d/%d", layer_id, page + 1, num_pages)
        resp = requests.get(query_url, params=params, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        features = data.get("features", [])
        if not features:
            break

        for f in features:
            attrs = f["attributes"]
            if return_geometry:
                rings = f.get("geometry", {}).get("rings", [])
                if rings and len(rings[0]) > 0:
                    ring = rings[0]
                    x_coords = [pt[0] for pt in ring]
                    y_coords = [pt[1] for pt in ring]
                    attrs["_longitude"] = sum(x_coords) / len(x_coords)
                    attrs["_latitude"] = sum(y_coords) / len(y_coords)
            all_records.append(attrs)

    logger.info("Fetched %d total records from ArcGIS layer %d", len(all_records), layer_id)
    return pd.DataFrame(all_records)


def _load_rochester_sales() -> pd.DataFrame:
    """Fetch and normalize Rochester sales data for velocity computation.

    Returns:
        DataFrame with VELOCITY_COLUMNS.
    """
    # Fetch parcel data (sale prices + coordinates)
    parcel_df = _fetch_arcgis(
        ROCHESTER_PARCEL_LAYER,
        where="SALE_PRICE > 0 AND CLASSCD IN ('210','220','230','240','250')",
        out_fields="PARCELID,SALE_PRICE,SALE_DATE",
        return_geometry=True,
    )
    if parcel_df.empty:
        logger.warning("Rochester parcel data returned empty")
        return pd.DataFrame(columns=VELOCITY_COLUMNS)

    # Fetch residential structure data (has living area)
    structure_df = _fetch_arcgis(
        ROCHESTER_RESIDENTIAL_TABLE,
        where="1=1",
        out_fields="SBL,RESFLRAREA",
    )
    if structure_df.empty:
        logger.warning("Rochester structure data returned empty")
        return pd.DataFrame(columns=VELOCITY_COLUMNS)

    # Join
    merged = parcel_df.merge(
        structure_df,
        left_on="PARCELID",
        right_on="SBL",
        how="inner",
    )

    logger.info(
        "Rochester: %d parcels × %d structures → %d joined records",
        len(parcel_df),
        len(structure_df),
        len(merged),
    )

    _safe_float = lambda s: pd.to_numeric(s, errors="coerce")

    df = pd.DataFrame()
    df["source"] = "rochester"
    df["parcel_id"] = merged["PARCELID"]
    
    # Rochester ArcGIS returns polygon centroids in Web Mercator (EPSG:3857).
    # Convert to WGS84 (EPSG:4326) for H3 velocity calculations.
    raw_lat = _safe_float(merged.get("_latitude", pd.Series(dtype="float")))
    raw_lng = _safe_float(merged.get("_longitude", pd.Series(dtype="float")))

    valid_coords = raw_lat.notna() & raw_lng.notna()
    if valid_coords.any():
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            lngs_4326, lats_4326 = transformer.transform(
                raw_lng[valid_coords].values, raw_lat[valid_coords].values
            )
            raw_lat = raw_lat.copy()
            raw_lng = raw_lng.copy()
            raw_lat.loc[valid_coords] = lats_4326
            raw_lng.loc[valid_coords] = lngs_4326
            logger.info("Converted %d Rochester velocity coordinates from EPSG:3857 → EPSG:4326", valid_coords.sum())
        except ImportError:
            logger.warning("pyproj not installed. Rochester coordinates may be in wrong CRS.")

    df["latitude"] = raw_lat
    df["longitude"] = raw_lng
    df["sale_date"] = pd.to_datetime(merged["SALE_DATE"], errors="coerce", format="mixed")
    df["sale_price"] = _safe_float(merged["SALE_PRICE"])
    df["total_living_area"] = _safe_float(merged["RESFLRAREA"])

    # Filter invalid records
    df = df.dropna(subset=["sale_price", "total_living_area", "latitude", "longitude"])
    df = df[(df["sale_price"] > 0) & (df["total_living_area"] > 0)]

    logger.info("Rochester: %d clean velocity records after filtering", len(df))
    return df[VELOCITY_COLUMNS]


# =============================================================================
# Combined Loader
# =============================================================================


def _load_all_sales() -> pd.DataFrame:
    """Load and combine sales data from all WNY sources.

    Returns:
        Combined DataFrame with VELOCITY_COLUMNS and deduplicated records.
    """
    logger.info("Loading combined WNY sales data for velocity computation...")

    dfs = []

    try:
        buffalo_df = _load_buffalo_sales()
        dfs.append(buffalo_df)
        logger.info("Buffalo: loaded %d records", len(buffalo_df))
    except Exception as e:
        logger.error("Failed to load Buffalo data: %s", e)

    try:
        rochester_df = _load_rochester_sales()
        dfs.append(rochester_df)
        logger.info("Rochester: loaded %d records", len(rochester_df))
    except Exception as e:
        logger.error("Failed to load Rochester data: %s", e)

    if not dfs:
        logger.error("No data loaded from any source")
        return pd.DataFrame(columns=VELOCITY_COLUMNS)

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate on (parcel_id, sale_date, sale_price) to handle overlapping fetches
    before = len(combined)
    combined = combined.drop_duplicates(subset=["parcel_id", "sale_date", "sale_price"])
    if before != len(combined):
        logger.info("Deduplication removed %d records", before - len(combined))

    logger.info(
        "Combined sales data: %d records (%d Buffalo, %d Rochester)",
        len(combined),
        len(combined[combined["source"] == "buffalo"]),
        len(combined[combined["source"] == "rochester"]),
    )

    return combined


# =============================================================================
# Geronimo DataSource declarations
# =============================================================================

training_sales = DataSource(
    name="training_sales",
    source="function",
    handle=_load_all_sales,
)

production_sales = DataSource(
    name="production_sales",
    source="function",
    handle=_load_all_sales,
)

# Auto-collect
training_sources = collect_data_sources(sys.modules[__name__], "training_")
production_sources = collect_data_sources(sys.modules[__name__], "production_")
