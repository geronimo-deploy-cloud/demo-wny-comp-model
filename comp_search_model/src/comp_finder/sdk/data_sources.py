"""Data source definitions for comp-finder.

Provides historical sales data loaders for the comp-vector index pipeline.
The loaders mirror those in `expected_transaction_price.sdk.data_sources`
and fetch from the same public APIs (Buffalo Socrata + Rochester ArcGIS).

This module is imported by pipeline.py and train.py to load the full
historical sales population for index building.
"""

import logging
import math
import sys
from typing import Optional

import pandas as pd
import requests

from geronimo.data_sources import DataSource, collect_data_sources

logger = logging.getLogger(__name__)

# =============================================================================
# Unified Schema (copied from expected_transaction_price)
# =============================================================================
# Both Buffalo and Rochester data are normalized to these column names.
# This is the contract between data ingestion and the feature/model layers.

UNIFIED_COLUMNS = [
    "source",
    "parcel_id",
    "print_key",
    "address",
    "city",
    "zip_code",
    "property_class",
    "property_class_desc",
    "sale_price",
    "sale_date",
    "assessed_value",
    "land_value",
    "year_built",
    "total_living_area",
    "first_floor_area",
    "second_floor_area",
    "beds",
    "baths",
    "half_baths",
    "kitchens",
    "stories",
    "fireplaces",
    "lot_frontage",
    "lot_depth",
    "lot_acres",
    "building_style",
    "overall_condition",
    "construction_grade",
    "exterior_wall",
    "heat_type",
    "central_air",
    "fuel_type",
    "basement_type",
    "latitude",
    "longitude",
    "school_district",
]

# =============================================================================
# Buffalo (Socrata SODA API)
# =============================================================================

BUFFALO_ASSESSMENT_URL = "https://data.buffalony.gov/resource/4t8s-9yih.json"
BUFFALO_SALES_URL = "https://data.buffalony.gov/resource/tndx-c4xz.json"
BUFFALO_SOCRATA_LIMIT = 50_000  # Socrata max per page


def _fetch_socrata(url: str, where: Optional[str] = None, limit: int = BUFFALO_SOCRATA_LIMIT) -> pd.DataFrame:
    """Fetch all records from a Socrata SODA API endpoint with pagination."""
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


def _load_buffalo_training_data() -> pd.DataFrame:
    """Load and normalize Buffalo property data."""
    assessment_df = _fetch_socrata(
        BUFFALO_ASSESSMENT_URL,
        where="property_class_code IN ('210', '220', '230', '240', '250')",
    )

    if assessment_df.empty:
        logger.warning("Buffalo assessment data returned empty")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    sales_df = _fetch_socrata(
        BUFFALO_SALES_URL,
        where="sale_price > '0'",
    )

    if sales_df.empty:
        logger.warning("Buffalo sales data returned empty")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    merged = sales_df.merge(
        assessment_df,
        on="print_key",
        how="inner",
        suffixes=("_sale", "_assess"),
    )

    def _safe_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _parse_buffalo_date(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", format="ISO8601")

    df = pd.DataFrame()
    df["source"] = "buffalo"
    df["parcel_id"] = merged.get("sbl", merged.get("print_key"))
    df["print_key"] = merged["print_key"]
    df["address"] = merged.get("address", merged.get("st_no", "").astype(str) + " " + merged.get("st_name", "").astype(str))
    df["city"] = "Buffalo"
    df["zip_code"] = merged.get("zip_code_5_digit", pd.Series(dtype="str"))
    df["property_class"] = merged.get("property_class_code", merged.get("class_code"))
    df["property_class_desc"] = merged.get("prop_class_description", merged.get("bldg_style_desc"))
    df["sale_price"] = _safe_float(merged["sale_price_sale"])
    df["sale_date"] = _parse_buffalo_date(merged["sale_date"])
    df["assessed_value"] = _safe_float(merged.get("total_value", pd.Series(dtype="float")))
    df["land_value"] = _safe_float(merged.get("land_value", pd.Series(dtype="float")))
    df["year_built"] = _safe_float(merged.get("year_built", pd.Series(dtype="float")))
    df["total_living_area"] = _safe_float(merged.get("total_living_area", pd.Series(dtype="float")))
    df["first_floor_area"] = _safe_float(merged.get("first_story_area", pd.Series(dtype="float")))
    df["second_floor_area"] = _safe_float(merged.get("second_story_area", pd.Series(dtype="float")))
    df["beds"] = _safe_float(merged.get("of_beds", pd.Series(dtype="float")))
    df["baths"] = _safe_float(merged.get("_of_baths", pd.Series(dtype="float")))
    df["half_baths"] = pd.Series(0.0, index=merged.index)
    df["kitchens"] = _safe_float(merged.get("of_kitchens", pd.Series(dtype="float")))
    df["stories"] = _safe_float(merged.get("nostory", pd.Series(dtype="float")))
    df["fireplaces"] = _safe_float(merged.get("of_fireplaces", pd.Series(dtype="float")))
    df["lot_frontage"] = _safe_float(merged.get("front", pd.Series(dtype="float")))
    df["lot_depth"] = _safe_float(merged.get("depth", pd.Series(dtype="float")))
    df["lot_acres"] = _safe_float(merged.get("acres", pd.Series(dtype="float")))
    df["building_style"] = merged.get("building_style_desc", merged.get("bldg_style_desc"))
    df["overall_condition"] = merged.get("overall_condition_description", merged.get("condition"))
    df["construction_grade"] = merged.get("construction_grade_description", pd.Series(dtype="str"))
    df["exterior_wall"] = merged.get("exterior_wall_description", pd.Series(dtype="str"))
    df["heat_type"] = merged.get("heat_type_description", pd.Series(dtype="str"))
    df["central_air"] = _safe_float(merged.get("central_air", pd.Series(dtype="float"))).map(
        {0.0: False, 1.0: True}
    )
    df["fuel_type"] = pd.Series(dtype="str")
    df["basement_type"] = merged.get("basement_type", pd.Series(dtype="str"))
    df["latitude"] = _safe_float(merged.get("latitude", pd.Series(dtype="float")))
    df["longitude"] = _safe_float(merged.get("longitude", pd.Series(dtype="float")))
    df["school_district"] = pd.Series(dtype="str")

    df = df.dropna(subset=["sale_price", "total_living_area"])
    df = df[df["sale_price"] > 0]
    df = df[df["total_living_area"] > 0]

    logger.info("Buffalo: %d clean training records after filtering", len(df))
    return df


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
    """Fetch all records from a Rochester ArcGIS REST API layer/table."""
    query_url = f"{ROCHESTER_BASE_URL}/{layer_id}/query"

    count_params = {"where": where, "returnCountOnly": "true", "f": "json"}
    resp = requests.get(query_url, params=count_params, timeout=60)
    resp.raise_for_status()
    total_count = resp.json().get("count", 0)
    logger.info("ArcGIS layer %d: %d total records matching '%s'", layer_id, total_count, where[:60])

    if total_count == 0:
        return pd.DataFrame()

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


def _rochester_safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _rochester_parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _rochester_to_unified(merged: pd.DataFrame, source_label: str = "rochester") -> pd.DataFrame:
    """Map a merged Rochester parcel+structure DataFrame to the unified schema."""
    def col(name: str, default=None) -> pd.Series:
        if name in merged.columns:
            return merged[name]
        return pd.Series([default] * len(merged), index=merged.index)

    df = pd.DataFrame(index=merged.index)
    df["source"] = source_label
    df["parcel_id"] = col("PARCELID")
    df["print_key"] = col("PRINTKEY")
    df["address"] = col("SITEADDRESS")
    df["city"] = col("CITY", "Rochester")
    df["zip_code"] = col("ZIP5")
    df["property_class"] = col("CLASSCD")
    df["property_class_desc"] = col("CLASSDSCRP")
    df["sale_price"] = _rochester_safe_float(col("SALE_PRICE"))
    df["sale_date"] = _rochester_parse_date(col("SALE_DATE"))
    df["assessed_value"] = _rochester_safe_float(col("CURRENT_TOTAL_VALUE"))
    df["land_value"] = _rochester_safe_float(col("CURRENT_LAND_VALUE"))
    df["year_built"] = _rochester_safe_float(col("YR_BUILT"))
    df["total_living_area"] = _rochester_safe_float(col("RESFLRAREA"))
    df["first_floor_area"] = _rochester_safe_float(col("FIRST_FLOOR_AREA"))
    df["second_floor_area"] = _rochester_safe_float(col("SECOND_FLOOR_AREA"))
    df["beds"] = _rochester_safe_float(col("BEDS"))
    df["baths"] = _rochester_safe_float(col("BATHS"))
    df["half_baths"] = _rochester_safe_float(col("HALFBATH"))
    df["kitchens"] = _rochester_safe_float(col("KITCHENS"))
    df["stories"] = _rochester_safe_float(col("STORIES"))
    df["fireplaces"] = _rochester_safe_float(col("FIREPLACES"))
    df["lot_frontage"] = _rochester_safe_float(col("LOT_FRONTAGE"))
    df["lot_depth"] = _rochester_safe_float(col("LOT_DEPTH"))
    df["lot_acres"] = _rochester_safe_float(col("SHAPEACRES"))
    df["building_style"] = col("BLDGSTYLE")
    df["overall_condition"] = col("OVER_COND")
    df["construction_grade"] = col("GRADE")
    df["exterior_wall"] = col("EXT_WALL")
    df["heat_type"] = col("HEAT_TYPE")
    df["central_air"] = col("AIR_COND").map({"Yes": True, "No": False})
    df["fuel_type"] = col("FUEL_TYPE")
    df["basement_type"] = col("BASEMENT_TYPE")

    raw_lat = _rochester_safe_float(col("_latitude"))
    raw_lng = _rochester_safe_float(col("_longitude"))
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
            logger.info("Converted %d Rochester coordinates from EPSG:3857 → EPSG:4326", valid_coords.sum())
        except ImportError:
            logger.warning("pyproj not installed. Rochester coordinates may be in wrong CRS.")

    df["latitude"] = raw_lat
    df["longitude"] = raw_lng
    df["school_district"] = pd.Series([None] * len(merged), index=merged.index, dtype="object")

    return df


def _load_rochester_training_data() -> pd.DataFrame:
    """Load and normalize Rochester property data."""
    parcel_df = _fetch_arcgis(
        ROCHESTER_PARCEL_LAYER,
        where="SALE_PRICE > 0 AND CLASSCD IN ('210','220','230','240','250')",
        out_fields="PARCELID,PRINTKEY,SITEADDRESS,CITY,ZIP5,CLASSCD,CLASSDSCRP,"
                   "SALE_PRICE,SALE_DATE,CURRENT_TOTAL_VALUE,CURRENT_LAND_VALUE,"
                   "LOT_FRONTAGE,LOT_DEPTH,SHAPEACRES,NORESUNITS,VALID",
        return_geometry=True,
    )

    if parcel_df.empty:
        logger.warning("Rochester parcel data returned empty")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    structure_df = _fetch_arcgis(
        ROCHESTER_RESIDENTIAL_TABLE,
        where="1=1",
        out_fields="SBL,BEDS,BATHS,HALFBATH,RESFLRAREA,FIRST_FLOOR_AREA,"
                   "SECOND_FLOOR_AREA,YR_BUILT,STORIES,BLDGSTYLE,GRADE,"
                   "OVER_COND,EXT_WALL,HEAT_TYPE,AIR_COND,FUEL_TYPE,"
                   "BASEMENT_TYPE,FIREPLACES,KITCHENS",
    )

    if structure_df.empty:
        logger.warning("Rochester structure data returned empty")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    merged = parcel_df.merge(
        structure_df,
        left_on="PARCELID",
        right_on="SBL",
        how="inner",
    )

    df = _rochester_to_unified(merged, source_label="rochester")

    df = df.dropna(subset=["sale_price", "total_living_area"])
    df = df[(df["sale_price"] > 100000) & (df["sale_price"] < 600000)]
    df = df[df["total_living_area"] > 0]
    df = df[df["sale_date"] > "2022-01-01"]

    logger.info("Rochester: %d clean training records after filtering", len(df))
    return df


# =============================================================================
# Combined Loader (used by pipeline and train)
# =============================================================================

def _load_combined_training_data() -> pd.DataFrame:
    """Load and combine training data from all WNY sources.

    Returns:
        Combined DataFrame with data from all sources, unified schema.
    """
    logger.info("Loading combined WNY training data...")

    dfs = []

    try:
        buffalo_df = _load_buffalo_training_data()
        dfs.append(buffalo_df)
        logger.info("Buffalo: loaded %d records", len(buffalo_df))
    except Exception as e:
        logger.error("Failed to load Buffalo data: %s", e)

    try:
        rochester_df = _load_rochester_training_data()
        dfs.append(rochester_df)
        logger.info("Rochester: loaded %d records", len(rochester_df))
    except Exception as e:
        logger.error("Failed to load Rochester data: %s", e)

    if not dfs:
        logger.error("No data loaded from any source")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    combined = pd.concat(dfs, ignore_index=True)

    for col in UNIFIED_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.Series(dtype="object")

    combined = combined[UNIFIED_COLUMNS]

    logger.info(
        "Combined training data: %d records (%d Buffalo, %d Rochester)",
        len(combined),
        len(combined[combined["source"] == "buffalo"]),
        len(combined[combined["source"] == "rochester"]),
    )

    return combined


# =============================================================================
# Training Data Sources (Geronimo-compliant)
# =============================================================================

# Primary training data source — full historical sales population
training_sales = DataSource(
    name="training_sales",
    source="function",
    handle=_load_combined_training_data,
)


# =============================================================================
# Auto-collect sources
# =============================================================================

# Automatically collect training and production sources from module
training_sources = collect_data_sources(sys.modules[__name__], "training_")
production_sources = collect_data_sources(sys.modules[__name__], "production_")

