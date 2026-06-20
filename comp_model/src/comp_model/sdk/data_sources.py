"""Data source definitions for comp_model.

Fetches and normalizes property assessment + sales data from two WNY sources:
  - Buffalo (City): Socrata SODA API (datasets 4t8s-9yih + tndx-c4xz)
  - Rochester (City): ArcGIS REST API (ROC_Parcel_Query_RPS_Merged service)

Both sources are joined and normalized into a common schema so the model
sees a uniform DataFrame regardless of the originating city.

NAMING CONVENTIONS:
- training_* : DataSources used for model training
- production_* : DataSources used for production inference/batch scoring

JOIN BEHAVIOR:
- The FIRST DataSource in each group is the primary source
- Subsequent DataSources are joined to the primary using their join_spec
- All DataSources in a group should share a common primary key

This module is imported by model.py and pipeline.py to load training/scoring data.
"""

import logging
import math
import os
import sys
from typing import Optional

import pandas as pd
import requests

from geronimo.data_sources import DataSource, JoinSpec, Query, collect_data_sources

logger = logging.getLogger(__name__)

# =============================================================================
# Unified Schema
# =============================================================================
# Both Buffalo and Rochester data are normalized to these column names.
# This is the contract between data ingestion and the feature/model layers.

UNIFIED_COLUMNS = [
    "source",              # "buffalo" or "rochester"
    "parcel_id",           # Canonical parcel identifier (print_key for Buffalo, PARCELID for Rochester)
    "print_key",           # Human-readable parcel key
    "address",             # Full street address
    "city",                # City name
    "zip_code",            # 5-digit ZIP
    "property_class",      # NYS property class code (e.g. "210")
    "property_class_desc", # Description (e.g. "1 Family Residential")
    "sale_price",          # Actual transaction price ($)
    "sale_date",           # Date of sale (datetime)
    "assessed_value",      # Current total assessed value ($)
    "land_value",          # Assessed land value ($)
    "year_built",          # Year of construction
    "total_living_area",   # Total habitable sq ft
    "first_floor_area",    # First floor sq ft
    "second_floor_area",   # Second floor sq ft
    "beds",                # Number of bedrooms
    "baths",               # Number of full bathrooms
    "half_baths",          # Number of half bathrooms
    "kitchens",            # Number of kitchens
    "stories",             # Number of stories
    "fireplaces",          # Number of fireplaces
    "lot_frontage",        # Lot frontage (ft)
    "lot_depth",           # Lot depth (ft)
    "lot_acres",           # Lot size in acres
    "building_style",      # Style description (e.g. "Ranch", "Old style")
    "overall_condition",   # Condition (e.g. "Normal", "Good")
    "construction_grade",  # Grade (e.g. "Average", "Good")
    "exterior_wall",       # Exterior wall material
    "heat_type",           # Heating type
    "central_air",         # Central air (boolean)
    "fuel_type",           # Fuel type (Rochester only, NaN for Buffalo)
    "basement_type",       # Basement type description
    "latitude",            # Geographic latitude
    "longitude",           # Geographic longitude
    "school_district",     # Assigned public school district
]


# =============================================================================
# Buffalo (Socrata SODA API)
# =============================================================================

BUFFALO_ASSESSMENT_URL = "https://data.buffalony.gov/resource/4t8s-9yih.json"
BUFFALO_SALES_URL = "https://data.buffalony.gov/resource/tndx-c4xz.json"
BUFFALO_SOCRATA_LIMIT = 50_000  # Socrata max per page


def _fetch_socrata(url: str, where: Optional[str] = None, limit: int = BUFFALO_SOCRATA_LIMIT) -> pd.DataFrame:
    """Fetch all records from a Socrata SODA API endpoint with pagination.

    Args:
        url: Socrata resource URL (e.g. https://data.buffalony.gov/resource/XXXX.json).
        where: Optional SoQL WHERE clause to filter records.
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

        # If we got fewer records than the limit, we've reached the end
        if len(batch) < limit:
            break

    logger.info("Fetched %d total records from %s", len(all_records), url)
    return pd.DataFrame(all_records)


def _load_buffalo_training_data() -> pd.DataFrame:
    """Load and normalize Buffalo property data for training.

    Fetches the assessment roll (physical characteristics) and property sales
    (actual transaction prices), joins them on print_key, and normalizes to
    the unified schema.

    Returns:
        DataFrame in unified schema with one row per property sale.
    """
    # --- Fetch assessment roll (property characteristics) ---
    # Filter to residential properties (class 210=1-family, 220=2-family, etc.)
    assessment_df = _fetch_socrata(
        BUFFALO_ASSESSMENT_URL,
        where="property_class_code IN ('210', '220', '230', '240', '250')",
    )

    if assessment_df.empty:
        logger.warning("Buffalo assessment data returned empty")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # --- Fetch sales data ---
    sales_df = _fetch_socrata(
        BUFFALO_SALES_URL,
        where="sale_price > '0'",  # Exclude $0 sales (non-arm's-length)
    )

    if sales_df.empty:
        logger.warning("Buffalo sales data returned empty")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # --- Join assessment + sales on print_key ---
    merged = sales_df.merge(
        assessment_df,
        on="print_key",
        how="inner",
        suffixes=("_sale", "_assess"),
    )

    logger.info(
        "Buffalo: %d assessments × %d sales → %d joined records",
        len(assessment_df), len(sales_df), len(merged),
    )

    # --- Normalize to unified schema ---
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
    df["half_baths"] = pd.Series(0.0, index=merged.index)  # Buffalo doesn't split half baths
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
    df["fuel_type"] = pd.Series(dtype="str")  # Not available in Buffalo data
    df["basement_type"] = merged.get("basement_type", pd.Series(dtype="str"))
    df["latitude"] = _safe_float(merged.get("latitude", pd.Series(dtype="float")))
    df["longitude"] = _safe_float(merged.get("longitude", pd.Series(dtype="float")))
    df["school_district"] = pd.Series(dtype="str") # Handled later by spatial join

    # Filter out invalid records
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
ROCHESTER_PARCEL_LAYER = 0       # TaxParcel layer
ROCHESTER_RESIDENTIAL_TABLE = 2  # ResidentialStructure table
ROCHESTER_MAX_RECORDS = 200_000  # Server maxRecordCount


def _fetch_arcgis(
    layer_id: int,
    where: str = "1=1",
    out_fields: str = "*",
    return_geometry: bool = False,
) -> pd.DataFrame:
    """Fetch all records from a Rochester ArcGIS REST API layer/table.

    Handles pagination automatically using resultOffset if the record count
    exceeds the server's maxRecordCount.

    Args:
        layer_id: Layer or table ID within the MapServer.
        where: SQL WHERE clause for filtering.
        out_fields: Comma-separated field names or "*" for all.
        return_geometry: Whether to include geometry in response.

    Returns:
        DataFrame with all matching records.
    """
    query_url = f"{ROCHESTER_BASE_URL}/{layer_id}/query"

    # First, get total record count
    count_params = {"where": where, "returnCountOnly": "true", "f": "json"}
    resp = requests.get(query_url, params=count_params, timeout=60)
    resp.raise_for_status()
    total_count = resp.json().get("count", 0)
    logger.info("ArcGIS layer %d: %d total records matching '%s'", layer_id, total_count, where[:60])

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

        # Extract attributes and compute polygon centroid for coordinates
        for f in features:
            attrs = f["attributes"]
            if return_geometry:
                rings = f.get("geometry", {}).get("rings", [])
                if rings and len(rings[0]) > 0:
                    ring = rings[0]
                    # Calculate simple mean center
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
    """Rochester dates come as 'MM/DD/YYYY' strings."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _rochester_to_unified(merged: pd.DataFrame, source_label: str = "rochester") -> pd.DataFrame:
    """Map a merged Rochester parcel+structure DataFrame to the unified schema.

    Used by both the bulk training loader and the single-property lookup. Tolerant
    of missing source columns — SALE_PRICE/SALE_DATE are absent on lookup queries
    that don't filter on sales history.
    """
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

    # Rochester ArcGIS returns polygon centroids in Web Mercator (EPSG:3857).
    # Convert to WGS84 (EPSG:4326) for school district join and H3 lookups.
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
    """Load and normalize Rochester property data for training.

    Fetches TaxParcel (sale prices + assessments) and ResidentialStructure
    (physical characteristics), joins on PARCELID=SBL, and normalizes to
    the unified schema.

    Returns:
        DataFrame in unified schema with one row per property.
    """
    # --- Fetch parcel data (has sale prices + assessed values) ---
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

    # --- Fetch residential structure data (has beds/baths/sqft/yr_built) ---
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

    # --- Join parcel + structure on PARCELID = SBL ---
    merged = parcel_df.merge(
        structure_df,
        left_on="PARCELID",
        right_on="SBL",
        how="inner",
    )

    logger.info(
        "Rochester: %d parcels × %d structures → %d joined records",
        len(parcel_df), len(structure_df), len(merged),
    )

    # --- Normalize to unified schema ---
    df = _rochester_to_unified(merged, source_label="rochester")

    # Filter out invalid records
    df = df.dropna(subset=["sale_price", "total_living_area"])
    df = df[(df["sale_price"] > 100000)
         &  (df["sale_price"] <600000)]
    df = df[df["total_living_area"] > 0]
    df = df[df["sale_date"] > "2022-01-01"]

    # Assign school districts via TIGER shapefile spatial join
    df = _assign_school_districts(df)

    logger.info("Rochester: %d clean training records after filtering", len(df))
    return df


# =============================================================================
# Rochester Single-Property Lookup (inference-time)
# =============================================================================

# Public residential property classes (1–5 unit residential). Matches the
# training query's CLASSCD filter so lookup results stay shape-compatible
# with the model's training distribution.
ROCHESTER_RESIDENTIAL_CLASS_CODES = ("210", "220", "230", "240", "250")


def lookup_rochester_assessment(
    address: str,
    zip_code: Optional[str] = None,
) -> Optional[dict]:
    """Look up a single City of Rochester parcel's current assessment by address.

    Uses the same MapServer that powers training (`_load_rochester_training_data`)
    but with a targeted `SITEADDRESS LIKE` filter — no `SALE_PRICE > 0` constraint,
    so it returns the current assessment whether or not the parcel has ever sold.

    Reuses `_fetch_arcgis`, `_rochester_to_unified`, and `_assign_school_districts`
    end-to-end. Returns a unified-schema dict (NaN/None entries dropped) or None
    if no City of Rochester parcel matches.

    Note: Spencerport, Greece, Henrietta, and other Monroe County suburbs are NOT
    in this MapServer's coverage — they live in their towns' assessment rolls.
    """
    if not address:
        return None

    # Normalize: keep just the street portion (Zillow often returns
    # "41 Bowery St, Spencerport, NY 14559"), uppercase for case-insensitive
    # match against the ArcGIS field, and escape single quotes for the SQL clause.
    street = address.split(",")[0].strip().upper().replace("'", "''")
    if not street:
        return None

    class_list = ",".join(f"'{c}'" for c in ROCHESTER_RESIDENTIAL_CLASS_CODES)
    where = (
        f"UPPER(SITEADDRESS) LIKE '%{street}%' "
        f"AND CLASSCD IN ({class_list})"
    )
    if zip_code:
        zip_clean = str(zip_code).strip().replace("'", "")[:5]
        if zip_clean:
            where += f" AND ZIP5 = '{zip_clean}'"

    parcel_df = _fetch_arcgis(
        ROCHESTER_PARCEL_LAYER,
        where=where,
        out_fields="PARCELID,PRINTKEY,SITEADDRESS,CITY,ZIP5,CLASSCD,CLASSDSCRP,"
                   "CURRENT_TOTAL_VALUE,CURRENT_LAND_VALUE,"
                   "LOT_FRONTAGE,LOT_DEPTH,SHAPEACRES",
        return_geometry=True,
    )

    if parcel_df.empty:
        logger.info("Rochester lookup: no parcel match for '%s' zip=%s", street, zip_code)
        return None

    if len(parcel_df) > 1:
        logger.warning(
            "Rochester lookup: %d parcels matched '%s'; using first (%s)",
            len(parcel_df), street, parcel_df["SITEADDRESS"].iloc[0],
        )
    parcel_df = parcel_df.head(1).reset_index(drop=True)

    # Fetch only the matching structure row instead of the whole table
    parcel_id = str(parcel_df["PARCELID"].iloc[0]).replace("'", "''")
    structure_df = _fetch_arcgis(
        ROCHESTER_RESIDENTIAL_TABLE,
        where=f"SBL = '{parcel_id}'",
        out_fields="SBL,BEDS,BATHS,HALFBATH,RESFLRAREA,FIRST_FLOOR_AREA,"
                   "SECOND_FLOOR_AREA,YR_BUILT,STORIES,BLDGSTYLE,GRADE,"
                   "OVER_COND,EXT_WALL,HEAT_TYPE,AIR_COND,FUEL_TYPE,"
                   "BASEMENT_TYPE,FIREPLACES,KITCHENS",
    )

    if structure_df.empty:
        merged = parcel_df  # assessment-only; physical fields stay NaN
    else:
        merged = parcel_df.merge(
            structure_df, left_on="PARCELID", right_on="SBL", how="left"
        )

    unified = _rochester_to_unified(merged, source_label="rochester")
    unified = _assign_school_districts(unified)

    row = unified.iloc[0].to_dict()
    return {k: v for k, v in row.items() if pd.notna(v) and v != ""}


# =============================================================================
# Combined Loader
# =============================================================================

def _assign_school_districts(df: pd.DataFrame) -> pd.DataFrame:
    """Download NYS school district shapefile and spatial join to df."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import urllib.request
        import zipfile
        import os
    except ImportError:
        logger.warning("geopandas or shapely not installed. Skipping school district tagging.")
        return df

    valid_coords = df['latitude'].notna() & df['longitude'].notna()
    if not valid_coords.any():
        return df
        
    shapefile_url = "https://www2.census.gov/geo/tiger/TIGER2023/UNSD/tl_2023_36_unsd.zip"
    cache_dir = "/tmp/tiger_school_districts"
    shapefile_path = os.path.join(cache_dir, "tl_2023_36_unsd.shp")
    
    if not os.path.exists(shapefile_path):
        logger.info("Downloading NYS School District boundaries...")
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "tl_2023_36_unsd.zip")
        urllib.request.urlretrieve(shapefile_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
            
    logger.info("Loading School District boundaries into GeoPandas...")
    districts_gdf = gpd.read_file(shapefile_path)
    districts_gdf = districts_gdf.to_crs("EPSG:4326")
    
    logger.info("Performing point spatial join...")
    # Reset index to avoid gaps from upstream filtering
    df = df.reset_index(drop=True)
    valid_coords = df['latitude'].notna() & df['longitude'].notna()

    points = [Point(xy) for xy in zip(df.loc[valid_coords, 'longitude'], df.loc[valid_coords, 'latitude'])]
    gdf = gpd.GeoDataFrame(df.loc[valid_coords].copy(), geometry=points, crs="EPSG:4326")
    
    joined = gpd.sjoin(gdf, districts_gdf[['NAME', 'geometry']], how="left", predicate="within")
    # Drop duplicates from overlapping polygons — keep first match
    joined = joined[~joined.index.duplicated(keep='first')]
    
    # Assign back to unified schema using explicit index alignment
    df.loc[valid_coords, 'school_district'] = joined['NAME'].values
    assigned = df['school_district'].notna().sum()
    logger.info("Assigned school districts to %d / %d records", assigned, len(df))
    return df


def _load_combined_training_data() -> pd.DataFrame:
    """Load and combine training data from all WNY sources.

    Fetches Buffalo + Rochester data in parallel-ready fashion, normalizes
    both to the unified schema, and concatenates into a single DataFrame.

    Returns:
        Combined DataFrame with data from all sources, unified schema.
    """
    logger.info("Loading combined WNY training data...")

    dfs = []

    # Buffalo
    try:
        buffalo_df = _load_buffalo_training_data()
        dfs.append(buffalo_df)
        logger.info("Buffalo: loaded %d records", len(buffalo_df))
    except Exception as e:
        logger.error("Failed to load Buffalo data: %s", e)

    # Rochester
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

    # Ensure column order matches the unified schema
    for col in UNIFIED_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.Series(dtype="object")
            
    # Assign School Districts via spatial join before finalizing schema
    combined = _assign_school_districts(combined)
            
    combined = combined[UNIFIED_COLUMNS]

    logger.info(
        "Combined training data: %d records (%d Buffalo, %d Rochester)",
        len(combined),
        len(combined[combined["source"] == "buffalo"]),
        len(combined[combined["source"] == "rochester"]),
    )

    return combined


# =============================================================================
# =============================================================================
# FRED Macro-Economic Data
# =============================================================================

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

def _fetch_fred_series(series_id: str, api_key: str) -> pd.DataFrame:
    """Fetch a data series from FRED and return it as a DataFrame."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json"
    }
    logger.info("Fetching FRED series %s...", series_id)
    resp = requests.get(FRED_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json().get("observations", [])
    
    if not data:
        logger.warning("FRED series %s returned empty data", series_id)
        return pd.DataFrame(columns=["date", series_id])
        
    df = pd.DataFrame(data)
    # FRED returns 'value' and 'date'. Drop '.' (missing) values
    df = df[df["value"] != "."].copy()
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df["value"])
    
    return df[["date", series_id]].sort_values("date")

def _load_macro_economic_data() -> pd.DataFrame:
    """Load macro economic indicators from FRED, aligned dynamically.
    
    Returns:
        DataFrame containing macro indicators, resampled to daily frequency
        using forward-fill so they can be easily joined with transaction dates.
    """
    # TODO: Use dotenv to load
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY environment variable not set. Returning empty macro data.")
        return pd.DataFrame(columns=["sale_date", "mortgage_30y", "fed_funds", "cpi", "unemployment"])

    series_map = {
        "MORTGAGE30US": "mortgage_30y",
        "FEDFUNDS": "fed_funds",
        "CPIAUCSL": "cpi",
        "UNRATE": "unemployment"
    }
    
    dfs = []
    for series_id, col_name in series_map.items():
        try:
            df = _fetch_fred_series(series_id, api_key)
            if not df.empty:
                df = df.rename(columns={series_id: col_name})
                df = df.set_index("date")
                # Resample to daily and forward fill so we have a value for every day
                df = df.resample("D").ffill()
                dfs.append(df)
        except Exception as e:
            logger.error("Failed to fetch FRED %s: %s", series_id, e)
            
    if not dfs:
        return pd.DataFrame(columns=["sale_date", "mortgage_30y", "fed_funds", "cpi", "unemployment"])
        
    # Combine all series
    macro_df = pd.concat(dfs, axis=1)
    
    # Reset index to make date a column, and rename to sale_date to align with primary dataset
    macro_df = macro_df.reset_index().rename(columns={"date": "sale_date"})
    
    # Forward fill any remaining NaNs across the combination, then drop rows that are all NaN
    macro_df = macro_df.ffill().dropna(subset=list(series_map.values()), how="all")
    
    logger.info("Loaded FRED macro data: %d daily rows", len(macro_df))
    return macro_df


# =============================================================================
# Training Data Sources (Geronimo-compliant)
# =============================================================================

# Combined data source — primary training source with all WNY data
training_properties = DataSource(
    name="training_properties",
    source="function",
    handle=_load_combined_training_data,
)

# Individual city sources — useful for debugging or city-specific training
training_buffalo = DataSource(
    name="training_buffalo",
    source="function",
    handle=_load_buffalo_training_data,
)

training_rochester = DataSource(
    name="training_rochester",
    source="function",
    handle=_load_rochester_training_data,
)

training_macro = DataSource(
    name="training_macro",
    source="function",
    handle=_load_macro_economic_data,
    join_spec=JoinSpec(
        left_on="sale_date",
        right_on="sale_date",
        how="left"
    )
)


# =============================================================================
# Production Data Sources
# =============================================================================

# Production sources use the same loaders — in production the model scores
# against current assessment data, not historical sales.

production_properties = DataSource(
    name="production_properties",
    source="function",
    handle=_load_combined_training_data,
)

production_macro = DataSource(
    name="production_macro",
    source="function",
    handle=_load_macro_economic_data,
    join_spec=JoinSpec(
        left_on="sale_date",
        right_on="sale_date",
        how="left"
    )
)


# =============================================================================
# Auto-collect sources
# =============================================================================

# Automatically collect training and production sources from module
training_sources = collect_data_sources(sys.modules[__name__], "training_")
production_sources = collect_data_sources(sys.modules[__name__], "production_")
