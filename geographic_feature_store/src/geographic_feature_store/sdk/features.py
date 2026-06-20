"""Geographic velocity feature definitions.

Each precomputed velocity grid is declared as a Geronimo Feature with a
derived_feature_fn that performs the BallTree computation. This makes the
FeatureSet self-documenting — you can read the class attributes to see
exactly which (radius, lookback_window) combinations are available.

Fitting:
    calls each Feature's derived_feature_fn on the sales data, which runs
    the BallTree computation for that specific (radius, window) combo.
    The computed grid is stored as fitted state.

Transforming:
    maps input lat/lng → H3 cell and looks up precomputed values from the
    fitted grid. Cheap O(n) operation.

H3 Resolution Reference:
    Res 7: ~1.22 km edge (~0.76 mi)  — coarse, fast
    Res 8: ~0.46 km edge (~0.29 mi)  — default, good balance
    Res 9: ~0.17 km edge (~0.11 mi)  — fine, 7x more cells
"""

import logging
from datetime import datetime
from typing import Optional

import h3
import numpy as np
import pandas as pd

from geronimo.features import FeatureSet, Feature

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration defaults
# =============================================================================

DEFAULT_H3_RESOLUTION = 8
EARTH_RADIUS_MILES = 3958.8

METROS = ["buffalo", "rochester"]

DEFAULT_METRO_BOUNDS = {
    "buffalo": {
        "min_lat": 42.70, "max_lat": 43.10,
        "min_lng": -79.05, "max_lng": -78.60,
    },
    "rochester": {
        "min_lat": 43.05, "max_lat": 43.30,
        "min_lng": -77.80, "max_lng": -77.40,
    },
}


# =============================================================================
# Artifact naming
# =============================================================================


def artifact_name(radius_miles: float, lookback_days: int) -> str:
    """Generate a consistent artifact name for a (radius, window) combo.

    Examples:
        artifact_name(1.0, 30)  -> "geo_velocity_r1_w30"
        artifact_name(5.0, 90)  -> "geo_velocity_r5_w90"
        artifact_name(10.0, 365) -> "geo_velocity_r10_w365"
    """
    r_str = str(int(radius_miles)) if radius_miles == int(radius_miles) else str(radius_miles)
    return f"geo_velocity_r{r_str}_w{lookback_days}"


# =============================================================================
# Grid generation
# =============================================================================


def generate_grid(
    sales_df: pd.DataFrame,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> pd.DataFrame:
    """Generate H3 grid cells covering the extent of the sales data.

    Derives bounding boxes dynamically from sales data per metro,
    falling back to conservative defaults.

    Args:
        sales_df: Sales DataFrame with 'source', 'latitude', 'longitude'.
        resolution: H3 grid resolution.

    Returns:
        DataFrame with columns: h3_index, metro, latitude, longitude
    """
    BUFFER_DEGREES = 0.05  # ~3.5 miles at WNY latitudes
    grids = []

    for metro in METROS:
        # Dynamic bounds from data
        bounds = None
        if "source" in sales_df.columns:
            metro_sales = sales_df[sales_df["source"] == metro].dropna(
                subset=["latitude", "longitude"]
            )
            if not metro_sales.empty:
                bounds = {
                    "min_lat": metro_sales["latitude"].min() - BUFFER_DEGREES,
                    "max_lat": metro_sales["latitude"].max() + BUFFER_DEGREES,
                    "min_lng": metro_sales["longitude"].min() - BUFFER_DEGREES,
                    "max_lng": metro_sales["longitude"].max() + BUFFER_DEGREES,
                }

        # Fallback to defaults
        if bounds is None:
            bounds = DEFAULT_METRO_BOUNDS.get(metro)
        if bounds is None:
            continue

        step = max(0.001, 0.004 * (2 ** (8 - resolution)))
        cells = set()
        for lat in np.arange(bounds["min_lat"], bounds["max_lat"], step):
            for lng in np.arange(bounds["min_lng"], bounds["max_lng"], step):
                cells.add(h3.latlng_to_cell(lat, lng, resolution))

        grid_data = []
        for cell in cells:
            lat, lng = h3.cell_to_latlng(cell)
            grid_data.append({
                "h3_index": cell, "metro": metro,
                "latitude": lat, "longitude": lng,
            })

        grid = pd.DataFrame(grid_data)
        grids.append(grid)
        logger.info("  %s: %d H3 cells", metro, len(grid))

    if not grids:
        return pd.DataFrame(columns=["h3_index", "metro", "latitude", "longitude"])
    return pd.concat(grids, ignore_index=True)


# =============================================================================
# Velocity computation for a single (radius, window) combo
# =============================================================================


def compute_single_velocity(
    grid_points: pd.DataFrame,
    sales_df: pd.DataFrame,
    reference_date: datetime,
    radius_miles: float,
    lookback_days: int,
) -> pd.DataFrame:
    """Compute velocity at each grid cell for ONE (radius, window) combo.

    This is the function wired into each Feature's derived_feature_fn.

    Args:
        grid_points: DataFrame with h3_index, metro, latitude, longitude.
        sales_df: DataFrame with latitude, longitude, sale_date, sale_price, total_living_area.
        reference_date: "As of" date.
        radius_miles: Search radius in miles.
        lookback_days: Lookback window in days.

    Returns:
        DataFrame with columns: h3_index, metro, latitude, longitude,
        sales_volume, median_ppsf, reference_date, computed_at
    """
    from sklearn.neighbors import BallTree

    if grid_points.empty or sales_df.empty:
        return _empty_velocity_df()

    # Prepare sales
    sales = sales_df.copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])
    sales = sales.dropna(subset=["latitude", "longitude", "sale_date", "sale_price"])
    ref_dt = pd.Timestamp(reference_date)
    sales = sales[sales["sale_date"] < ref_dt].reset_index(drop=True)

    if sales.empty:
        return _empty_velocity_df()

    # Price per square foot
    areas = sales["total_living_area"].fillna(0).values
    prices = sales["sale_price"].fillna(0).values
    with np.errstate(divide="ignore", invalid="ignore"):
        ppsf = np.where(areas > 0, prices / areas, prices)

    sale_dates = sales["sale_date"].values

    # BallTree
    sales_coords_rad = np.radians(sales[["latitude", "longitude"]].fillna(0).values)
    tree = BallTree(sales_coords_rad, metric="haversine")
    grid_coords_rad = np.radians(grid_points[["latitude", "longitude"]].fillna(0).values)

    radius_rad = radius_miles / EARTH_RADIUS_MILES
    neighbor_indices, neighbor_distances = tree.query_radius(
        grid_coords_rad, r=radius_rad, return_distance=True
    )

    n_grid = len(grid_points)
    lookback_td = np.timedelta64(lookback_days, "D")
    min_date = ref_dt - lookback_td

    vol_col = np.zeros(n_grid)
    median_ppsf_col = np.full(n_grid, np.nan)

    for i in range(n_grid):
        idx = neighbor_indices[i].astype(int)
        if len(idx) == 0:
            continue
        dates = sale_dates[idx]
        time_mask = dates >= min_date
        valid_idx = idx[time_mask]
        count = len(valid_idx)
        vol_col[i] = count
        if count > 0:
            median_ppsf_col[i] = np.median(ppsf[valid_idx])

    computed_at = datetime.utcnow()
    return pd.DataFrame({
        "h3_index": grid_points["h3_index"].values,
        "metro": grid_points["metro"].values,
        "latitude": grid_points["latitude"].values,
        "longitude": grid_points["longitude"].values,
        "sales_volume": vol_col.astype(int),
        "median_ppsf": median_ppsf_col,
        "reference_date": ref_dt.date(),
        "computed_at": computed_at,
    })


def _empty_velocity_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "h3_index", "metro", "latitude", "longitude",
        "sales_volume", "median_ppsf", "reference_date", "computed_at",
    ])


# =============================================================================
# Feature builder functions — one per (radius, window) combo
#
# Each returns a tuple of (sales_volume_series, median_ppsf_series).
# The FeatureSet wires these as derived_feature_fn on individual Feature
# descriptors, making the set of precomputed grids self-documenting.
# =============================================================================


def _velocity_r1_w30(df):
    return df["_vel_r1_w30_sales_volume"], df["_vel_r1_w30_median_ppsf"]

def _velocity_r1_w90(df):
    return df["_vel_r1_w90_sales_volume"], df["_vel_r1_w90_median_ppsf"]

def _velocity_r1_w180(df):
    return df["_vel_r1_w180_sales_volume"], df["_vel_r1_w180_median_ppsf"]

def _velocity_r1_w365(df):
    return df["_vel_r1_w365_sales_volume"], df["_vel_r1_w365_median_ppsf"]

def _velocity_r5_w30(df):
    return df["_vel_r5_w30_sales_volume"], df["_vel_r5_w30_median_ppsf"]

def _velocity_r5_w90(df):
    return df["_vel_r5_w90_sales_volume"], df["_vel_r5_w90_median_ppsf"]

def _velocity_r5_w180(df):
    return df["_vel_r5_w180_sales_volume"], df["_vel_r5_w180_median_ppsf"]

def _velocity_r5_w365(df):
    return df["_vel_r5_w365_sales_volume"], df["_vel_r5_w365_median_ppsf"]

def _velocity_r10_w30(df):
    return df["_vel_r10_w30_sales_volume"], df["_vel_r10_w30_median_ppsf"]

def _velocity_r10_w90(df):
    return df["_vel_r10_w90_sales_volume"], df["_vel_r10_w90_median_ppsf"]

def _velocity_r10_w180(df):
    return df["_vel_r10_w180_sales_volume"], df["_vel_r10_w180_median_ppsf"]

def _velocity_r10_w365(df):
    return df["_vel_r10_w365_sales_volume"], df["_vel_r10_w365_median_ppsf"]


# Volume extractors
def _vol_r1_w30(df): return _velocity_r1_w30(df)[0]
def _ppsf_r1_w30(df): return _velocity_r1_w30(df)[1]
def _vol_r1_w90(df): return _velocity_r1_w90(df)[0]
def _ppsf_r1_w90(df): return _velocity_r1_w90(df)[1]
def _vol_r1_w180(df): return _velocity_r1_w180(df)[0]
def _ppsf_r1_w180(df): return _velocity_r1_w180(df)[1]
def _vol_r1_w365(df): return _velocity_r1_w365(df)[0]
def _ppsf_r1_w365(df): return _velocity_r1_w365(df)[1]

def _vol_r5_w30(df): return _velocity_r5_w30(df)[0]
def _ppsf_r5_w30(df): return _velocity_r5_w30(df)[1]
def _vol_r5_w90(df): return _velocity_r5_w90(df)[0]
def _ppsf_r5_w90(df): return _velocity_r5_w90(df)[1]
def _vol_r5_w180(df): return _velocity_r5_w180(df)[0]
def _ppsf_r5_w180(df): return _velocity_r5_w180(df)[1]
def _vol_r5_w365(df): return _velocity_r5_w365(df)[0]
def _ppsf_r5_w365(df): return _velocity_r5_w365(df)[1]

def _vol_r10_w30(df): return _velocity_r10_w30(df)[0]
def _ppsf_r10_w30(df): return _velocity_r10_w30(df)[1]
def _vol_r10_w90(df): return _velocity_r10_w90(df)[0]
def _ppsf_r10_w90(df): return _velocity_r10_w90(df)[1]
def _vol_r10_w180(df): return _velocity_r10_w180(df)[0]
def _ppsf_r10_w180(df): return _velocity_r10_w180(df)[1]
def _vol_r10_w365(df): return _velocity_r10_w365(df)[0]
def _ppsf_r10_w365(df): return _velocity_r10_w365(df)[1]


# =============================================================================
# FeatureSet — declarative feature catalog
# =============================================================================


class GeoVelocityFeatures(FeatureSet):
    """Geographic velocity features for WNY metro areas.

    Each Feature below is a precomputed velocity grid. Reading this class
    tells you exactly which (radius, lookback_window) combinations are
    available for the model to consume.

    Fitting runs the BallTree computation for each declared Feature.
    Transforming looks up precomputed values by H3 cell for new locations.
    """

    # ------------------------------------------------------------------
    # 1-mile radius: hyperlocal neighborhood activity
    # ------------------------------------------------------------------
    sales_volume_r1_w30 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r1_w30,
        description="Sales volume within 1 mile, 30-day lookback",
    )
    median_ppsf_r1_w30 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r1_w30,
        description="Median price/sqft within 1 mile, 30-day lookback",
    )

    sales_volume_r1_w90 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r1_w90,
        description="Sales volume within 1 mile, 90-day lookback",
    )
    median_ppsf_r1_w90 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r1_w90,
        description="Median price/sqft within 1 mile, 90-day lookback",
    )

    sales_volume_r1_w180 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r1_w180,
        description="Sales volume within 1 mile, 180-day lookback",
    )
    median_ppsf_r1_w180 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r1_w180,
        description="Median price/sqft within 1 mile, 180-day lookback",
    )

    sales_volume_r1_w365 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r1_w365,
        description="Sales volume within 1 mile, 365-day lookback",
    )
    median_ppsf_r1_w365 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r1_w365,
        description="Median price/sqft within 1 mile, 365-day lookback",
    )

    # ------------------------------------------------------------------
    # 5-mile radius: neighborhood-level market signal
    # ------------------------------------------------------------------
    sales_volume_r5_w30 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r5_w30,
        description="Sales volume within 5 miles, 30-day lookback",
    )
    median_ppsf_r5_w30 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r5_w30,
        description="Median price/sqft within 5 miles, 30-day lookback",
    )

    sales_volume_r5_w90 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r5_w90,
        description="Sales volume within 5 miles, 90-day lookback",
    )
    median_ppsf_r5_w90 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r5_w90,
        description="Median price/sqft within 5 miles, 90-day lookback",
    )

    sales_volume_r5_w180 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r5_w180,
        description="Sales volume within 5 miles, 180-day lookback",
    )
    median_ppsf_r5_w180 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r5_w180,
        description="Median price/sqft within 5 miles, 180-day lookback",
    )

    sales_volume_r5_w365 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r5_w365,
        description="Sales volume within 5 miles, 365-day lookback",
    )
    median_ppsf_r5_w365 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r5_w365,
        description="Median price/sqft within 5 miles, 365-day lookback",
    )

    # ------------------------------------------------------------------
    # 10-mile radius: metro-level market context
    # ------------------------------------------------------------------
    sales_volume_r10_w30 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r10_w30,
        description="Sales volume within 10 miles, 30-day lookback",
    )
    median_ppsf_r10_w30 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r10_w30,
        description="Median price/sqft within 10 miles, 30-day lookback",
    )

    sales_volume_r10_w90 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r10_w90,
        description="Sales volume within 10 miles, 90-day lookback",
    )
    median_ppsf_r10_w90 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r10_w90,
        description="Median price/sqft within 10 miles, 90-day lookback",
    )

    sales_volume_r10_w180 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r10_w180,
        description="Sales volume within 10 miles, 180-day lookback",
    )
    median_ppsf_r10_w180 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r10_w180,
        description="Median price/sqft within 10 miles, 180-day lookback",
    )

    sales_volume_r10_w365 = Feature(
        dtype="numeric",
        derived_feature_fn=_vol_r10_w365,
        description="Sales volume within 10 miles, 365-day lookback",
    )
    median_ppsf_r10_w365 = Feature(
        dtype="numeric",
        derived_feature_fn=_ppsf_r10_w365,
        description="Median price/sqft within 10 miles, 365-day lookback",
    )

    # ------------------------------------------------------------------
    # Configuration for the velocity computation
    # ------------------------------------------------------------------

    # Maps each (radius, window) to its artifact name and the Feature attrs
    VELOCITY_CONFIGS = [
        (1.0, 30), (1.0, 90), (1.0, 180), (1.0, 365),
        (5.0, 30), (5.0, 90), (5.0, 180), (5.0, 365),
        (10.0, 30), (10.0, 90), (10.0, 180), (10.0, 365),
    ]

    def __init__(self, h3_resolution: int = DEFAULT_H3_RESOLUTION):
        """Initialize with tunable H3 resolution.

        Args:
            h3_resolution: H3 grid resolution (default 8, ~460m edge).
        """
        super().__init__()
        self.h3_resolution = h3_resolution

        # Fitted state
        self._velocity_data: dict[str, pd.DataFrame] = {}
        self._reference_date: Optional[datetime] = None
        self._grid: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fit: compute velocity at every grid cell (expensive)
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "GeoVelocityFeatures":
        """Fit by computing velocity at every H3 grid cell.

        For each declared Feature (each radius × window combination):
          1. Generate H3 grid covering the sales data extent
          2. Run BallTree to compute velocity at each grid cell
          3. Store the precomputed DataFrame

        Args:
            df: Sales DataFrame with: latitude, longitude, sale_date,
                sale_price, total_living_area, source.

        Returns:
            Self for chaining.
        """
        self._reference_date = datetime.utcnow()

        # Generate grid once — shared by all velocity computations
        logger.info("Generating H3 grid (resolution=%d)...", self.h3_resolution)
        self._grid = generate_grid(df, resolution=self.h3_resolution)
        logger.info("Total grid cells: %d", len(self._grid))

        # Compute velocity for each declared (radius, window) combo
        self._velocity_data = {}
        for radius, window in self.VELOCITY_CONFIGS:
            name = artifact_name(radius, window)
            logger.info("  Fitting %s (radius=%.0fmi, window=%dd)...", name, radius, window)

            vel_df = compute_single_velocity(
                grid_points=self._grid,
                sales_df=df,
                reference_date=self._reference_date,
                radius_miles=radius,
                lookback_days=window,
            )
            self._velocity_data[name] = vel_df

            nonzero = int((vel_df["sales_volume"] > 0).sum()) if not vel_df.empty else 0
            logger.info(
                "    %d cells, %d with sales",
                len(vel_df), nonzero,
            )

        self._is_fitted = True
        logger.info(
            "Fit complete: %d Features, %d grid cells",
            len(self._velocity_data) * 2,  # volume + ppsf per combo
            len(self._grid),
        )
        return self

    # ------------------------------------------------------------------
    # Transform: H3 cell lookup (cheap)
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Look up precomputed velocity values for input locations.

        Maps each input row's lat/lng to its H3 cell and joins the
        precomputed velocity values. The output DataFrame has one column
        per declared Feature (sales_volume_r1_w30, median_ppsf_r1_w30, ...).

        Args:
            df: Input DataFrame with 'latitude' and 'longitude' columns.

        Returns:
            DataFrame with one column per declared velocity Feature.
        """
        if not self._is_fitted:
            raise ValueError("GeoVelocityFeatures not fitted. Call fit() first.")

        if "latitude" not in df.columns or "longitude" not in df.columns:
            raise ValueError("Input DataFrame must have 'latitude' and 'longitude' columns.")

        # Map each row to its H3 cell
        h3_cells = df.apply(
            lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], self.h3_resolution)
            if pd.notna(r["latitude"]) and pd.notna(r["longitude"])
            else None,
            axis=1,
        )

        # Build an intermediate df with the velocity columns keyed by _vel_<name>_<metric>
        # so the derived_feature_fn extractors can pull from it
        enriched = df.copy()
        enriched["_h3_index"] = h3_cells

        for name, vel_df in self._velocity_data.items():
            if vel_df.empty:
                # Feature computed but no data — fill NaN
                short = name.replace("geo_velocity_", "")
                enriched[f"_vel_{short}_sales_volume"] = np.nan
                enriched[f"_vel_{short}_median_ppsf"] = np.nan
                continue

            lookup = vel_df.set_index("h3_index")[["sales_volume", "median_ppsf"]]
            short = name.replace("geo_velocity_", "")
            lookup = lookup.rename(columns={
                "sales_volume": f"_vel_{short}_sales_volume",
                "median_ppsf": f"_vel_{short}_median_ppsf",
            })
            enriched = enriched.merge(
                lookup, left_on="_h3_index", right_index=True, how="left",
            )

        # Now run the standard FeatureSet transform which calls each Feature's derived_feature_fn
        # The fns read from the _vel_* columns we just joined
        result = pd.DataFrame(index=df.index)
        for feature in self._features.values():
            if feature.drop:
                continue
            if feature.has_derived_fn:
                result[feature.name] = feature.apply(enriched)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, store) -> None:
        """Save fitted velocity data as independent artifacts.

        Args:
            store: ArtifactStore instance.
        """
        for name, vel_df in self._velocity_data.items():
            store.save(
                name, vel_df,
                artifact_type="geo_velocity",
                tags={
                    "pipeline": "geographic-feature-store",
                    "reference_date": str(self._reference_date.date()) if self._reference_date else "unknown",
                    "grid_resolution": str(self.h3_resolution),
                },
            )
            logger.info("Saved artifact %s: %d rows", name, len(vel_df))

        store.save("grid", self._grid, artifact_type="h3_grid",
                    tags={"resolution": str(self.h3_resolution)})

        config = {
            "h3_resolution": self.h3_resolution,
            "reference_date": str(self._reference_date) if self._reference_date else None,
        }
        store.save("features_config", config, artifact_type="config")

    def load(self, store) -> None:
        """Load fitted velocity data from ArtifactStore.

        Args:
            store: ArtifactStore instance.
        """
        config = store.get("features_config")
        self.h3_resolution = config["h3_resolution"]
        if config.get("reference_date"):
            self._reference_date = datetime.fromisoformat(config["reference_date"])

        self._grid = store.get("grid")

        self._velocity_data = {}
        for radius, window in self.VELOCITY_CONFIGS:
            name = artifact_name(radius, window)
            self._velocity_data[name] = store.get(name)

        self._is_fitted = True
        logger.info("Loaded %d velocity artifacts from ArtifactStore", len(self._velocity_data))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def velocity_data(self) -> dict[str, pd.DataFrame]:
        """Access the fitted velocity DataFrames (one per radius×window)."""
        return self._velocity_data

    @property
    def reference_date(self) -> Optional[datetime]:
        return self._reference_date

    @property
    def grid(self) -> Optional[pd.DataFrame]:
        return self._grid

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_features = len([f for f in self._features.values() if not f.drop])
        n_cells = len(self._grid) if self._grid is not None else 0
        return (
            f"GeoVelocityFeatures("
            f"res={self.h3_resolution}, "
            f"{n_features} features, "
            f"{n_cells} cells, "
            f"{status})"
        )
