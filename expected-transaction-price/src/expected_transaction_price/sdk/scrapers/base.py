import logging
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

class BaseScraper:
    """Base class for composable real estate web scraping.
    
    Enforces atomic extraction methods and declarative data traversal
    for robust conversion of heavily-nested JSON payloads or raw DOM trees
    into perfectly structured unified features.
    """
    
    def __init__(self, url: str):
        self.url = url
        self.html: Optional[str] = None
        self.soup = None
        self.raw_payload: Dict[str, Any] = {}
        
    def fetch_html(self) -> bool:
        """Fetch the HTML source using curl_cffi to mimic browser TLS fingerprints."""
        try:
            from curl_cffi import requests as cffi_requests
            
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            
            # Impersonate Chrome 124 TLS and HTTP/2 Headers natively
            resp = cffi_requests.get(
                self.url,
                headers=headers,
                impersonate="chrome124",
                timeout=15,
            )
            
            if resp.status_code < 400:
                self.html = resp.text
                from bs4 import BeautifulSoup
                self.soup = BeautifulSoup(self.html, 'html.parser')
                return True
            else:
                logger.error(f"Failed to fetch {self.url}. Status: {resp.status_code}")
                # Usually Zillow returns a 403 if Datacenter IP is detected
                return False
        except Exception as e:
            logger.error(f"Network error fetching {self.url}: {e}")
            return False
            
    def _safe_navigate(self, payload: Dict[str, Any], path: List[str], default: Any = None) -> Any:
        """Declaratively navigate a deeply nested dictionary using a path array.
        
        Args:
            payload: The dictionary to search.
            path: Array of consecutive keys, e.g. ["hdpData", "homeInfo", "price"]
            default: The strict default to return if traversal fails midway.
            
        Returns:
            The deeply nested value, or the default if the path is broken.
        """
        current = payload
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
        
    def setup_data_layer(self):
        """Locates and parses the site's injected JSON state blob or sets up alternative parsers.
        
        Subclasses must implement this to populate self.raw_payload.
        """
        raise NotImplementedError
        
    # --- Atomic Extractor Properties (Subclasses MUST implement) ---
    def get_price(self) -> float: raise NotImplementedError
    def get_address(self) -> Optional[str]: return None
    def get_zip(self) -> Optional[str]: return None
    def get_year_built(self) -> Optional[float]: raise NotImplementedError
    def get_total_living_area(self) -> Optional[float]: raise NotImplementedError
    def get_beds(self) -> Optional[int]: raise NotImplementedError
    def get_baths(self) -> Optional[float]: raise NotImplementedError
    def get_lot_acres(self) -> Optional[float]: raise NotImplementedError
    def get_coordinates(self) -> tuple[Optional[float], Optional[float]]: raise NotImplementedError
    def get_building_style(self) -> Optional[str]: raise NotImplementedError
    def get_central_air(self) -> Optional[bool]: raise NotImplementedError
    def get_heat_type(self) -> Optional[str]: raise NotImplementedError
    def get_fuel_type(self) -> Optional[str]: raise NotImplementedError
    def get_basement_type(self) -> Optional[str]: raise NotImplementedError
    def get_exterior_wall(self) -> Optional[str]: raise NotImplementedError
    def get_fireplaces(self) -> Optional[int]: return 0
    def get_stories(self) -> Optional[int]: return None
    def get_assessed_value(self) -> float: return 0.0
    def get_land_value(self) -> float: return 0.0

    def extract_raw_features(self) -> Dict[str, Any]:
        """Orchestrator that calls all atomic properties to build the standardized dictionary."""
        if not self.html:
            self.fetch_html()
            
        self.setup_data_layer()
        
        lat, lon = self.get_coordinates()
        
        return {
            "sale_price": self.get_price(), # Technically we might want this for comparison, not input to model
            "address": self.get_address(),
            "zip_code": self.get_zip(),
            "year_built": self.get_year_built(),
            "total_living_area": self.get_total_living_area(),
            "first_floor_area": None,   # Rarely available strictly on standard listings
            "second_floor_area": None,
            "beds": self.get_beds(),
            "baths": self.get_baths(),
            "half_baths": 0,            # Normalizing usually collapses half baths into baths in scraping mode
            "kitchens": 1,              # Default assumption for single family
            "stories": self.get_stories() or 1,
            "lot_frontage": None,
            "lot_depth": None,
            "lot_acres": self.get_lot_acres(),
            "latitude": lat,
            "longitude": lon,
            "building_style": self.get_building_style() or "Unknown",
            "exterior_wall": self.get_exterior_wall() or "Unknown",
            "heat_type": self.get_heat_type() or "Unknown",
            "central_air": self.get_central_air() or False,
            "basement_type": self.get_basement_type() or "Unknown",
            "school_district": "Unknown", # Will require spatial join at inference time!
            "assessed_value": self.get_assessed_value(),
            "land_value": self.get_land_value(),
        }
