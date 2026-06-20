import json
import logging
import re
from typing import Dict, Any, Optional

from .base import BaseScraper

logger = logging.getLogger(__name__)


class ZillowScraper(BaseScraper):
    """Specific extractor for Zillow property URLs.
    
    Exploits the `__NEXT_DATA__` JSON blob embedded in Next.js rendered HTML
    to securely extract data vectors without CSS brittleness.
    
    Zillow's SSR payload provides a limited `resoFacts` subset. Fields like
    heating, cooling, basement, and exterior wall are loaded by the frontend
    via a PerimeterX-protected GraphQL call. When structured data is missing,
    this scraper falls back to keyword extraction from the listing description.
    """
    
    def setup_data_layer(self):
        """Locates the Next.js JSON blob inside Zillow's HTML source."""
        self.raw_payload = {}
        if not self.soup:
            return

        script_tag = self.soup.find('script', id='__NEXT_DATA__')
        if script_tag and script_tag.string:
            try:
                full_json = json.loads(script_tag.string)

                query_state_str = self._safe_navigate(full_json, ["props", "pageProps", "componentProps", "gdpClientCache"])
                if query_state_str:
                    try:
                        # Zillow stringifies the gdpClientCache payload inside the Next JSON
                        if isinstance(query_state_str, str):
                            query_state = json.loads(query_state_str)
                        else:
                            query_state = query_state_str

                        # gdpClientCache holds multiple cached GraphQL queries keyed by
                        # query signature (e.g. VoyagerQuery, ForSalePriorityQuery). Only
                        # some carry the property block, and the order isn't guaranteed —
                        # pick the first entry that actually contains "property".
                        for entry in query_state.values():
                            if isinstance(entry, dict) and entry.get("property"):
                                self.raw_payload = entry["property"]
                                break
                    except Exception:
                        pass
                else:
                    self.raw_payload = full_json
            except json.JSONDecodeError:
                logger.error("Failed to parse Zillow NEXT_DATA JSON.")

    # ------------------------------------------------------------------
    # Description-based fallback extraction
    # ------------------------------------------------------------------
    def _get_description(self) -> str:
        """Return the listing description in lowercase for keyword matching."""
        desc = self._safe_navigate(self.raw_payload, ["description"]) or ""
        return desc.lower()

    def _match_description(self, patterns: list[str]) -> Optional[str]:
        """Return the first pattern that matches the listing description.
        
        Args:
            patterns: List of (regex_pattern, return_value) tuples
        """
        desc = self._get_description()
        for pattern, value in patterns:
            if re.search(pattern, desc):
                return value
        return None
                
    # --- Declarative Traversal Mapping ---
    def get_price(self) -> float:
        val = self._safe_navigate(self.raw_payload, ["price"])
        if val is not None:
            return float(val)
        return 0.0

    def get_address(self) -> Optional[str]:
        # Prefer the structured address subobject; fall back to top-level field.
        street = self._safe_navigate(self.raw_payload, ["address", "streetAddress"])
        if not street:
            street = self._safe_navigate(self.raw_payload, ["streetAddress"])
        return str(street) if street else None

    def get_zip(self) -> Optional[str]:
        zipcode = self._safe_navigate(self.raw_payload, ["address", "zipcode"])
        if not zipcode:
            zipcode = self._safe_navigate(self.raw_payload, ["zipcode"])
        return str(zipcode) if zipcode else None

    def get_year_built(self) -> Optional[float]:
        # Check resoFacts first (reliable), then top-level
        val = self._safe_navigate(self.raw_payload, ["resoFacts", "yearBuilt"])
        if val is None:
            val = self._safe_navigate(self.raw_payload, ["yearBuilt"])
        return float(val) if val else None

    def get_total_living_area(self) -> Optional[float]:
        val = self._safe_navigate(self.raw_payload, ["livingArea"])
        if val is None:
            val = self._safe_navigate(self.raw_payload, ["livingAreaValue"])
        return float(val) if val else None

    def get_beds(self) -> Optional[int]:
        val = self._safe_navigate(self.raw_payload, ["bedrooms"])
        return int(val) if val else None

    def get_baths(self) -> Optional[float]:
        val = self._safe_navigate(self.raw_payload, ["bathrooms"])
        return float(val) if val else None

    def get_lot_acres(self) -> Optional[float]:
        # Prefer lotAreaValue + lotAreaUnits for accuracy
        area_val = self._safe_navigate(self.raw_payload, ["lotAreaValue"])
        area_units = self._safe_navigate(self.raw_payload, ["lotAreaUnits"])
        if area_val is not None:
            v = float(area_val)
            if area_units and area_units.lower() == "acres":
                return v
            else:
                # Assume sqft
                return v / 43560.0

        # Fallback to lotSize (reported in sqft as an integer)
        val = self._safe_navigate(self.raw_payload, ["lotSize"])
        if val:
            v = float(val)
            if v > 100:  # sqft
                return v / 43560.0
            return v
        return None

    def get_coordinates(self) -> tuple[Optional[float], Optional[float]]:
        lat = self._safe_navigate(self.raw_payload, ["latitude"])
        lon = self._safe_navigate(self.raw_payload, ["longitude"])
        return (float(lat) if lat else None, float(lon) if lon else None)

    def get_building_style(self) -> Optional[str]:
        val = self._safe_navigate(self.raw_payload, ["homeType"])
        return str(val).replace("_", " ").capitalize() if val else "Unknown"

    def get_central_air(self) -> Optional[bool]:
        # Structured data first
        has_cooling = self._safe_navigate(self.raw_payload, ["resoFacts", "hasCooling"])
        if has_cooling is not None:
            return bool(has_cooling)
        cooling_list = self._safe_navigate(self.raw_payload, ["resoFacts", "cooling"])
        if cooling_list:
            return True
        # Description fallback
        desc = self._get_description()
        if re.search(r"central\s*(a/?c|air)", desc):
            return True
        if re.search(r"\bA/C\b", self._safe_navigate(self.raw_payload, ["description"]) or ""):
            return True
        return False
        
    def get_heat_type(self) -> Optional[str]:
        # Structured data first
        val = self._safe_navigate(self.raw_payload, ["resoFacts", "heating"])
        if isinstance(val, list) and len(val) > 0:
            return str(val[0]).capitalize()
        if isinstance(val, str) and val:
            return val.capitalize()
        # Description fallback — look for common heating type keywords
        result = self._match_description([
            (r"forced\s*air", "Forced Air"),
            (r"radiant\s*heat", "Radiant"),
            (r"baseboard", "Baseboard"),
            (r"steam\s*heat", "Steam"),
            (r"hot\s*water\s*heat", "Hot Water"),
            (r"heat\s*pump", "Heat Pump"),
            (r"furnace", "Forced Air"),
            (r"boiler", "Hot Water"),
        ])
        return result
        
    def get_basement_type(self) -> Optional[str]:
        # Structured data first
        val = self._safe_navigate(self.raw_payload, ["resoFacts", "basement"])
        if isinstance(val, list) and len(val) > 0:
            return str(val[0]).capitalize()
        if isinstance(val, str) and val:
            return val.capitalize()
        has_basement = self._safe_navigate(self.raw_payload, ["resoFacts", "hasBasement"])
        if has_basement:
            return "Unfinished"
        # Description fallback
        result = self._match_description([
            (r"finished\s*basement", "Finished"),
            (r"full\s*basement", "Full"),
            (r"partial\s*basement", "Partial"),
            (r"walk[\s-]*out\s*basement", "Walk Out"),
            (r"basement", "Unfinished"),
        ])
        return result
        
    def get_exterior_wall(self) -> Optional[str]:
        # Structured data first
        val = self._safe_navigate(self.raw_payload, ["resoFacts", "exteriorFeatures"])
        if isinstance(val, list) and len(val) > 0:
            return str(val[0]).capitalize()
        if isinstance(val, str) and val:
            return val.capitalize()
        # Description fallback
        result = self._match_description([
            (r"vinyl\s*siding", "Vinyl"),
            (r"aluminum\s*siding", "Aluminum"),
            (r"wood\s*siding", "Wood"),
            (r"brick\s*(exterior|siding|veneer|home|house)", "Brick"),
            (r"stucco", "Stucco"),
            (r"fiber\s*cement|hardie", "Fiber Cement"),
            (r"stone\s*(exterior|siding|veneer)", "Stone"),
            (r"cedar\s*(shake|shingle|siding)", "Cedar"),
        ])
        return result

    def get_stories(self) -> Optional[int]:
        """Extract story count from resoFacts or description."""
        val = self._safe_navigate(self.raw_payload, ["resoFacts", "stories"])
        if val is not None:
            return int(val)
        desc = self._get_description()
        if re.search(r"ranch|one.level|single.level|one.story", desc):
            return 1
        if re.search(r"two.story|2.story|colonial|bi.level", desc):
            return 2
        if re.search(r"three.story|3.story|tri.level", desc):
            return 3
        return None

    def get_assessed_value(self) -> float:
        # taxHistory[] holds Zillow's tax-assessed value per year.
        # The most recent entry's `value` is what shows on the listing page.
        tax_history = self._safe_navigate(self.raw_payload, ["taxHistory"])
        if isinstance(tax_history, list) and tax_history:
            for entry in sorted(tax_history, key=lambda e: e.get("time") or 0, reverse=True):
                val = entry.get("value")
                if val:
                    return float(val)
        # Legacy SSR path (rarely populated)
        val = self._safe_navigate(self.raw_payload, ["resoFacts", "taxAssessedValue"])
        if val is not None:
            return float(val)
        return 0.0

    def get_land_value(self) -> float:
        for path in (
            ["resoFacts", "landValue"],
            ["resoFacts", "landAssessedValue"],
            ["landValue"],
        ):
            val = self._safe_navigate(self.raw_payload, path)
            if val:
                return float(val)
        # Heuristic fallback: 20% of total assessed
        assessed = self.get_assessed_value()
        return assessed * 0.2 if assessed > 0 else 0.0
