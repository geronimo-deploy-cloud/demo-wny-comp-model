import os
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MLSClient:
    """Client for authenticating and fetching properties from a RESO Web API MLS backend."""
    
    def __init__(self):
        # We design this to securely pull from the environment so credentials never touch the code.
        self.client_id = os.getenv("MLS_CLIENT_ID")
        self.client_secret = os.getenv("MLS_CLIENT_SECRET")
        self.token_url = os.getenv("MLS_TOKEN_URL", "https://api.local-mls.com/oauth2/token")
        self.api_base_url = os.getenv("MLS_API_BASE_URL", "https://api.local-mls.com/reso/odata")
        
        self.access_token: Optional[str] = None

    def _authenticate(self) -> bool:
        """Securely retrieves an OAuth2 Bearer token from the MLS identity provider."""
        if not self.client_id or not self.client_secret:
            logger.warning("MLS Credentials missing! Please configure MLS_CLIENT_ID and MLS_CLIENT_SECRET in your .env")
            return False
            
        try:
            # Standard OAuth 2.0 Client Credentials Flow
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "api"  # Scopes vary by MLS provider
            }
            resp = requests.post(self.token_url, data=payload, timeout=10)
            if resp.status_code == 200:
                self.access_token = resp.json().get("access_token")
                return True
            else:
                logger.error(f"Failed to authenticate with MLS. Code: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"Network error during MLS authentication: {e}")
            return False

    def fetch_property(self, listing_key: str) -> Optional[Dict[str, Any]]:
        """Queries the RESO Web API for a specific Property using its unique ListingKey or ListingId."""
        
        # If we are missing credentials entirely, we provide a mock payload to prove out the pipeline!
        if not self.client_id:
            logger.info("Operating in MOCK mode since MLS credentials are not configured.")
            return self._mock_reso_response()
            
        if not self.access_token:
            if not self._authenticate():
                return None
                
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
        
        # OData querying standard
        query_url = f"{self.api_base_url}/Property('{listing_key}')"
        try:
            resp = requests.get(query_url, headers=headers, timeout=15)
            if resp.status_code == 200:
                return self._normalize_to_schema(resp.json())
            elif resp.status_code == 401:
                logger.warning("MLS Token expired. Invalidating and returning None.")
                self.access_token = None
                return None
            else:
                logger.error(f"MLS fetch failed for {listing_key}. Status: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Network error fetching from MLS: {e}")
            return None

    def _normalize_to_schema(self, reso_data: Dict[str, Any]) -> Dict[str, Any]:
        """Strictly coerces the guaranteed RESO payload into our expected-transaction-price schema."""
        # Using .get protects us, but the RESO Standard generally guarantees keys exist even if null
        
        # Consolidate half baths into baths precisely as our scraper normalization does
        full_baths = float(reso_data.get("BathroomsFull", 0) or 0)
        half_baths = float(reso_data.get("BathroomsHalf", 0) or 0)
        
        lat = reso_data.get("Latitude")
        lon = reso_data.get("Longitude")
        
        return {
            "sale_price": float(reso_data.get("ClosePrice") or reso_data.get("ListPrice") or 0.0),
            "year_built": float(reso_data.get("YearBuilt")) if reso_data.get("YearBuilt") else None,
            "total_living_area": float(reso_data.get("LivingArea")) if reso_data.get("LivingArea") else None,
            "first_floor_area": None,
            "second_floor_area": None,
            "beds": int(reso_data.get("BedroomsTotal")) if reso_data.get("BedroomsTotal") else None,
            "baths": full_baths + (0.5 * half_baths),
            "half_baths": 0,
            "kitchens": 1,
            "stories": int(reso_data.get("Levels", ["1"])[0]) if isinstance(reso_data.get("Levels"), list) else 1,
            "fireplaces": int(reso_data.get("FireplacesTotal") or 0),
            "lot_frontage": None,
            "lot_depth": None,
            "lot_acres": float(reso_data.get("LotSizeAcres")) if reso_data.get("LotSizeAcres") else None,
            "latitude": float(lat) if lat else None,
            "longitude": float(lon) if lon else None,
            "building_style": reso_data.get("ArchitecturalStyle", ["Unknown"])[0] if isinstance(reso_data.get("ArchitecturalStyle"), list) else "Unknown",
            "overall_condition": reso_data.get("PropertyCondition", ["Normal"])[0] if isinstance(reso_data.get("PropertyCondition"), list) else "Normal",
            "exterior_wall": "Unknown",
            "heat_type": reso_data.get("Heating", ["Unknown"])[0] if isinstance(reso_data.get("Heating"), list) else "Unknown",
            "central_air": bool(reso_data.get("Cooling")),
            "fuel_type": "Unknown",
            "basement_type": "Unknown",
            "school_district": "Unknown"
        }
        
    def _mock_reso_response(self) -> Dict[str, Any]:
        """Provides a simulated RESO payload so testing isn't blocked by credential provisioning."""
        return self._normalize_to_schema({
            "ListingKey": "B1245678",
            "ListPrice": 450000.0,
            "YearBuilt": 2005,
            "LivingArea": 2850,
            "BedroomsTotal": 4,
            "BathroomsFull": 2,
            "BathroomsHalf": 1,
            "LotSizeAcres": 0.45,
            "Latitude": 42.8864,
            "Longitude": -78.8784,  # Buffalo generic coordinates
            "ArchitecturalStyle": ["Colonial"],
            "PropertyCondition": ["Good"],
            "Heating": ["Forced Air"],
            "Cooling": ["Central Air"]
        })
