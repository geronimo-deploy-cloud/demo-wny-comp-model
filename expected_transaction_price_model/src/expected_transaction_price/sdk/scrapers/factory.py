from typing import Optional
from urllib.parse import urlparse

from .base import BaseScraper
from .zillow import ZillowScraper

class ScraperFactory:
    """Routes incoming property URLs to the appropriate concrete scraper."""
    
    @staticmethod
    def get_scraper(url: str) -> Optional[BaseScraper]:
        domain = urlparse(url).netloc.lower()
        
        # Strip www for cleaner matching
        if domain.startswith("www."):
            domain = domain[4:]
            
        if "zillow.com" in domain:
            return ZillowScraper(url)
        # We can add Realtor.com, Trulia, etc as we build them out
        # elif "realtor.com" in domain:
        #     return RealtorScraper(url)
            
        return None
