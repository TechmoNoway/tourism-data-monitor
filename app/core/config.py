from typing import ClassVar, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator

try:
    from app.core.facebook_best_pages import FACEBOOK_BEST_PAGES as FB_BEST_PAGES_FULL
except ImportError:
    FB_BEST_PAGES_FULL = {}


class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "Tourism Data Monitor API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 4242

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/tourism_db"

    # API Credentials
    YOUTUBE_API_KEY: Optional[str] = None
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    
    # Apify (for Facebook, TikTok, and Google Maps scraping)
    APIFY_API_TOKEN: Optional[str] = None

    # Collection Settings
    DEFAULT_POSTS_LIMIT: int = 50
    DEFAULT_COMMENTS_LIMIT: int = 100
    MAX_COLLECTION_RETRIES: int = 3

    # Real-time Collection Configuration (Weekly Updates)
    # Adjust these values to control cost during demo
    # Full collection: TARGET_COMMENTS=200, Weekly update: TARGET_COMMENTS=10-20
    TARGET_COMMENTS_PER_ATTRACTION: int = 10  # Weekly incremental update (demo mode)
    TARGET_POSTS_PER_ATTRACTION: int = 5      # Weekly incremental update (demo mode)
    MAX_ATTRACTIONS_PER_PROVINCE: int = 5     # Limit attractions during demo
    
    # Platform-specific limits for weekly updates
    PLATFORM_LIMITS: ClassVar[dict] = {
        'facebook': 10,      # Demo: 10 comments per platform
        'tiktok': 10,        # Demo: 10 comments per platform
        'google_maps': 10,   # Demo: 10 comments per platform
        'youtube': 5         # Demo: 5 comments (lower quality)
    }
    
    # For full data collection (disable during demo)
    FULL_COLLECTION_MODE: bool = False
    FULL_TARGET_COMMENTS: int = 200  # Used when FULL_COLLECTION_MODE=True
    FULL_TARGET_POSTS: int = 50      # Used when FULL_COLLECTION_MODE=True
    FULL_PLATFORM_LIMITS: ClassVar[dict] = {
        'facebook': 120,
        'tiktok': 120,
        'google_maps': 100,
        'youtube': 50
    }

    # Facebook Collection Settings
    FACEBOOK_COLLECTION_ENABLED: bool = True
    FACEBOOK_POSTS_PER_LOCATION: int = 20
    FACEBOOK_COMMENTS_PER_POST: int = 150  
    FACEBOOK_USE_BEST_PAGES: bool = False
    
    FACEBOOK_BEST_PAGES: ClassVar[dict] = FB_BEST_PAGES_FULL

    # Scheduler Configuration
    # Option 1 (USE_SCHEDULER_SERVICE=False): OS-based scheduling (Cron on Linux, Task Scheduler on Windows)
    # Option 2 (USE_SCHEDULER_SERVICE=True): Python Scheduler (runs as service/background process)
    USE_SCHEDULER_SERVICE: bool = False  # Set True for Python Scheduler, False for OS-based scheduling
    
    SCHEDULER_ENABLED: bool = False
    DAILY_COLLECTION_HOUR: int = 2
    DAILY_COLLECTION_MINUTE: int = 0

    CORS_ORIGINS: list[str] = []
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        elif isinstance(v, list):
            return v
        return []

    PROVINCES: ClassVar[dict] = {
        "lam_dong": {"name": "Lâm Đồng", "code": "LD", "city": "Đà Lạt"},
        "da_nang": {"name": "Đà Nẵng", "code": "DN", "city": "Đà Nẵng"},
        "binh_thuan": {"name": "Bình Thuận", "code": "BT", "city": "Phan Thiết"},
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore" 

settings = Settings()


