from typing import ClassVar, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "Tourism Data Monitor API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite:///./tourism.db"

    # API Credentials - YouTube
    YOUTUBE_API_KEY: Optional[str] = None

    # API Credentials - Google Maps/Places
    GOOGLE_MAPS_API_KEY: Optional[str] = None

    # API Credentials - TikTok
    TIKTOK_CLIENT_KEY: Optional[str] = None
    TIKTOK_CLIENT_SECRET: Optional[str] = None
    TIKTOK_ACCESS_TOKEN: Optional[str] = None

    # API Credentials - Facebook (temporarily disabled)
    # FACEBOOK_APP_ID: Optional[str] = None
    # FACEBOOK_APP_SECRET: Optional[str] = None
    # FACEBOOK_ACCESS_TOKEN: Optional[str] = None

    # Apify (for Facebook & TikTok scraping)
    APIFY_API_TOKEN: Optional[str] = None

    # RapidAPI (alternative to Apify for Facebook & TikTok)
    RAPIDAPI_KEY: Optional[str] = None

    # Collection Settings
    DEFAULT_POSTS_LIMIT: int = 50
    DEFAULT_COMMENTS_LIMIT: int = 100
    MAX_COLLECTION_RETRIES: int = 3

    # Facebook Collection Settings (validated 2025-10-21)
    FACEBOOK_COLLECTION_ENABLED: bool = True
    FACEBOOK_POSTS_PER_LOCATION: int = 20
    FACEBOOK_COMMENTS_PER_POST: int = 50
    FACEBOOK_USE_BEST_PAGES: bool = True  # Use validated high-engagement pages
    
    # Facebook Best Pages (from Smart Page Selection validation)
    FACEBOOK_BEST_PAGES: ClassVar[dict] = {
        "ba_na_hills": {
            "name": "Sun World Bà Nà Hills",
            "url": "https://www.facebook.com/SunWorldBaNaHills",
            "expected_comments_per_post": 6.1,
            "validated": "2025-10-21"
        },
        "da_lat": {
            "name": "Hồ Tuyền Lâm",
            "url": "https://www.facebook.com/TuyenLamLake",
            "expected_comments_per_post": 11.2,
            "validated": "2025-10-21"
        },
        "phu_quoc": {
            "name": "Phú Quốc Island",
            "url": "https://www.facebook.com/PhuQuocIsland",
            "expected_comments_per_post": 3.7,
            "validated": "2025-10-21"
        }
    }

    # Scheduler Settings
    SCHEDULER_ENABLED: bool = False
    DAILY_COLLECTION_HOUR: int = 2
    DAILY_COLLECTION_MINUTE: int = 0

    # CORS
    CORS_ORIGINS: list[str] = []
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        elif isinstance(v, list):
            return v
        return []

    # Provinces
    PROVINCES: ClassVar[dict] = {
        "lam_dong": {"name": "Lâm Đồng", "code": "LD", "city": "Đà Lạt"},
        "da_nang": {"name": "Đà Nẵng", "code": "DN", "city": "Đà Nẵng"},
        "binh_thuan": {"name": "Bình Thuận", "code": "BT", "city": "Phan Thiết"},
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env

settings = Settings()


