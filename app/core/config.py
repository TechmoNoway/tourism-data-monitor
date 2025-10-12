from typing import ClassVar
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Tourism Data Monitor API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str = "sqlite:///./tourism.db"

    PROVINCES: ClassVar[dict] = {
        "lam_dong": {"name": "Lâm Đồng", "code": "LD", "city": "Đà Lạt"},
        "da_nang": {"name": "Đà Nẵng", "code": "DN", "city": "Đà Nẵng"},
        "binh_thuan": {"name": "Bình Thuận", "code": "BT", "city": "Phan Thiết"},
    }
    
    class Config:
        env_file = ".env"

settings = Settings()


