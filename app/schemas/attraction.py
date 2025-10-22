from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from app.schemas.province import Province


class TouristAttractionBase(BaseModel):
    name: str
    category: Optional[str] = None
    address: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[str] = None

class TouristAttractionCreate(TouristAttractionBase):
    province_id: int

class TouristAttractionUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    address: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[str] = None
    rating: Optional[float] = None
    total_reviews: Optional[int] = None
    is_active: Optional[bool] = None

class TouristAttraction(TouristAttractionBase):
    id: int
    province_id: int
    google_place_id: Optional[str] = None
    rating: float
    total_reviews: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TouristAttractionWithProvince(TouristAttraction):
    province: 'Province'

class TouristAttractionWithStats(TouristAttraction):
    province: 'Province'
    recent_comments_count: int
    sentiment_breakdown: dict
    activity_score: float
    trending_aspects: List[str]


