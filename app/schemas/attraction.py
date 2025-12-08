from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from app.schemas.province import Province


class TouristAttractionBase(BaseModel):
    name: str
    category: Optional[str] = None
    tourism_type: Optional[str] = None  # beach, mountain, historical, cultural, nature, urban, adventure
    address: Optional[str] = None
    description: Optional[str] = None

class TouristAttractionCreate(TouristAttractionBase):
    province_id: int

class TouristAttractionUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    tourism_type: Optional[str] = None
    address: Optional[str] = None
    description: Optional[str] = None
    total_reviews: Optional[int] = None
    is_active: Optional[bool] = None

class TouristAttraction(TouristAttractionBase):
    id: int
    province_id: int
    google_place_id: Optional[str] = None
    total_reviews: int  # From Google Maps/external sources
    total_comments: int  # From social platforms
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TouristAttractionWithProvince(TouristAttraction):
    province: Province

class TouristAttractionWithStats(TouristAttraction):
    province: Province
    recent_comments_count: int
    sentiment_breakdown: dict
    activity_score: float
    trending_aspects: List[str]

class AspectSentiment(BaseModel):
    aspect: str
    total_mentions: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_percentage: float
    sentiment_score: float

class AttractionDetailStats(BaseModel):
    attraction: TouristAttractionWithProvince
    
    total_comments_6months: int
    meaningful_comments_6months: int
    
    overall_sentiment: dict 
    
    aspects: List[AspectSentiment]
    
    top_positive_comments: List[dict]
    top_negative_comments: List[dict]
    
    class Config:
        from_attributes = True



