from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ProvinceBase(BaseModel):
    name: str
    code: str
    main_city: Optional[str] = None

class ProvinceCreate(ProvinceBase):
    pass

class ProvinceUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    main_city: Optional[str] = None

class Province(ProvinceBase):
    id: int
    total_attractions: int = 0
    total_comments: int = 0
    total_posts: int = 0
    created_at: datetime

    class Config:
        from_attributes = True

class ProvinceWithStats(Province):
    active_attractions: int
    avg_sentiment: float
    tourism_types_breakdown: dict  # Count of each tourism type

