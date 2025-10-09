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
    created_at: datetime

    class Config:
        from_attributes = True

class ProvinceWithStats(Province):
    total_attractions: int
    active_attractions: int
    total_comments: int
    avg_sentiment: float

