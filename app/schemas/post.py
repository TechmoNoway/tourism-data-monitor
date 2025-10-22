from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel

class PlatformEnum(str, Enum):
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    GOOGLE_REVIEW = "google_review"
    GOOGLE_MAPS = "google_maps"

class SocialPostBase(BaseModel):
    platform: PlatformEnum
    platform_post_id: str
    post_url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    author_id: Optional[str] = None

class SocialPostCreate(SocialPostBase):
    attraction_id: int
    post_date: Optional[datetime] = None

class SocialPostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    share_count: Optional[int] = None
    last_updated: Optional[datetime] = None

class SocialPost(SocialPostBase):
    id: int
    attraction_id: int
    view_count: int
    like_count: int
    comment_count: int
    share_count: int
    post_date: Optional[datetime] = None
    scraped_at: datetime
    last_updated: datetime


    class Config:
        from_attributes = True