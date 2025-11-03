from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from enum import Enum


class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class CommentBase(BaseModel):
    platform: str
    platform_comment_id: str
    content: str
    author: Optional[str] = None
    author_id: Optional[str] = None
    comment_date: Optional[datetime] = None
    like_count: Optional[int] = 0
    reply_count: Optional[int] = 0

class CommentCreate(CommentBase):
    post_id: int
    attraction_id: int
    quality_tier: Optional[str] = None
    quality_score: Optional[float] = None  
    is_meaningful: Optional[bool] = True    

class CommentUpdate(BaseModel):
    content: Optional[str] = None
    like_count: Optional[int] = None
    reply_count: Optional[int] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[SentimentEnum] = None
    confidence_score: Optional[float] = None
    aspects_detected: Optional[str] = None
    is_bot_suspected: Optional[bool] = None
    bot_confidence: Optional[float] = None

    class Config:
        from_attributes = True

class Comment(CommentBase):
    id: int
    post_id: int
    attraction_id: int
    scraped_at: datetime
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    confidence_score: Optional[float] = None
    
    class Config:
        from_attributes = True

class CommentWithAnalysis(Comment):
    aspect_breakdown: Optional[dict] = None
    sentiment_explanation: Optional[str] = None

