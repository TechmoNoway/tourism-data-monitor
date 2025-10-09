from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
from app.schemas.comment import SentimentEnum


class SentimentAnalysisResult(BaseModel):
    sentiment: SentimentEnum
    confidence: float
    aspects: Dict[str, Dict]

class AspectSentiment(BaseModel):
    aspect: str
    sentiment: SentimentEnum
    confidence: float
    mentions: int

class AttractionAnalytics(BaseModel):
    attraction_id: int
    attraction_name: str
    province_name: str
    analysis_date: datetime

    # comment statistics
    total_comments: int
    new_comments_last_week: int
    sentiment_breakdown: Dict[str, int] 
    avg_sentiment: float

    # aspect analysis
    trending_aspects: List[AspectSentiment]
    top_positive_aspects: List[str]
    top_negative_aspects: List[str]

    activity_score: float
    engagement_trend: str
    bot_percentage: float

class AnalysisLogBase(BaseModel):
    attraction_id: int
    total_comments: int
    positive_comments: int
    negative_comments: int
    neutral_comments: int
    bot_comments: int
    avg_sentiment: float
    trending_aspects: Optional[str] = None
    activity_score: float

class AnalysisLogCreate(AnalysisLogBase):
    pass

class AnalysisLog(AnalysisLogBase):
    id: int
    analysis_date: datetime

    class Config:
        from_attributes = True

class TrendAnalysis(BaseModel):
    attraction_id: int
    time_period: str # "7d", "30d", "90d"
    sentiment_trend: List[Dict]  # [{date, positive, negative, neutral}]
    engagement_trend: List[Dict]  # [{date, comments, likes, shares}]
    popular_aspects_trend: Dict[str, List]  # {aspect: [scores over time]}