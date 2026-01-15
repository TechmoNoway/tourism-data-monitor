from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DemandIndexBase(BaseModel):
    attraction_id: int
    province_id: int
    period_start: datetime
    period_end: datetime
    period_type: str
    overall_index: float
    comment_volume_score: float
    sentiment_score: float
    engagement_score: float
    growth_score: float
    total_comments: int
    positive_rate: float
    negative_rate: float
    neutral_rate: float
    avg_sentiment: float
    total_engagement: int
    growth_rate: float
    rank_in_province: Optional[int] = None
    rank_national: Optional[int] = None


class DemandIndexResponse(DemandIndexBase):
    id: int
    calculated_at: datetime
    attraction_name: Optional[str] = None
    province_name: Optional[str] = None
    
    class Config:
        from_attributes = True


class ProvinceDemandIndexResponse(BaseModel):
    id: int
    province_id: int
    province_name: str
    period_start: datetime
    period_end: datetime
    period_type: str
    overall_index: float
    comment_volume_score: float
    sentiment_score: float
    engagement_score: float
    growth_score: float
    total_comments: int
    total_attractions: int
    active_attractions: int
    positive_rate: float
    avg_sentiment: float
    growth_rate: float
    rank_national: Optional[int] = None
    calculated_at: datetime
    
    class Config:
        from_attributes = True


class TopAttraction(BaseModel):
    id: int
    name: str
    province_name: str
    overall_index: float
    total_comments: int
    positive_rate: float
    growth_rate: float
    rank: int


class TopProvince(BaseModel):
    id: int
    name: str
    overall_index: float
    total_comments: int
    active_attractions: int
    positive_rate: float
    growth_rate: float
    rank: int


class DemandTrendPoint(BaseModel):
    date: datetime
    overall_index: float
    comment_volume: int
    sentiment_score: float


class DemandAnalytics(BaseModel):
    current_index: float
    previous_index: float
    change_percentage: float
    trend: str
    total_comments: int
    positive_rate: float
    negative_rate: float
    neutral_rate: float
    top_topics: List[dict]
    trend_data: List[DemandTrendPoint]


class ComparativeAnalysis(BaseModel):
    attraction_id: int
    attraction_name: str
    province_name: str
    current_index: float
    rank_in_province: int
    rank_national: int
    compared_to_avg_province: float
    compared_to_avg_national: float
    performance: str
