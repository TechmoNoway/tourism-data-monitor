from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.models.base import Base


class DemandIndex(Base):
    __tablename__ = "demand_indexes"

    id = Column(Integer, primary_key=True, index=True)
    attraction_id = Column(Integer, ForeignKey("tourist_attractions.id", ondelete="CASCADE"), nullable=False)
    province_id = Column(Integer, ForeignKey("provinces.id", ondelete="CASCADE"), nullable=False)
    
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    period_type = Column(String(20), nullable=False)
    
    overall_index = Column(Float, default=0.0)
    comment_volume_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    growth_score = Column(Float, default=0.0)
    
    total_comments = Column(Integer, default=0)
    positive_rate = Column(Float, default=0.0)
    negative_rate = Column(Float, default=0.0)
    neutral_rate = Column(Float, default=0.0)
    
    avg_sentiment = Column(Float, default=0.0)
    total_engagement = Column(Integer, default=0)
    
    growth_rate = Column(Float, default=0.0)
    
    rank_in_province = Column(Integer)
    rank_national = Column(Integer)
    
    calculated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    attraction = relationship("TouristAttraction", back_populates="demand_indexes")
    province = relationship("Province")

    __table_args__ = (
        Index('idx_demand_attraction_period', 'attraction_id', 'period_start', 'period_end'),
        Index('idx_demand_province_period', 'province_id', 'period_start', 'period_end'),
        Index('idx_demand_overall', 'overall_index'),
        Index('idx_demand_calculated', 'calculated_at'),
        Index('idx_demand_period_type', 'period_type'),
    )


class ProvinceDemandIndex(Base):
    __tablename__ = "province_demand_indexes"

    id = Column(Integer, primary_key=True, index=True)
    province_id = Column(Integer, ForeignKey("provinces.id", ondelete="CASCADE"), nullable=False)
    
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    period_type = Column(String(20), nullable=False)
    
    overall_index = Column(Float, default=0.0)
    comment_volume_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    growth_score = Column(Float, default=0.0)
    
    total_comments = Column(Integer, default=0)
    total_attractions = Column(Integer, default=0)
    active_attractions = Column(Integer, default=0)
    
    positive_rate = Column(Float, default=0.0)
    avg_sentiment = Column(Float, default=0.0)
    
    growth_rate = Column(Float, default=0.0)
    rank_national = Column(Integer)
    
    calculated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    province = relationship("Province")

    __table_args__ = (
        Index('idx_prov_demand_period', 'province_id', 'period_start', 'period_end'),
        Index('idx_prov_demand_overall', 'overall_index'),
        Index('idx_prov_demand_calculated', 'calculated_at'),
    )
