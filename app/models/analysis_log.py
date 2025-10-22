from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Text
from app.models.base import Base


class AnalysisLog(Base):
    __tablename__ = "analysis_logs"

    id = Column(Integer, primary_key=True, index=True)
    attraction_id = Column(Integer, ForeignKey("tourist_attractions.id"))
    analysis_date = Column(DateTime, default=datetime.now(timezone.utc))
    total_comments = Column(Integer, default=0)
    positive_comments = Column(Integer, default=0)
    negative_comments = Column(Integer, default=0)
    neutral_comments = Column(Integer, default=0)
    bot_comments = Column(Integer, default=0)
    avg_sentiment = Column(Float, default=0.0)
    trending_aspects = Column(Text) # JSON string
    activity_score = Column(Float, default=0.0)





