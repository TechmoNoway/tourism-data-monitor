from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from app.models.base import Base


class TouristAttraction(Base):
    __tablename__ = "tourist_attractions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100))
    address = Column(String(500))
    province_id = Column(Integer, ForeignKey("provinces.id"))
    description = Column(Text)
    keywords = Column(Text) #json string of keywords
    google_place_id = Column(String(200))
    rating = Column(Float, default=0.0)
    total_reviews = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    province = relationship("Province", back_populates="attractions")
    posts = relationship("SocialPost", back_populates="attraction")
    comments = relationship("Comment", back_populates="attraction")