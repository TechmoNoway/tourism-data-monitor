from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from app.models.base import Base

class SocialPost(Base):
    __tablename__ = "social_posts"
    __table_args__ = (
        UniqueConstraint('platform', 'platform_post_id', name='uq_platform_post'),
    )

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(50), nullable=False)
    platform_post_id = Column(String(200), nullable=False, index=True)
    post_url = Column(String(1000))
    title = Column(String(300))
    content = Column(Text)
    author = Column(String(200))
    author_id = Column(String(200))
    attraction_id = Column(Integer, ForeignKey("tourist_attractions.id"))
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    post_date = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.now(timezone.utc))
    last_updated = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    # relationships
    attraction = relationship("TouristAttraction", back_populates="posts")
    comments = relationship("Comment", back_populates="post")



