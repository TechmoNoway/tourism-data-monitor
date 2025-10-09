from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("social_posts.id"))
    attraction_id = Column(Integer, ForeignKey("tourist_attractions.id"))
    content = Column(Text)
    author = Column(String(200))
    author_id = Column(String(100))
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    comment_date = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.now(timezone.utc))

    # analysis results
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    confidence_score = Column(Float)
    aspects_detected = Column(Text) # JSON string
    is_bot_suspected = Column(Boolean, default=False)
    bot_confidence_score = Column(Float)

    post = relationship("SocialPost", back_populates="comments")
    attraction = relationship("TouristAttraction", back_populates="comments")

















