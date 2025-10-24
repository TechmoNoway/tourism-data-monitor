from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.models.base import Base

class Comment(Base):
    __tablename__ = "comments"
    __table_args__ = (
        UniqueConstraint('platform', 'platform_comment_id', name='uq_platform_comment'),
    )

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(50), nullable=False)
    platform_comment_id = Column(String(200), nullable=False, index=True)
    post_id = Column(Integer, ForeignKey("social_posts.id"))
    attraction_id = Column(Integer, ForeignKey("tourist_attractions.id"))
    content = Column(Text)
    author = Column(String(200))
    author_id = Column(String(100))
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    comment_date = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.now(timezone.utc))

    # Data cleaning fields
    cleaned_content = Column(Text)
    is_valid = Column(Boolean, default=True)
    language = Column(String(10))  # 'vi', 'en', 'zh-cn', 'ko', 'ja', 'th', 'unknown'
    word_count = Column(Integer)
    
    # Sentiment analysis fields
    sentiment = Column(String(20))  # 'positive', 'neutral', 'negative'
    sentiment_score = Column(Float)
    analysis_model = Column(String(50))  # 'phobert', 'xlm-roberta', 'rule-based'
    analyzed_at = Column(DateTime)
    
    # Topic classification fields
    topics = Column(JSONB)  # ['scenery', 'food', 'service', 'pricing', 'accessibility']
    aspect_sentiments = Column(JSONB)  # {'scenery': 'positive', 'food': 'neutral', 'service': 'negative'}
    
    # Quality/spam detection fields
    is_spam = Column(Boolean, default=False)  # Detected as spam/bot
    spam_score = Column(Float)

    post = relationship("SocialPost", back_populates="comments")
    attraction = relationship("TouristAttraction", back_populates="comments")

















