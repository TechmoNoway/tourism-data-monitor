"""
Comprehensive analysis service to ensure all comments are analyzed
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.comment import Comment
from app.models.tourist_attraction import TouristAttraction
from app.services.sentiment_analyzer import MultilingualSentimentAnalyzer
from app.services.topic_classifier import TopicClassifier

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service to ensure all comments are analyzed with:
    - Language detection
    - Sentiment analysis
    - Topic classification
    - Aspect-level sentiments
    """
    
    def __init__(self, use_gpu: bool = False):
        self.sentiment_analyzer = MultilingualSentimentAnalyzer(use_gpu=use_gpu)
        self.topic_classifier = TopicClassifier()
        self.logger = logging.getLogger("analysis_service")
        
        self.logger.info(f"Analysis Service initialized (GPU: {use_gpu})")
    
    def analyze_unanalyzed_comments(
        self,
        db: Session,
        attraction_id: Optional[int] = None,
        batch_size: int = 50,
        limit: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Analyze all unanalyzed comments
        
        Args:
            db: Database session
            attraction_id: Optional filter by attraction
            batch_size: Number of comments to process at once
            limit: Maximum number to process (None = all)
        
        Returns:
            Dict with analysis statistics
        """
        # Query unanalyzed comments
        query = db.query(Comment).filter(Comment.sentiment.is_(None))
        
        if attraction_id:
            query = query.filter(Comment.attraction_id == attraction_id)
        
        if limit:
            query = query.limit(limit)
        
        comments = query.all()
        
        if not comments:
            self.logger.info("No unanalyzed comments found")
            return {
                'total': 0,
                'analyzed': 0,
                'meaningful': 0,
                'errors': 0
            }
        
        self.logger.info(f"Analyzing {len(comments)} comments...")
        
        stats = {
            'total': len(comments),
            'analyzed': 0,
            'meaningful': 0,
            'errors': 0,
            'languages': {}
        }
        
        # Process in batches
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            
            for comment in batch:
                try:
                    # Sentiment analysis
                    result = self.sentiment_analyzer.analyze_sentiment(comment.content)
                    
                    comment.cleaned_content = result['cleaned_content']
                    comment.language = result['language']
                    comment.word_count = result['word_count']
                    comment.sentiment = result['sentiment']
                    comment.sentiment_score = result['sentiment_score']
                    comment.analysis_model = result['analysis_model']
                    comment.analyzed_at = datetime.now()
                    comment.is_valid = result['word_count'] >= 3
                    
                    # Mark as meaningful if valid
                    if comment.is_valid and comment.sentiment in ['positive', 'neutral', 'negative']:
                        comment.is_meaningful = True
                        stats['meaningful'] += 1
                    
                    # Topic classification
                    topics = self.topic_classifier.classify_topics(comment.content, comment.language)
                    comment.topics = topics if topics else None
                    
                    # Aspect-level sentiments
                    if topics:
                        aspect_sentiments = self.topic_classifier.get_aspect_sentiments(
                            comment.content,
                            topics,
                            comment.sentiment,
                            comment.language
                        )
                        comment.aspect_sentiments = aspect_sentiments
                    
                    stats['analyzed'] += 1
                    
                    # Track languages
                    lang = comment.language or 'unknown'
                    stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing comment {comment.id}: {e}")
                    stats['errors'] += 1
            
            # Commit batch
            db.commit()
            
            if (i + batch_size) % 100 == 0:
                self.logger.info(f"Progress: {i + batch_size}/{len(comments)} comments")
        
        self.logger.info(f"✓ Analysis complete: {stats['analyzed']} analyzed, {stats['meaningful']} meaningful")
        return stats
    
    def update_attraction_statistics(
        self,
        db: Session,
        attraction_id: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Update statistics for attractions
        
        Args:
            db: Database session
            attraction_id: Optional specific attraction (None = all)
        
        Returns:
            Dict with update statistics
        """
        query = db.query(TouristAttraction)
        
        if attraction_id:
            query = query.filter(TouristAttraction.id == attraction_id)
        
        attractions = query.all()
        
        updated = 0
        
        for attraction in attractions:
            # Count meaningful comments
            meaningful_count = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attraction.id,
                Comment.is_meaningful == True
            ).scalar()
            
            # Update if changed
            if attraction.total_comments != meaningful_count:
                attraction.total_comments = meaningful_count
                attraction.total_reviews = meaningful_count
                attraction.updated_at = datetime.now()
                updated += 1
        
        db.commit()
        
        self.logger.info(f"✓ Updated {updated} attraction statistics")
        
        return {
            'attractions_checked': len(attractions),
            'attractions_updated': updated
        }
    
    def get_analysis_status(self, db: Session) -> Dict[str, any]:
        """Get current analysis status"""
        total_comments = db.query(func.count(Comment.id)).scalar()
        analyzed_comments = db.query(func.count(Comment.id)).filter(
            Comment.sentiment.isnot(None)
        ).scalar()
        unanalyzed_comments = total_comments - analyzed_comments
        
        meaningful_comments = db.query(func.count(Comment.id)).filter(
            Comment.is_meaningful == True
        ).scalar()
        
        # Language distribution
        languages = db.query(
            Comment.language,
            func.count(Comment.id)
        ).filter(
            Comment.language.isnot(None)
        ).group_by(Comment.language).all()
        
        return {
            'total_comments': total_comments,
            'analyzed': analyzed_comments,
            'unanalyzed': unanalyzed_comments,
            'meaningful': meaningful_comments,
            'analysis_percentage': round(analyzed_comments / total_comments * 100, 2) if total_comments > 0 else 0,
            'languages': {lang: count for lang, count in languages}
        }


def create_analysis_service(use_gpu: bool = False) -> AnalysisService:
    """Factory function to create analysis service"""
    return AnalysisService(use_gpu=use_gpu)
