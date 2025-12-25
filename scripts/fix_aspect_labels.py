"""
Script to re-classify aspect labels for all comments
Fixes comments that have wrong aspect labels
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import SessionLocal
from app.models.comment import Comment
from app.services.topic_classifier import TopicClassifier
from sqlalchemy import func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_aspect_labels():
    """Re-classify aspect labels for all meaningful comments"""
    db = SessionLocal()
    
    try:
        # Initialize classifier
        logger.info("Initializing topic classifier...")
        classifier = TopicClassifier()
        
        # Get all meaningful comments with topics
        logger.info("Fetching comments...")
        comments = db.query(Comment).filter(
            Comment.is_meaningful == True,
            Comment.topics.isnot(None),
            Comment.topics != []
        ).all()
        
        logger.info(f"Found {len(comments)} comments to re-classify")
        
        updated_count = 0
        error_count = 0
        
        for i, comment in enumerate(comments, 1):
            try:
                # Re-classify topics
                new_topics = classifier.classify_topics(comment.content, comment.language or 'vi')
                
                # Only update if topics changed
                if set(new_topics) != set(comment.topics or []):
                    old_topics = comment.topics or []
                    comment.topics = new_topics
                    
                    # Re-calculate aspect sentiments if needed
                    if new_topics and comment.sentiment:
                        new_aspect_sentiments = classifier.get_aspect_sentiments(
                            comment.content,
                            new_topics,
                            comment.sentiment,
                            comment.language or 'vi'
                        )
                        comment.aspect_sentiments = new_aspect_sentiments
                    else:
                        comment.aspect_sentiments = {}
                    
                    updated_count += 1
                    
                    if updated_count % 50 == 0:
                        logger.info(f"Progress: {i}/{len(comments)} processed, {updated_count} updated")
                        db.commit()
                    
                    # Log significant changes
                    if len(old_topics) > 0 and len(new_topics) == 0:
                        logger.warning(f"Comment {comment.id}: Removed all topics. Content: {comment.content[:100]}")
                    elif set(old_topics) != set(new_topics):
                        logger.info(f"Comment {comment.id}: {old_topics} -> {new_topics}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing comment {comment.id}: {e}")
                continue
        
        # Final commit
        db.commit()
        
        logger.info("=" * 60)
        logger.info(f"✓ Re-classification complete!")
        logger.info(f"  Total processed: {len(comments)}")
        logger.info(f"  Updated: {updated_count}")
        logger.info(f"  Errors: {error_count}")
        logger.info(f"  Unchanged: {len(comments) - updated_count - error_count}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Starting aspect label re-classification...")
    fix_aspect_labels()
