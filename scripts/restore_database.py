import sys
from pathlib import Path
import logging
from sqlalchemy import func
from app.database.connection import SessionLocal
from app.models.comment import Comment
from app.collectors.comment_filter import CommentFilter
from app.services.topic_classifier import TopicClassifier
from app.services.sentiment_analyzer import MultilingualSentimentAnalyzer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def restore_comment(comment: Comment, filter: CommentFilter, classifier: TopicClassifier, 
                    sentiment_analyzer: MultilingualSentimentAnalyzer) -> bool:
    """Restore a single comment with original NLP"""
    try:
        text = comment.content
        language = comment.language or 'vi'
        
        # 1. Re-assess quality with ORIGINAL code
        tier, score = filter.assess_comment_quality(text)
        comment.quality_tier = tier
        comment.quality_score = score
        
        # 2. Re-classify topics with ORIGINAL code
        topics = classifier.classify_topics(text, language)
        comment.topics = topics
        
        # 3. Re-analyze sentiment with ORIGINAL code
        sentiment_result = sentiment_analyzer.analyze_sentiment(text, language)
        comment.sentiment = sentiment_result['sentiment']
        comment.sentiment_score = sentiment_result.get('sentiment_score', 0.5)
        
        # 4. Re-calculate aspect sentiments with ORIGINAL code
        if topics:
            aspect_sentiments = classifier.get_aspect_sentiments(
                text, topics, comment.sentiment, language
            )
            comment.aspect_sentiments = aspect_sentiments
        
        return True
        
    except Exception as e:
        logger.error(f"Error restoring comment {comment.id}: {e}")
        return False


def main():
    logger.info("=" * 80)
    logger.info("DATABASE RESTORE - Fixing corrupted data from failed NLP improvements")
    logger.info("=" * 80)
    
    db = SessionLocal()
    
    try:
        # Initialize ORIGINAL NLP components
        logger.info("\n[LOAD] Loading ORIGINAL NLP components...")
        comment_filter = CommentFilter()
        topic_classifier = TopicClassifier()
        sentiment_analyzer = MultilingualSentimentAnalyzer(use_gpu=False)
        logger.info("[OK] Components loaded\n")
        
        # Count total comments
        total = db.query(func.count(Comment.id)).scalar()
        logger.info(f"[STATS] Total comments to restore: {total}")
        
        # Confirm before proceeding
        confirm = input("\n[WARNING] This will overwrite ALL comment analysis data. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Cancelled by user")
            return
        
        # Process in batches
        batch_size = 100
        total_batches = (total + batch_size - 1) // batch_size
        success_count = 0
        error_count = 0
        
        logger.info(f"\n[PROCESS] Processing {total_batches} batches...")
        
        for batch_num in range(1, total_batches + 1):
            offset = (batch_num - 1) * batch_size
            
            # Fetch batch
            comments = db.query(Comment).order_by(Comment.id).offset(offset).limit(batch_size).all()
            
            if not comments:
                break
            
            logger.info(f"   Batch {batch_num}/{total_batches}: {len(comments)} comments")
            
            # Restore each comment
            for comment in comments:
                if restore_comment(comment, comment_filter, topic_classifier, sentiment_analyzer):
                    success_count += 1
                else:
                    error_count += 1
            
            # Commit batch
            try:
                db.commit()
                logger.info(f"   [OK] Batch {batch_num} committed")
            except Exception as e:
                db.rollback()
                logger.error(f"   [ERROR] Batch {batch_num} failed: {e}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("RESTORE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"[SUCCESS] Successfully restored: {success_count}")
        logger.info(f"[ERROR] Errors: {error_count}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\n[WARNING] Interrupted by user")
        db.rollback()
    except Exception as e:
        logger.error(f"\n[ERROR] Fatal error: {e}")
        db.rollback()
    finally:
        db.close()
        logger.info("\n[OK] Database connection closed")


if __name__ == '__main__':
    main()
