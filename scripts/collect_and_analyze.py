"""
Complete data collection and analysis pipeline
- Collects 100+ raw reviews per attraction (aiming for 60+ meaningful after filtering)
- Automatically runs sentiment analysis
- Updates attraction statistics
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.database.connection import SessionLocal
from app.models.tourist_attraction import TouristAttraction
from app.models.comment import Comment
from app.models.province import Province
from app.models.social_post import SocialPost
from sqlalchemy import func, and_
from app.core.config import settings
from app.collectors.google_maps_collector import GoogleMapsApifyCollector
from app.services.sentiment_analyzer import MultilingualSentimentAnalyzer
from app.services.topic_classifier import TopicClassifier
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_attractions_without_comments(min_comments: int = 60):
    """Get attractions with less than min_comments meaningful comments"""
    db = SessionLocal()
    try:
        attractions = db.query(TouristAttraction).outerjoin(
            Comment,
            and_(
                Comment.attraction_id == TouristAttraction.id,
                Comment.is_meaningful == True
            )
        ).group_by(TouristAttraction.id).having(
            func.count(Comment.id) < min_comments
        ).all()
        
        return attractions
    finally:
        db.close()


async def collect_for_attraction(collector: GoogleMapsApifyCollector, attraction: TouristAttraction, target_reviews: int = 100):
    """Collect Google Maps reviews for an attraction"""
    logger.info(f"  📍 {attraction.name}")
    
    try:
        # First, search for the place to get place_id
        search_query = f"{attraction.name} vietnam"
        logger.info(f"  🔍 Searching: {search_query}")
        
        places = await collector.collect_posts(
            keywords=[search_query],
            location="Vietnam",
            limit=1
        )
        
        if not places:
            logger.warning(f"  ⚠ Place not found on Google Maps")
            return 0
        
        place = places[0]
        place_id = place.get("post_id")
        place_name = place.get("author", attraction.name)
        
        logger.info(f"  ✓ Found: {place_name}")
        
        # Create or get social post for this place
        db = SessionLocal()
        try:
            existing_post = db.query(SocialPost).filter(
                SocialPost.attraction_id == attraction.id,
                SocialPost.platform_post_id == place_id
            ).first()
            
            if not existing_post:
                post_data = collector._convert_raw_post(place, attraction.id)
                post = SocialPost(**post_data.dict())
                db.add(post)
                db.commit()
                db.refresh(post)
                post_id = post.id
                logger.info(f"  ✓ Created place post (ID: {post_id})")
            else:
                post_id = existing_post.id
                logger.info(f"  ✓ Using existing post (ID: {post_id})")
            
        finally:
            db.close()
        
        # Collect reviews
        logger.info(f"  📝 Collecting {target_reviews} reviews...")
        reviews = await collector.collect_comments(place_id, limit=target_reviews)
        
        if not reviews:
            logger.warning(f"  ⚠ No reviews found")
            return 0
        
        logger.info(f"  ✓ Found {len(reviews)} reviews")
        
        # Save reviews to database
        db = SessionLocal()
        try:
            saved_count = 0
            for review_data in reviews:
                # Check if review already exists
                existing = db.query(Comment).filter(
                    Comment.attraction_id == attraction.id,
                    Comment.platform_comment_id == review_data["comment_id"]
                ).first()
                
                if existing:
                    continue
                
                # Convert to CommentCreate and save
                comment_data = collector._convert_raw_comment(review_data, post_id, attraction.id)
                comment = Comment(**comment_data.dict())
                comment.is_meaningful = False  # Will be analyzed later
                
                db.add(comment)
                saved_count += 1
            
            db.commit()
            logger.info(f"  ✓ Saved {saved_count} new reviews")
            return saved_count
            
        except Exception as e:
            db.rollback()
            logger.error(f"  ❌ Database error: {str(e)}")
            return 0
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"  ❌ Collection error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


def analyze_new_comments(use_gpu: bool = True, batch_size: int = 64):
    """Analyze all unanalyzed comments"""
    logger.info("\n" + "="*80)
    logger.info("SENTIMENT ANALYSIS")
    logger.info("="*80)
    
    db = SessionLocal()
    
    try:
        # Get unanalyzed comments
        comments = db.query(Comment).filter(Comment.sentiment.is_(None)).all()
        
        if not comments:
            logger.info("No comments to analyze!")
            return 0
        
        total = len(comments)
        logger.info(f"Found {total} comments to analyze")
        logger.info(f"GPU: {'Enabled' if use_gpu else 'Disabled'}, Batch size: {batch_size}\n")
        
        # Initialize analyzers
        analyzer = MultilingualSentimentAnalyzer(use_gpu=use_gpu)
        topic_classifier = TopicClassifier()
        
        analyzed_count = 0
        meaningful_count = 0
        
        for i in range(0, total, batch_size):
            batch = comments[i:i + batch_size]
            batch_texts = [c.content for c in batch]
            
            results = analyzer.analyze_batch(batch_texts, batch_size=batch_size)
            
            for comment, result in zip(batch, results):
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
                    meaningful_count += 1
                
                # Topic classification
                topics = topic_classifier.classify_topics(comment.content, comment.language)
                comment.topics = topics if topics else None
                
                # Aspect-level sentiments
                if topics:
                    aspect_sentiments = topic_classifier.get_aspect_sentiments(
                        comment.content,
                        topics,
                        comment.sentiment,
                        comment.language
                    )
                    comment.aspect_sentiments = aspect_sentiments
                
                analyzed_count += 1
            
            db.commit()
            
            progress = (i + len(batch)) / total * 100
            logger.info(f"   Progress: {i + len(batch)}/{total} ({progress:.1f}%)")
        
        logger.info(f"\n✓ Analyzed {analyzed_count} comments")
        logger.info(f"✓ {meaningful_count} marked as meaningful")
        
        return meaningful_count
        
    finally:
        db.close()


def update_attraction_stats():
    """Update total_comments and total_reviews for all attractions"""
    logger.info("\n" + "="*80)
    logger.info("UPDATING ATTRACTION STATISTICS")
    logger.info("="*80)
    
    db = SessionLocal()
    
    try:
        attractions = db.query(TouristAttraction).all()
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
        logger.info(f"✓ Updated {updated} attractions")
        
    finally:
        db.close()


async def main():
    """Main collection and analysis pipeline"""
    logger.info("="*80)
    logger.info("COMPLETE DATA COLLECTION & ANALYSIS PIPELINE")
    logger.info("="*80)
    
    # Check Apify token
    if not settings.APIFY_API_TOKEN:
        logger.error("❌ APIFY_API_TOKEN not configured!")
        return
    
    # Initialize collector
    collector = GoogleMapsApifyCollector(settings.APIFY_API_TOKEN, skip_sentiment=True)
    
    if not collector.authenticate():
        logger.error("❌ Failed to authenticate with Apify")
        return
    
    # Step 1: Collection
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("="*80)
    
    attractions = get_attractions_without_comments(min_comments=60)
    
    if not attractions:
        logger.info("✓ All attractions have sufficient comments!")
    else:
        logger.info(f"\nFound {len(attractions)} attractions needing more comments")
        logger.info(f"Target: 100 raw reviews per attraction (aiming for 60+ meaningful)\n")
        
        total_collected = 0
        success_count = 0
        
        for i, attraction in enumerate(attractions, 1):
            logger.info(f"\n[{i}/{len(attractions)}] Processing:")
            collected = await collect_for_attraction(collector, attraction, target_reviews=100)
            total_collected += collected
            
            if collected > 0:
                success_count += 1
            
            # Small delay between requests
            if i < len(attractions):
                await asyncio.sleep(3)
        
        logger.info(f"\n✓ Collection complete: {total_collected} raw reviews from {success_count} attractions")
    
    # Step 2: Sentiment Analysis
    logger.info("\n" + "="*80)
    logger.info("STEP 2: SENTIMENT ANALYSIS")
    logger.info("="*80)
    
    meaningful_count = analyze_new_comments(use_gpu=True, batch_size=64)
    
    # Step 3: Update Statistics
    logger.info("\n" + "="*80)
    logger.info("STEP 3: UPDATE STATISTICS")
    logger.info("="*80)
    
    update_attraction_stats()
    
    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"✓ Collected and analyzed data")
    logger.info(f"✓ {meaningful_count} new meaningful comments")
    logger.info(f"✓ Attraction statistics updated")
    logger.info("\n🎉 All done! Check the UI for updated data.")


if __name__ == "__main__":
    asyncio.run(main())
