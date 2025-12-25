"""
Smart Collection Pipeline with Intelligence:
- Multi-platform priority system
- Target-based early stopping
- Duplicate detection
- Image collection
- Automatic analysis
- Statistics updates
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province
from app.models.comment import Comment
from app.models.social_post import SocialPost
from app.core.config import settings
from app.collectors.data_pipeline import DataCollectionPipeline
from app.services.sentiment_analyzer import MultilingualSentimentAnalyzer
from app.services.topic_classifier import TopicClassifier
from app.collectors.google_maps_collector import GoogleMapsApifyCollector

logger = logging.getLogger(__name__)


class SmartCollectionPipeline:
    """
    Intelligent collection pipeline that:
    1. Tries platforms in priority order
    2. Stops when target is reached
    3. Collects images
    4. Prevents duplicates
    5. Analyzes all comments
    6. Updates statistics
    """
    
    def __init__(self, 
                 youtube_api_key: Optional[str] = None,
                 apify_api_token: Optional[str] = None,
                 use_gpu: bool = False):
        self.pipeline = DataCollectionPipeline(youtube_api_key, apify_api_token)
        self.logger = logging.getLogger("smart_collection")
        
        # Initialize analyzers
        self.sentiment_analyzer = MultilingualSentimentAnalyzer(use_gpu=use_gpu)
        self.topic_classifier = TopicClassifier()
        
        # Image collector
        self.image_collector = None
        if apify_api_token:
            try:
                self.image_collector = GoogleMapsApifyCollector(apify_api_token, skip_sentiment=True)
                self.image_collector.authenticate()
                self.logger.info("Image collector initialized")
            except Exception as e:
                self.logger.warning(f"Image collector not available: {e}")
        
        # Get collection settings
        if settings.FULL_COLLECTION_MODE:
            self.target_comments = settings.FULL_TARGET_COMMENTS
            self.target_posts = settings.FULL_TARGET_POSTS
            self.platform_limits = settings.FULL_PLATFORM_LIMITS
        else:
            self.target_comments = settings.TARGET_COMMENTS_PER_ATTRACTION
            self.target_posts = settings.TARGET_POSTS_PER_ATTRACTION
            self.platform_limits = settings.PLATFORM_LIMITS
        
        self.platforms_priority = ['facebook', 'google_maps', 'tiktok', 'youtube']
        
        self.logger.info(f"Smart Collection Pipeline initialized")
        self.logger.info(f"Target: {self.target_comments} comments, {self.target_posts} posts")
        self.logger.info(f"Platform limits: {self.platform_limits}")
    
    async def collect_for_attraction(
        self, 
        attraction_id: int,
        force_collect: bool = False
    ) -> Dict[str, Any]:
        """
        Smart collection for a single attraction with:
        - Multi-platform priority
        - Early stopping at 80% target
        - Image collection
        - Automatic analysis
        - Statistics update
        """
        db: Session = next(get_db())
        
        try:
            attraction = db.query(TouristAttraction).filter(
                TouristAttraction.id == attraction_id
            ).first()
            
            if not attraction:
                raise ValueError(f"Attraction {attraction_id} not found")
            
            # Check current status
            current_comments = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attraction_id,
                Comment.is_meaningful == True
            ).scalar()
            
            current_posts = db.query(func.count(SocialPost.id)).filter(
                SocialPost.attraction_id == attraction_id
            ).scalar()
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"🎯 {attraction.name} ({attraction.province.name})")
            self.logger.info(f"   Current: {current_posts} posts, {current_comments} comments")
            self.logger.info(f"   Target: {self.target_comments} comments")
            self.logger.info(f"{'='*70}")
            
            # Check if target already reached
            if current_comments >= self.target_comments and not force_collect:
                self.logger.info(f"✓ Target already reached!")
                return {
                    'attraction_id': attraction_id,
                    'attraction_name': attraction.name,
                    'status': 'target_reached',
                    'posts_before': current_posts,
                    'comments_before': current_comments,
                    'posts_collected': 0,
                    'comments_collected': 0,
                    'platforms_used': []
                }
            
            result = {
                'attraction_id': attraction_id,
                'attraction_name': attraction.name,
                'province_name': attraction.province.name,
                'posts_before': current_posts,
                'comments_before': current_comments,
                'posts_collected': 0,
                'comments_collected': 0,
                'platforms_used': [],
                'status': 'in_progress'
            }
            
            # Try platforms in priority order
            for platform in self.platforms_priority:
                # Check if we have this collector
                if platform not in self.pipeline.collectors:
                    self.logger.debug(f"   {platform}: Not available")
                    continue
                
                # Refresh counts
                current_comments = db.query(func.count(Comment.id)).filter(
                    Comment.attraction_id == attraction_id,
                    Comment.is_meaningful == True
                ).scalar()
                
                comments_needed = self.target_comments - current_comments
                
                # Check if target reached
                if comments_needed <= 0:
                    self.logger.info(f"✓ Target reached: {current_comments} comments")
                    result['status'] = 'completed'
                    break
                
                # Check for 80% early stopping
                if current_comments >= self.target_comments * 0.8:
                    self.logger.info(f"✓ 80% target reached: {current_comments} comments")
                    result['status'] = 'completed'
                    break
                
                # Collect from this platform
                self.logger.info(f"\n🔍 Trying {platform.upper()}: Need {comments_needed} more comments")
                
                platform_limit = self.platform_limits.get(platform, 50)
                
                try:
                    posts_before = db.query(func.count(SocialPost.id)).filter(
                        SocialPost.attraction_id == attraction_id
                    ).scalar()
                    
                    comments_before = db.query(func.count(Comment.id)).filter(
                        Comment.attraction_id == attraction_id
                    ).scalar()
                    
                    # Collect from platform
                    await self.pipeline.collect_for_attraction(
                        attraction_id=attraction_id,
                        platforms=[platform],
                        limit_per_platform=platform_limit
                    )
                    
                    posts_after = db.query(func.count(SocialPost.id)).filter(
                        SocialPost.attraction_id == attraction_id
                    ).scalar()
                    
                    comments_after = db.query(func.count(Comment.id)).filter(
                        Comment.attraction_id == attraction_id
                    ).scalar()
                    
                    new_posts = posts_after - posts_before
                    new_comments = comments_after - comments_before
                    
                    if new_posts > 0 or new_comments > 0:
                        self.logger.info(f"   ✓ {platform}: +{new_posts} posts, +{new_comments} comments")
                        result['posts_collected'] += new_posts
                        result['comments_collected'] += new_comments
                        result['platforms_used'].append(platform)
                    else:
                        self.logger.info(f"   ⚠ {platform}: No new data")
                    
                    # Small delay between platforms
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"   ✗ {platform}: {str(e)}")
            
            # Post-collection processing
            self.logger.info(f"\n🔧 Post-processing...")
            
            # 1. Collect image if missing
            if not attraction.image_url and self.image_collector:
                try:
                    self.logger.info("   📸 Fetching image...")
                    image_url = await self._fetch_attraction_image(attraction)
                    if image_url:
                        attraction.image_url = image_url
                        db.commit()
                        self.logger.info(f"   ✓ Image saved")
                except Exception as e:
                    self.logger.warning(f"   ⚠ Image fetch failed: {e}")
            
            # 2. Analyze unanalyzed comments
            unanalyzed_count = await self._analyze_unanalyzed_comments(attraction_id, db)
            if unanalyzed_count > 0:
                self.logger.info(f"   ✓ Analyzed {unanalyzed_count} comments")
            
            # 3. Update attraction statistics
            self._update_attraction_stats(attraction, db)
            self.logger.info(f"   ✓ Statistics updated")
            
            # Final counts
            result['posts_total'] = db.query(func.count(SocialPost.id)).filter(
                SocialPost.attraction_id == attraction_id
            ).scalar()
            
            result['comments_total'] = db.query(func.count(Comment.id)).filter(
                Comment.attraction_id == attraction_id,
                Comment.is_meaningful == True
            ).scalar()
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"✓ COMPLETED: {attraction.name}")
            self.logger.info(f"   Final: {result['posts_total']} posts, {result['comments_total']} comments")
            self.logger.info(f"   New: +{result['posts_collected']} posts, +{result['comments_collected']} comments")
            self.logger.info(f"{'='*70}\n")
            
            return result
            
        finally:
            db.close()
    
    async def collect_for_province(
        self,
        province_id: int,
        max_attractions: Optional[int] = None
    ) -> Dict[str, Any]:
        """Collect for all attractions in a province"""
        db: Session = next(get_db())
        
        try:
            province = db.query(Province).filter(Province.id == province_id).first()
            
            if not province:
                raise ValueError(f"Province {province_id} not found")
            
            query = db.query(TouristAttraction).filter(
                TouristAttraction.province_id == province_id,
                TouristAttraction.is_active == True
            )
            
            if max_attractions and max_attractions > 0:
                query = query.limit(max_attractions)
            
            attractions = query.all()
            
            self.logger.info(f"\n🌍 PROVINCE: {province.name}")
            self.logger.info(f"   Processing {len(attractions)} attractions\n")
            
            results = {
                'province_id': province_id,
                'province_name': province.name,
                'attractions': [],
                'total_posts_collected': 0,
                'total_comments_collected': 0
            }
            
            for attraction in attractions:
                try:
                    result = await self.collect_for_attraction(attraction.id)
                    results['attractions'].append(result)
                    results['total_posts_collected'] += result['posts_collected']
                    results['total_comments_collected'] += result['comments_collected']
                    
                    # Delay between attractions
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting for {attraction.name}: {e}")
            
            return results
            
        finally:
            db.close()
    
    async def _fetch_attraction_image(self, attraction: TouristAttraction) -> Optional[str]:
        """Fetch image for attraction from Google Maps"""
        if not self.image_collector:
            return None
        
        try:
            search_query = f"{attraction.name} Vietnam"
            places = await self.image_collector.collect_posts(
                keywords=[search_query],
                location="Vietnam",
                limit=1
            )
            
            if places and len(places) > 0:
                place = places[0]
                media_urls = place.get('media_urls', [])
                if media_urls and len(media_urls) > 0:
                    return media_urls[0]
        except Exception as e:
            self.logger.debug(f"Image fetch error: {e}")
        
        return None
    
    async def _analyze_unanalyzed_comments(self, attraction_id: int, db: Session) -> int:
        """Analyze all unanalyzed comments for an attraction"""
        # Get unanalyzed comments
        comments = db.query(Comment).filter(
            Comment.attraction_id == attraction_id,
            Comment.sentiment.is_(None)
        ).all()
        
        if not comments:
            return 0
        
        analyzed_count = 0
        
        for comment in comments:
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
                
                analyzed_count += 1
                
            except Exception as e:
                self.logger.error(f"Error analyzing comment {comment.id}: {e}")
        
        db.commit()
        return analyzed_count
    
    def _update_attraction_stats(self, attraction: TouristAttraction, db: Session):
        """Update attraction statistics"""
        # Count meaningful comments
        meaningful_count = db.query(func.count(Comment.id)).filter(
            Comment.attraction_id == attraction.id,
            Comment.is_meaningful == True
        ).scalar()
        
        attraction.total_comments = meaningful_count
        attraction.total_reviews = meaningful_count
        attraction.updated_at = datetime.now()
        
        db.commit()
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available platforms"""
        return list(self.pipeline.collectors.keys())


def create_smart_pipeline(**credentials) -> SmartCollectionPipeline:
    """Factory function to create smart collection pipeline"""
    return SmartCollectionPipeline(**credentials)
