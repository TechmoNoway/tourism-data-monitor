import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from app.core.config import settings
from app.database.connection import get_db
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province
from app.collectors.youtube_collector import create_youtube_collector
from app.collectors.facebook_posts_scraper import FacebookPostsScraper
from app.collectors.tiktok_collector import create_tiktok_collector
from app.collectors.google_maps_collector import create_google_maps_collector


class DataCollectionPipeline:
  
    def __init__(self, 
                 youtube_api_key: Optional[str] = None,
                 apify_api_token: Optional[str] = None):        
        self.logger = logging.getLogger("data_collection_pipeline")
        
        self.collectors = {}
        
        if youtube_api_key:
            try:
                self.collectors['youtube'] = create_youtube_collector(youtube_api_key)
                self.logger.info("YouTube collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize YouTube collector: {str(e)}")
        
        if apify_api_token:
            try:
                facebook_collector = FacebookPostsScraper(apify_api_token)
                if facebook_collector.authenticate():
                    self.collectors['facebook'] = facebook_collector
                    self.logger.info("[OK] Facebook Posts Scraper initialized")
                else:
                    self.logger.error("Failed to authenticate Facebook Posts Scraper")
            except Exception as e:
                self.logger.error(f"Failed to initialize Facebook collector: {str(e)}")
            
            try:
                self.collectors['tiktok'] = create_tiktok_collector(apify_api_token)
                self.logger.info("[OK] TikTok collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize TikTok collector: {str(e)}")
            
            try:
                self.collectors['google_maps'] = create_google_maps_collector(apify_api_token)
                self.logger.info("[OK] Google Maps collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Maps collector: {str(e)}")
        else:
            self.logger.warning("No Apify token provided - Facebook, TikTok & Google Maps collectors not available")
    
    async def collect_for_attraction(
        self, 
        attraction_id: int, 
        platforms: Optional[List[str]] = None,
        limit_per_platform: int = 50
    ) -> Dict[str, Any]:
        db = next(get_db())
        attraction = db.query(TouristAttraction).filter(
            TouristAttraction.id == attraction_id
        ).first()
        
        if not attraction:
            raise ValueError(f"Attraction with ID {attraction_id} not found")
        
        attraction_name = attraction.name
        province_name = attraction.province.name
        
        if platforms is None:
            platforms = list(self.collectors.keys())
        
        results = {
            'attraction_id': attraction_id,
            'attraction_name': attraction_name,
            'province_name': province_name,
            'collection_time': datetime.utcnow().isoformat(),
            'platforms': {},
            'total_posts': 0,
            'total_comments': 0,
            'errors': []
        }
        
        for platform in platforms:
            if platform not in self.collectors:
                error_msg = f"Collector for {platform} not available"
                self.logger.warning(error_msg)
                results['errors'].append(error_msg)
                continue
            
            try:
                platform_result = await self._collect_from_platform(
                    platform, 
                    attraction_id, 
                    attraction_name,
                    province_name,
                    limit_per_platform
                )
                
                results['platforms'][platform] = platform_result
                results['total_posts'] += platform_result['posts_collected']
                results['total_comments'] += platform_result['comments_collected']
                
                self.logger.info(f"Completed collection from {platform} for {attraction_name}")
                
            except Exception as e:
                error_msg = f"Error collecting from {platform}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                
                collector = self.collectors[platform]
                collector.log_collection_activity(
                    attraction_id, 0, 0, "error", error_msg
                )
        
        db.close()
        
        self.logger.info(f"Collection completed for {attraction_name}: {results['total_posts']} posts, {results['total_comments']} comments")
        return results
    
    async def _collect_from_platform(
        self, 
        platform: str, 
        attraction_id: int, 
        attraction_name: str,
        province_name: str,
        limit: int
    ) -> Dict[str, Any]:
        collector = self.collectors[platform]
        
        if platform == 'facebook' and hasattr(collector, 'collect_posts_with_comments'):
            return await self._collect_facebook_with_best_pages(
                collector, attraction_id, attraction_name, province_name, limit
            )
        
        keywords = collector.generate_search_keywords(attraction_name, province_name)
        
        self.logger.info(f"Searching {platform} with keywords: {keywords[:3]}...")  # Log first 3 keywords
        
        raw_posts = await collector.collect_posts(keywords, province_name, limit)
        initial_count = len(raw_posts)
        
        # Filter for relevance
        filtered_posts = collector.filter_relevant_posts(raw_posts, attraction_name, province_name)
        filtered_count = initial_count - len(filtered_posts)
        
        # Store filtered posts in database (returns mapping: platform_post_id -> db_post_id)
        post_id_mapping = collector.process_and_store_posts(filtered_posts, attraction_id)
        
        all_comments = []
        for post in filtered_posts:
            platform_post_id = None
            # For TikTok, we need the URL for comment collection
            tiktok_url = None
            
            if platform == 'youtube':
                platform_post_id = post.get('video_id')
            elif platform == 'google_reviews':
                platform_post_id = post.get('place_id')
            elif platform == 'google_maps':
                platform_post_id = post.get('post_id')  # Google Maps uses place_id as post_id
            elif platform == 'facebook':
                platform_post_id = post.get('post_id')
            elif platform == 'tiktok':
                platform_post_id = post.get('post_id') or post.get('video_id')
                # TikTok comments need URL, not ID
                tiktok_url = post.get('url')
            
            if not platform_post_id:
                continue
            
            db_post_id = post_id_mapping.get(platform_post_id)
            if not db_post_id:
                self.logger.warning(f"No database ID found for post {platform_post_id}, skipping comments")
                continue
            
            try:
                self.logger.info(f"📥 Collecting comments for post {platform_post_id} (DB ID: {db_post_id})")
                
                # TikTok uses URL for comment collection, others use ID
                if platform == 'tiktok':
                    if not tiktok_url:
                        self.logger.warning(f"No URL found for TikTok post {platform_post_id}, skipping comments")
                        continue
                    # Scale 3x: Previous 20 → 60 comments per post
                    comments = await collector.collect_comments(tiktok_url, 60)
                else:
                    # Scale 3x: Previous 20 → 60 comments per post
                    comments = await collector.collect_comments(platform_post_id, 60)
                
                if comments:
                    collector.process_and_store_comments(comments, db_post_id, attraction_id)
                    all_comments.extend(comments)
                    self.logger.info(f"   [SUCCESS] Stored {len(comments)} comments")
                else:
                    self.logger.info("   [INFO] No comments found")
                    
            except Exception as e:
                self.logger.warning(f"Failed to collect comments for post {platform_post_id}: {str(e)}")
        
        # Count newly created posts (mapping includes both new and existing)
        posts_stored = len(post_id_mapping)
        
        # Log collection activity with filtering statistics
        collector.log_collection_activity(
            attraction_id, 
            posts_stored, 
            len(all_comments), 
            "success",
            posts_filtered=filtered_count,
            initial_posts=initial_count
        )
        
        return {
            'platform': platform,
            'posts_collected': posts_stored,
            'posts_filtered': filtered_count,
            'initial_posts': initial_count,
            'filter_efficiency': f"{(filtered_count/initial_count*100):.1f}%" if initial_count > 0 else "0%",
            'comments_collected': len(all_comments),
            'keywords_used': keywords[:5],  # Only include first 5 keywords in results
            'collection_time': datetime.utcnow().isoformat()
        }
    
    async def _collect_facebook_with_best_pages(
        self,
        collector,
        attraction_id: int,
        attraction_name: str,
        province_name: str,
        limit: int
    ) -> Dict[str, Any]:        
        self.logger.info(f"[SEARCH] Using Facebook SEARCH-ONLY strategy for {attraction_name}")
        
        search_query = attraction_name[:30]
        
        self.logger.info(f"[QUERY] Search query: '{search_query}'")
        
        posts_with_comments = await collector.collect_posts_with_comments(
            keywords=[search_query],  # Pass as list with single query
            limit=limit,
            comments_per_post=settings.FACEBOOK_COMMENTS_PER_POST
        )
        
        initial_count = len(posts_with_comments)
        
        filtered_posts = collector.filter_relevant_posts(posts_with_comments, attraction_name, province_name)
        filtered_count = initial_count - len(filtered_posts)
        
        post_id_mapping = collector.process_and_store_posts(filtered_posts, attraction_id)
        
        all_comments = []
        for post in filtered_posts:
            platform_post_id = post.get('post_id')
            if not platform_post_id:
                continue
            
            db_post_id = post_id_mapping.get(platform_post_id)
            if not db_post_id:
                continue
            
            comments = post.get('comments', [])
            
            if comments:
                collector.process_and_store_comments(comments, db_post_id, attraction_id)
                all_comments.extend(comments)
        
        posts_stored = len(post_id_mapping)
        collector.log_collection_activity(
            attraction_id,
            posts_stored,
            len(all_comments),
            "success",
            posts_filtered=filtered_count,
            initial_posts=initial_count
        )
        
        avg_comments = (len(all_comments) / posts_stored) if posts_stored > 0 else 0.0
        
        return {
            'platform': 'facebook',
            'strategy': 'search_only',
            'search_query': search_query,
            'posts_collected': posts_stored,
            'posts_filtered': filtered_count,
            'initial_posts': initial_count,
            'filter_efficiency': f"{(filtered_count/initial_count*100):.1f}%" if initial_count > 0 else "0%",
            'comments_collected': len(all_comments),
            'avg_comments_per_post': f"{avg_comments:.1f}",
            'collection_time': datetime.now(timezone.utc).isoformat()
        }

    async def collect_for_province(
        self, 
        province_id: int, 
        platforms: Optional[List[str]] = None,
        limit_per_attraction: int = 20
    ) -> Dict[str, Any]:
        db = next(get_db())
        province = db.query(Province).filter(Province.id == province_id).first()
        
        if not province:
            raise ValueError(f"Province with ID {province_id} not found")
        
        attractions = db.query(TouristAttraction).filter(
            TouristAttraction.province_id == province_id
        ).all()
        
        results = {
            'province_id': province_id,
            'province_name': province.name,
            'attractions_processed': 0,
            'total_posts': 0,
            'total_comments': 0,
            'collection_time': datetime.utcnow().isoformat(),
            'attractions': [],
            'errors': []
        }
        
        for attraction in attractions:
            try:
                attraction_result = await self.collect_for_attraction(
                    attraction.id, 
                    platforms, 
                    limit_per_attraction
                )
                
                results['attractions'].append(attraction_result)
                results['attractions_processed'] += 1
                results['total_posts'] += attraction_result['total_posts']
                results['total_comments'] += attraction_result['total_comments']
                
                # Add delay between attractions to avoid rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                error_msg = f"Error collecting for attraction {attraction.name}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        db.close()
        
        self.logger.info(f"Province collection completed for {province.name}: {results['total_posts']} posts, {results['total_comments']} comments")
        return results
    
    async def collect_all_provinces(
        self, 
        platforms: Optional[List[str]] = None,
        limit_per_attraction: int = 10
    ) -> Dict[str, Any]:
        db = next(get_db())
        provinces = db.query(Province).all()
        
        results = {
            'collection_type': 'all_provinces',
            'provinces_processed': 0,
            'total_attractions': 0,
            'total_posts': 0,
            'total_comments': 0,
            'collection_time': datetime.utcnow().isoformat(),
            'provinces': [],
            'errors': []
        }
        
        for province in provinces:
            try:
                province_result = await self.collect_for_province(
                    province.id, 
                    platforms, 
                    limit_per_attraction
                )
                
                results['provinces'].append(province_result)
                results['provinces_processed'] += 1
                results['total_attractions'] += province_result['attractions_processed']
                results['total_posts'] += province_result['total_posts']
                results['total_comments'] += province_result['total_comments']
                
                await asyncio.sleep(5)
                
            except Exception as e:
                error_msg = f"Error collecting for province {province.name}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        db.close()
        
        self.logger.info(f"Full collection completed: {results['total_posts']} posts, {results['total_comments']} comments")
        return results
    
    def get_available_platforms(self) -> List[str]:
        return list(self.collectors.keys())
    
    def get_collection_stats(self) -> Dict[str, Any]:
        stats = {
            'total_collectors': len(self.collectors),
            'available_platforms': list(self.collectors.keys()),
            'collector_info': {}
        }
        
        for platform, collector in self.collectors.items():
            stats['collector_info'][platform] = {
                'platform': collector.platform_name,
                'authenticated': True  # If it's in the dict, it's authenticated
            }
        
        return stats


# Factory function for easy instantiation
def create_data_pipeline(**api_credentials) -> DataCollectionPipeline:
    return DataCollectionPipeline(
        youtube_api_key=api_credentials.get('youtube_api_key'),
        apify_api_token=api_credentials.get('apify_api_token')
    )