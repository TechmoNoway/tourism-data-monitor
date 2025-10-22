import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.database.connection import get_db
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province
from app.collectors.youtube_collector import create_youtube_collector
from app.collectors.google_reviews_collector import create_google_reviews_collector
from app.collectors.facebook_apify_collector import create_facebook_apify_collector
from app.collectors.facebook_posts_scraper import FacebookPostsScraper
from app.collectors.tiktok_apify_collector import create_tiktok_apify_collector
from app.collectors.facebook_rapid_collector import create_facebook_rapid_collector
from app.collectors.tiktok_rapid_collector import create_tiktok_rapid_collector
from app.collectors.google_maps_apify_collector import create_google_maps_apify_collector


class DataCollectionPipeline:
    """
    Main pipeline for collecting tourism data from multiple platforms
    """
    
    def __init__(self, 
                 youtube_api_key: Optional[str] = None,
                 google_maps_api_key: Optional[str] = None,
                 apify_api_token: Optional[str] = None,
                 rapidapi_key: Optional[str] = None,
                 use_rapidapi: bool = False):
        """
        Initialize data collection pipeline
        
        Args:
            youtube_api_key: YouTube Data API v3 key
            google_maps_api_key: Google Maps/Places API key
            apify_api_token: Apify API token (for Facebook & TikTok scraping)
            rapidapi_key: RapidAPI key (alternative to Apify)
            use_rapidapi: If True, use RapidAPI instead of Apify for FB/TikTok
        """
        
        self.logger = logging.getLogger("data_collection_pipeline")
        
        self.collectors = {}
        
        # YouTube - uses API (works well)
        if youtube_api_key:
            try:
                self.collectors['youtube'] = create_youtube_collector(youtube_api_key)
                self.logger.info("YouTube collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize YouTube collector: {str(e)}")
        
        # Google Reviews - uses API (works well)
        if google_maps_api_key:
            try:
                self.collectors['google_reviews'] = create_google_reviews_collector(google_maps_api_key)
                self.logger.info("Google Reviews collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Reviews collector: {str(e)}")
        
        # Facebook & TikTok - choose between Apify or RapidAPI
        if use_rapidapi and rapidapi_key:
            # Use RapidAPI for Facebook & TikTok
            try:
                self.collectors['facebook'] = create_facebook_rapid_collector(rapidapi_key)
                self.logger.info("Facebook (RapidAPI) collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Facebook RapidAPI collector: {str(e)}")
            
            try:
                self.collectors['tiktok'] = create_tiktok_rapid_collector(rapidapi_key)
                self.logger.info("TikTok (RapidAPI) collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize TikTok RapidAPI collector: {str(e)}")
                
        elif apify_api_token:
            # Use Apify for Facebook & TikTok (default)
            try:
                # Use FacebookPostsScraper for better post+comment collection
                facebook_collector = FacebookPostsScraper(apify_api_token)
                if facebook_collector.authenticate():
                    self.collectors['facebook'] = facebook_collector
                    self.logger.info("✓ Facebook Posts Scraper (Apify) initialized - Cost optimized!")
                else:
                    self.logger.error("Failed to authenticate Facebook Posts Scraper")
            except Exception as e:
                self.logger.error(f"Failed to initialize Facebook Apify collector: {str(e)}")
            
            try:
                self.collectors['tiktok'] = create_tiktok_apify_collector(apify_api_token)
                self.logger.info("TikTok (Apify) collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize TikTok Apify collector: {str(e)}")
            
            try:
                self.collectors['google_maps'] = create_google_maps_apify_collector(apify_api_token)
                self.logger.info("✓ Google Maps (Apify) collector initialized - High quality reviews!")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Maps Apify collector: {str(e)}")
        else:
            self.logger.warning("No Apify or RapidAPI token provided - Facebook & TikTok collectors not available")
    
    async def collect_for_attraction(
        self, 
        attraction_id: int, 
        platforms: Optional[List[str]] = None,
        limit_per_platform: int = 50
    ) -> Dict[str, Any]:
        """
        Collect data for a specific tourist attraction from all platforms
        
        Args:
            attraction_id: Database ID of the tourist attraction
            platforms: List of platforms to collect from (default: all available)
            limit_per_platform: Maximum items to collect per platform
            
        Returns:
            Summary of collection results
        """
        # Get attraction info from database
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
        """
        Collect from a specific platform with relevance filtering
        
        Args:
            platform: Platform name
            attraction_id: Database ID
            attraction_name: Name of attraction
            province_name: Name of province
            limit: Max items to collect
            
        Returns:
            Collection statistics
        """
        collector = self.collectors[platform]
        
        # Special handling for Facebook with validated best pages
        if platform == 'facebook' and hasattr(collector, 'collect_posts_with_comments'):
            return await self._collect_facebook_with_best_pages(
                collector, attraction_id, attraction_name, province_name, limit
            )
        
        # Generate optimized keywords using the collector's method
        keywords = collector.generate_search_keywords(attraction_name, province_name)
        
        self.logger.info(f"Searching {platform} with keywords: {keywords[:3]}...")  # Log first 3 keywords
        
        # Collect posts/places from API
        raw_posts = await collector.collect_posts(keywords, province_name, limit)
        initial_count = len(raw_posts)
        
        # Filter for relevance
        filtered_posts = collector.filter_relevant_posts(raw_posts, attraction_name, province_name)
        filtered_count = initial_count - len(filtered_posts)
        
        # Store filtered posts in database (returns mapping: platform_post_id -> db_post_id)
        post_id_mapping = collector.process_and_store_posts(filtered_posts, attraction_id)
        
        # Collect comments for each post (including existing posts!)
        all_comments = []
        for post in filtered_posts:
            # Get platform-specific post ID
            platform_post_id = None
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
            
            if not platform_post_id:
                continue
            
            # Get database post ID from mapping
            db_post_id = post_id_mapping.get(platform_post_id)
            if not db_post_id:
                self.logger.warning(f"No database ID found for post {platform_post_id}, skipping comments")
                continue
            
            try:
                self.logger.info(f"📥 Collecting comments for post {platform_post_id} (DB ID: {db_post_id})")
                comments = await collector.collect_comments(platform_post_id, 20)  # Limit comments per post
                
                if comments:
                    collector.process_and_store_comments(comments, db_post_id, attraction_id)
                    all_comments.extend(comments)
                    self.logger.info(f"   ✅ Stored {len(comments)} comments")
                else:
                    self.logger.info(f"   ℹ️  No comments found")
                    
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
        """
        Collect Facebook data using validated best pages (Smart Page Selection strategy)
        
        This method uses high-engagement pages identified through testing rather than
        keyword search, resulting in better comment collection rates.
        
        Returns:
            Collection statistics with comments included
        """
        from app.core.config import settings
        
        self.logger.info(f"🎯 Using Facebook Best Pages strategy for {attraction_name}")
        
        # Map attraction/province to best page
        page_url = None
        location_key = None
        use_direct_page = False
        
        # Try to match by attraction or province name
        attraction_lower = attraction_name.lower()
        province_lower = province_name.lower()
        
        if 'bà nà' in attraction_lower or 'ba na' in attraction_lower or 'đà nẵng' in province_lower:
            location_key = 'ba_na_hills'
        elif 'đà lạt' in attraction_lower or 'da lat' in attraction_lower or 'lâm đồng' in province_lower:
            location_key = 'da_lat'
        elif 'phú quốc' in attraction_lower or 'phu quoc' in attraction_lower or 'kiên giang' in province_lower:
            location_key = 'phu_quoc'
        
        if location_key and location_key in settings.FACEBOOK_BEST_PAGES:
            page_config = settings.FACEBOOK_BEST_PAGES[location_key]
            page_url = page_config['url']
            expected_comments = page_config['expected_comments_per_post']
            use_direct_page = True
            
            self.logger.info(f"✓ Matched to best page: {page_config['name']} "
                           f"(~{expected_comments:.1f} comments/post)")
            self.logger.info(f"📍 Using DIRECT page URL: {page_url}")
        else:
            # Fallback to keyword search if no best page matched
            self.logger.warning(f"⚠️  No best page for {attraction_name}, falling back to keyword search")
            keywords = collector.generate_search_keywords(attraction_name, province_name)
            page_url = keywords[0] if keywords else attraction_name
            self.logger.info(f"🔍 Using KEYWORD search: {page_url}")
        
        # CRITICAL: Ensure keywords is always a list (fix string iteration bug)
        keywords_list = [page_url] if isinstance(page_url, str) else page_url
        
        # Collect posts WITH comments using 2-actor strategy
        posts_with_comments = await collector.collect_posts_with_comments(
            keywords=keywords_list,
            limit=limit,
            comments_per_post=settings.FACEBOOK_COMMENTS_PER_POST
        )
        
        initial_count = len(posts_with_comments)
        
        # Filter for relevance
        filtered_posts = collector.filter_relevant_posts(posts_with_comments, attraction_name, province_name)
        filtered_count = initial_count - len(filtered_posts)
        
        # Store posts and get mapping (returns dict: platform_post_id -> db_post_id)
        post_id_mapping = collector.process_and_store_posts(filtered_posts, attraction_id)
        
        # Process and store comments (already collected with posts)
        all_comments = []
        for post in filtered_posts:
            # Get platform post ID
            platform_post_id = post.get('post_id')
            if not platform_post_id:
                continue
            
            # Get database post ID from mapping
            db_post_id = post_id_mapping.get(platform_post_id)
            if not db_post_id:
                continue
            
            comments = post.get('comments', [])
            
            if comments:
                collector.process_and_store_comments(comments, db_post_id, attraction_id)
                all_comments.extend(comments)
        
        # Log activity
        posts_stored = len(post_id_mapping)
        collector.log_collection_activity(
            attraction_id,
            posts_stored,
            len(all_comments),
            "success",
            posts_filtered=filtered_count,
            initial_posts=initial_count
        )
        
        # Safe division for average calculation (fix division-by-zero)
        avg_comments = (len(all_comments) / posts_stored) if posts_stored > 0 else 0.0
        
        self.logger.info(f"✅ Facebook collection completed: {posts_stored} posts, "
                        f"{len(all_comments)} comments ({avg_comments:.1f} avg/post)")
        
        return {
            'platform': 'facebook',
            'strategy': 'best_pages' if use_direct_page else 'keyword_search',
            'best_page_used': page_config['name'] if location_key else None,
            'page_url_used': page_url if use_direct_page else None,
            'posts_collected': posts_stored,
            'posts_filtered': filtered_count,
            'initial_posts': initial_count,
            'filter_efficiency': f"{(filtered_count/initial_count*100):.1f}%" if initial_count > 0 else "0%",
            'comments_collected': len(all_comments),
            'avg_comments_per_post': f"{avg_comments:.1f}",
            'collection_time': datetime.utcnow().isoformat()
        }
    # Collect data for all attractions in a province
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
                
                # Add longer delay between provinces
                await asyncio.sleep(5)
                
            except Exception as e:
                error_msg = f"Error collecting for province {province.name}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        db.close()
        
        self.logger.info(f"Full collection completed: {results['total_posts']} posts, {results['total_comments']} comments")
        return results
    
    def get_available_platforms(self) -> List[str]:
        """
        Get list of available collectors
        
        Returns:
            List of platform names
        """
        return list(self.collectors.keys())
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available collectors
        
        Returns:
            Collector statistics
        """
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
    """
    Create DataCollectionPipeline with API credentials
    
    Args:
        youtube_api_key: YouTube Data API v3 key
        google_maps_api_key: Google Maps/Places API key
        apify_api_token: Apify API token (for Facebook & TikTok scraping)
        rapidapi_key: RapidAPI key (alternative to Apify)
        use_rapidapi: If True, use RapidAPI instead of Apify (default: False)
    
    Returns:
        Configured DataCollectionPipeline instance
    """
    return DataCollectionPipeline(
        youtube_api_key=api_credentials.get('youtube_api_key'),
        google_maps_api_key=api_credentials.get('google_maps_api_key'),
        apify_api_token=api_credentials.get('apify_api_token'),
        rapidapi_key=api_credentials.get('rapidapi_key'),
        use_rapidapi=api_credentials.get('use_rapidapi', False)
    )