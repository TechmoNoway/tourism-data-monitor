import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.database.connection import get_db
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province
from app.collectors.youtube_collector import create_youtube_collector
from app.collectors.google_reviews_collector import create_google_reviews_collector
from app.collectors.facebook_collector import create_facebook_collector


class DataCollectionPipeline:
    """
    Main pipeline for collecting tourism data from multiple platforms
    """
    
    def __init__(self, 
                 youtube_api_key: Optional[str] = None,
                 google_maps_api_key: Optional[str] = None,
                 facebook_access_token: Optional[str] = None,
                 facebook_app_id: Optional[str] = None,
                 facebook_app_secret: Optional[str] = None):
        
        self.logger = logging.getLogger("data_collection_pipeline")
        
        # Initialize collectors
        self.collectors = {}
        
        if youtube_api_key:
            try:
                self.collectors['youtube'] = create_youtube_collector(youtube_api_key)
                self.logger.info("YouTube collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize YouTube collector: {str(e)}")
        
        if google_maps_api_key:
            try:
                self.collectors['google_reviews'] = create_google_reviews_collector(google_maps_api_key)
                self.logger.info("Google Reviews collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Reviews collector: {str(e)}")
        
        if facebook_access_token and facebook_app_id and facebook_app_secret:
            try:
                self.collectors['facebook'] = create_facebook_collector(
                    facebook_access_token, 
                    facebook_app_id, 
                    facebook_app_secret
                )
                self.logger.info("Facebook collector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Facebook collector: {str(e)}")
    
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
        
        # Prepare keywords for search
        keywords = [attraction.name]
        if attraction.description:
            # Add location-specific keywords
            keywords.extend([
                f"{attraction.name} review",
                f"{attraction.name} du lịch",
                f"{attraction.name} {attraction.province.name}"
            ])
        
        # If no platforms specified, use all available
        if platforms is None:
            platforms = list(self.collectors.keys())
        
        # Collection results
        results = {
            'attraction_id': attraction_id,
            'attraction_name': attraction.name,
            'collection_time': datetime.utcnow().isoformat(),
            'platforms': {},
            'total_posts': 0,
            'total_comments': 0,
            'errors': []
        }
        
        # Collect from each platform
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
                    keywords, 
                    attraction.province.name,
                    limit_per_platform
                )
                
                results['platforms'][platform] = platform_result
                results['total_posts'] += platform_result['posts_collected']
                results['total_comments'] += platform_result['comments_collected']
                
                self.logger.info(f"Completed collection from {platform} for {attraction.name}")
                
            except Exception as e:
                error_msg = f"Error collecting from {platform}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                
                # Log error to database
                collector = self.collectors[platform]
                collector.log_collection_activity(
                    attraction_id, 0, 0, "error", error_msg
                )
        
        db.close()
        
        self.logger.info(f"Collection completed for {attraction.name}: {results['total_posts']} posts, {results['total_comments']} comments")
        return results
    
    async def _collect_from_platform(
        self, 
        platform: str, 
        attraction_id: int, 
        keywords: List[str], 
        location: str,
        limit: int
    ) -> Dict[str, Any]:
        """
        Collect data from a specific platform
        
        Args:
            platform: Platform name
            attraction_id: Tourist attraction ID
            keywords: Search keywords
            location: Location filter
            limit: Maximum items to collect
            
        Returns:
            Collection results for the platform
        """
        collector = self.collectors[platform]
        
        # Collect posts/places
        posts = await collector.collect_posts(keywords, location, limit)
        
        # Store posts in database
        post_ids = collector.process_and_store_posts(posts, attraction_id)
        
        # Collect comments for each post
        all_comments = []
        for i, post in enumerate(posts):
            if i >= len(post_ids):  # Skip if post wasn't stored
                continue
                
            post_id = post_ids[i]
            
            # Get platform-specific post ID
            platform_post_id = None
            if platform == 'youtube':
                platform_post_id = post['video_id']
            elif platform == 'google_reviews':
                platform_post_id = post['place_id']
            elif platform == 'facebook':
                platform_post_id = post['post_id']
            
            if platform_post_id:
                try:
                    comments = await collector.collect_comments(platform_post_id, 20)  # Limit comments per post
                    
                    if comments:
                        collector.process_and_store_comments(comments, post_id)
                        all_comments.extend(comments)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to collect comments for post {platform_post_id}: {str(e)}")
        
        # Log collection activity
        collector.log_collection_activity(
            attraction_id, 
            len(post_ids), 
            len(all_comments), 
            "success"
        )
        
        return {
            'platform': platform,
            'posts_collected': len(post_ids),
            'comments_collected': len(all_comments),
            'keywords_used': keywords,
            'collection_time': datetime.utcnow().isoformat()
        }
    
    async def collect_for_province(
        self, 
        province_id: int, 
        platforms: Optional[List[str]] = None,
        limit_per_attraction: int = 20
    ) -> Dict[str, Any]:
        """
        Collect data for all attractions in a province
        
        Args:
            province_id: Province ID
            platforms: List of platforms to collect from
            limit_per_attraction: Maximum items per attraction per platform
            
        Returns:
            Summary of collection results
        """
        # Get attractions in province
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
        
        # Collect for each attraction
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
        """
        Collect data for all provinces
        
        Args:
            platforms: List of platforms to collect from
            limit_per_attraction: Maximum items per attraction per platform
            
        Returns:
            Summary of collection results
        """
        # Get all provinces
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
        
        # Collect for each province
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
    Create a data collection pipeline with provided API credentials
    
    Args:
        **api_credentials: Dictionary containing API keys and tokens
        
    Returns:
        DataCollectionPipeline instance
    """
    return DataCollectionPipeline(
        youtube_api_key=api_credentials.get('youtube_api_key'),
        google_maps_api_key=api_credentials.get('google_maps_api_key'),
        facebook_access_token=api_credentials.get('facebook_access_token'),
        facebook_app_id=api_credentials.get('facebook_app_id'),
        facebook_app_secret=api_credentials.get('facebook_app_secret')
    )