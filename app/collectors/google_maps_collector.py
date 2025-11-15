from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    from apify_client import ApifyClient
except ImportError:
    logging.warning("Apify client not installed. Run: pip install apify-client")

from app.collectors.base_collector import BaseCollector
from app.schemas.post import SocialPostCreate, PlatformEnum
from app.schemas.comment import CommentCreate


class GoogleMapsApifyCollector(BaseCollector):

    
    def __init__(self, apify_api_token: str):
        super().__init__("google_maps")
        self.apify_token = apify_api_token
        self.client = None
        
    def authenticate(self, **credentials) -> bool:
        try:
            self.client = ApifyClient(self.apify_token)
            user_info = self.client.user().get()
            if user_info:
                self.logger.info(f"[OK] Apify authenticated: {user_info.get('username', 'Unknown')}")
                self.logger.info(f"[BALANCE] Balance: ${user_info.get('balance', 0):.2f}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Apify authentication failed: {str(e)}")
            return False
    
    async def collect_posts(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
        
        if isinstance(keywords, str):
            keywords = [keywords]
        
        all_places = []
        
        try:
            # Search for each keyword
            for keyword in keywords[:5]:  
                search_query = f"{keyword} {location}" if location else keyword
                
                self.logger.info(f"[SEARCH] Searching Google Maps: {search_query}")
                
                actor_id = "compass/google-maps-extractor"
                
                run_input = {
                    "searchString": search_query,
                    "maxCrawledPlaces": min(limit, 30),  
                    "language": "vi",
                    "includeWebResults": False,
                    "includeHistogram": False,
                    "maxImages": 1,  
                    "maxReviews": 0  
                }
                
                self.logger.info(f"Running Google Maps Extractor (max {min(limit, 30)} places)")
                
                run = self.client.actor(actor_id).call(run_input=run_input)
                dataset_id = run.get("defaultDatasetId")
                
                if not dataset_id:
                    self.logger.error("[ERROR] No dataset_id returned")
                    continue
                
                items = self.client.dataset(dataset_id).list_items().items
                self.logger.info(f"[STATS] Found {len(items)} places")
                
                for item in items:
                    place = self._parse_place(item)
                    if place:
                        all_places.append(place)
                        
        except Exception as e:
            self.logger.error(f"Error searching Google Maps: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.logger.info(f"[OK] Found {len(all_places)} Google Maps places")
        return all_places
    
    async def collect_comments(
        self, 
        platform_post_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
        
        reviews = []
        
        try:
            self.logger.info(f"[REVIEWS] Collecting reviews for place: {platform_post_id}")
            
            actor_id = "compass/Google-Maps-Reviews-Scraper"
            
            run_input = {
                "placeIds": [platform_post_id],
                "maxReviews": min(limit, 100),
                "reviewsSort": "newest",  # or "mostRelevant", "highestRanking", "lowestRanking"
                "language": "vi",
                "onlyWithText": True,  # Skip empty reviews
                "skipEmptyReviews": True  # Quality filtering
            }
            
            self.logger.info(f"[BALANCE] Running Google Maps Reviews Scraper (max {limit} reviews)")
            
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            
            if not dataset_id:
                self.logger.error("[ERROR] No dataset_id returned from Reviews Scraper")
                return reviews
            
            items = self.client.dataset(dataset_id).list_items().items
            self.logger.info(f"[REVIEWS] Reviews Scraper returned {len(items)} reviews")
            
            for item in items:
                review = self._parse_review(item)
                if review:
                    reviews.append(review)
            
            self.logger.info(f"[SUCCESS] Successfully parsed {len(reviews)} reviews")
            
        except Exception as e:
            self.logger.error(f"Error collecting reviews: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return reviews
    
    def _parse_place(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Google Maps place data into post format"""
        try:
            place_id = item.get("placeId")
            if not place_id:
                self.logger.warning("No place ID found")
                return None
            
            place_data = {
                "post_id": place_id,
                "url": item.get("url") or f"https://www.google.com/maps/place/?q=place_id:{place_id}",
                "text": item.get("description", ""),
                "author": item.get("title", "Unknown Place"),
                "author_id": place_id,
                "published_at": None,  # Places don't have publish dates
                "like_count": 0,
                "comment_count": item.get("totalScore", 0),
                "share_count": 0,
                "view_count": 0,
                "media_urls": [item.get("imageUrl")] if item.get("imageUrl") else [],
                "post_type": "place",
                # Extra metadata
                "rating": item.get("rating"),
                "address": item.get("address"),
                "phone": item.get("phone"),
                "website": item.get("website"),
                "category": item.get("categoryName")
            }
            
            return place_data
            
        except Exception as e:
            self.logger.error(f"Error parsing place: {str(e)}")
            return None
    
    def _parse_review(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Google Maps review into comment format"""
        try:
            text = item.get("text") or ""
            if not text or not text.strip():
                self.logger.debug(f"Skipping review without text content from {item.get('name', 'Unknown')}")
                return None
            
            word_count = len(text.split())
            if word_count < 3:
                self.logger.debug(f"Skipping short review ({word_count} words): {text[:50]}...")
                return None
            
            review_id = item.get("reviewId")
            if not review_id:
                import hashlib
                author = item.get("name", "")
                review_id = hashlib.md5(f"{author}{text}".encode()).hexdigest()
            
            review_data = {
                "comment_id": review_id,
                "text": text,
                "author_name": item.get("name", "Unknown"),
                "author_id": item.get("reviewerId", ""),
                "published_at": item.get("publishedAtDate"),
                "like_count": item.get("likesCount", 0),
                "reply_count": 0,
                # Extra metadata
                "rating": item.get("stars"),
                "review_url": item.get("reviewUrl")
            }
            
            return review_data
            
        except Exception as e:
            self.logger.error(f"Error parsing review: {str(e)}")
            return None
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        """Convert raw place data to SocialPostCreate schema"""
        return SocialPostCreate(
            attraction_id=attraction_id,
            platform=PlatformEnum.GOOGLE_MAPS,
            platform_post_id=raw_post["post_id"],
            content=raw_post.get("text", ""),
            author=raw_post.get("author", "Unknown"),
            author_id=raw_post.get("author_id"),
            post_url=raw_post.get("url"),
            post_date=self._parse_datetime(raw_post.get("published_at")),
            like_count=raw_post.get("like_count", 0),
            comment_count=raw_post.get("comment_count", 0),
            share_count=raw_post.get("share_count", 0),
            view_count=raw_post.get("view_count", 0),
            media_urls=raw_post.get("media_urls", []),
            post_type=raw_post.get("post_type", "place")
        )
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int, attraction_id: int) -> CommentCreate:
        """Convert raw review to CommentCreate schema"""
        return CommentCreate(
            post_id=post_id,
            attraction_id=attraction_id,
            platform=PlatformEnum.GOOGLE_MAPS,
            platform_comment_id=raw_comment["comment_id"],
            content=raw_comment.get("text", ""),
            author=raw_comment.get("author_name", "Unknown"),
            author_id=raw_comment.get("author_id"),
            comment_date=self._parse_datetime(raw_comment.get("published_at")),
            like_count=raw_comment.get("like_count", 0),
            reply_count=raw_comment.get("reply_count", 0)
        )
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime from various formats"""
        if not date_str:
            return None
        
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            pass
        
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except Exception:
            return None
    
    async def auto_discover_attractions(
        self,
        province_name: str,
        main_city: str,
        attraction_types: List[str],
        limit_per_type: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Auto-discover tourist attractions in a province using Google Maps search
        
        Args:
            province_name: Province name (e.g., "Khánh Hòa")
            main_city: Main city to search in (e.g., "Nha Trang")
            attraction_types: List of attraction categories to search
                             (e.g., ["tourist attraction", "beach", "temple"])
            limit_per_type: Maximum results per category (default: 3)
        
        Returns:
            List of attraction dictionaries ready for TouristAttraction model
            
        Example:
            >>> attractions = await collector.auto_discover_attractions(
            ...     province_name="Khánh Hòa",
            ...     main_city="Nha Trang",
            ...     attraction_types=["beach", "temple", "museum"],
            ...     limit_per_type=5
            ... )
            >>> print(f"Found {len(attractions)} attractions")
        """
        all_attractions = []
        discovered_place_ids = set()
        
        self.logger.info(f"[SEARCH] Auto-discovering attractions in {province_name} ({main_city})")
        self.logger.info(f"Categories: {', '.join(attraction_types)}")
        self.logger.info(f"Limit per type: {limit_per_type}")
        
        for idx, attraction_type in enumerate(attraction_types, 1):
            self.logger.info(f"[{idx}/{len(attraction_types)}] Searching: {attraction_type}")
            
            try:
                # Use existing collect_posts method to search
                places = await self.collect_posts(
                    keywords=[attraction_type],
                    location=main_city,
                    limit=limit_per_type
                )
                
                self.logger.info(f"   [OK] Found {len(places)} places")
                
                # Parse and deduplicate
                for place in places:
                    place_id = place.get('post_id')  # Google place_id
                    
                    if not place_id or place_id in discovered_place_ids:
                        continue
                    
                    discovered_place_ids.add(place_id)
                    
                    # Convert to attraction format
                    attraction_data = {
                        'name': place.get('author', 'Unknown Place'),  # author = place name
                        'province_name': province_name,
                        'address': place.get('address', ''),
                        'description': place.get('text') or f"{attraction_type.title()} in {main_city}, {province_name}",
                        'google_place_id': place_id,
                        'rating': place.get('rating', 0.0) if place.get('rating') else 0.0,
                        'phone': place.get('phone', ''),
                        'website': place.get('website', ''),
                        'category': attraction_type,
                        'source': 'google_maps_auto_discovery'
                    }
                    
                    all_attractions.append(attraction_data)
                    
            except Exception as e:
                self.logger.error(f"   [ERROR] Error searching {attraction_type}: {str(e)}")
                continue
        
        self.logger.info(f"[SUCCESS] Discovery complete: {len(all_attractions)} unique attractions found")
        return all_attractions


def create_google_maps_collector(apify_api_token: str) -> GoogleMapsApifyCollector:
    """Factory function to create Google Maps collector with Apify"""
    collector = GoogleMapsApifyCollector(apify_api_token)
    if collector.authenticate():
        return collector
    raise RuntimeError("Failed to authenticate Google Maps collector")
