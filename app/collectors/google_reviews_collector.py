"""
Google Places/Reviews Collector for tourist attractions
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    import googlemaps
except ImportError:
    logging.warning("Google Maps client not installed. Run: pip install googlemaps")

from app.collectors.base_collector import BaseCollector
from app.schemas.post import SocialPostCreate
from app.schemas.comment import CommentCreate


class GoogleReviewsCollector(BaseCollector):
    """
    Collector for Google Places reviews of tourist attractions
    """
    
    def __init__(self, api_key: str):
        super().__init__("google_reviews")
        self.api_key = api_key
        self.gmaps_client = None
        
    def authenticate(self, **credentials) -> bool:
        """
        Initialize Google Maps client
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.gmaps_client = googlemaps.Client(key=self.api_key)
            # Test the connection
            self.gmaps_client.geocode("Hanoi, Vietnam")
            self.logger.info("Google Maps API authentication successful")
            return True
        except Exception as e:
            self.logger.error(f"Google Maps API authentication failed: {str(e)}")
            return False
    
    async def collect_posts(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for places and collect basic information
        Note: Google Places API treats each place as a "post"
        
        Args:
            keywords: List of place names or types to search for
            location: Location filter (city, province)
            limit: Maximum number of places to collect
            
        Returns:
            List of place data
        """
        if not self.gmaps_client:
            raise RuntimeError("Google Maps client not authenticated")
            
        all_places = []
        
        for keyword in keywords:
            try:
                # Search for places
                search_query = keyword
                if location:
                    search_query += f" {location}"
                
                # Text search for places
                places_result = self.gmaps_client.places(
                    query=search_query,
                    type='tourist_attraction',
                    language='vi',
                    region='vn'
                )
                
                for place in places_result.get('results', []):
                    place_id = place['place_id']
                    
                    # Get detailed place information including reviews
                    place_details = self.gmaps_client.place(
                        place_id=place_id,
                        fields=[
                            'place_id', 'name', 'formatted_address', 'rating', 
                            'user_ratings_total', 'reviews', 'photos', 'geometry',
                            'types', 'website', 'international_phone_number',
                            'opening_hours', 'price_level'
                        ],
                        language='vi'
                    )
                    
                    if 'result' in place_details:
                        place_info = place_details['result']
                        
                        # Get photo URL if available
                        photo_url = None
                        if 'photos' in place_info and place_info['photos']:
                            photo_ref = place_info['photos'][0]['photo_reference']
                            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={self.api_key}"
                        
                        processed_place = {
                            'place_id': place_id,
                            'name': place_info.get('name', ''),
                            'address': place_info.get('formatted_address', ''),
                            'rating': place_info.get('rating', 0.0),
                            'reviews_count': place_info.get('user_ratings_total', 0),
                            'location': place_info.get('geometry', {}).get('location', {}),
                            'types': place_info.get('types', []),
                            'website': place_info.get('website', ''),
                            'phone': place_info.get('international_phone_number', ''),
                            'photo_url': photo_url,
                            'price_level': place_info.get('price_level'),
                            'opening_hours': place_info.get('opening_hours', {}),
                            'reviews': place_info.get('reviews', []),
                            'keywords': [keyword],
                            'url': f"https://maps.google.com/maps/place/?q=place_id:{place_id}"
                        }
                        
                        all_places.append(processed_place)
                        
                        self.logger.info(f"Collected place: {place_info.get('name', '')}")
                        
                        if len(all_places) >= limit:
                            break
                            
            except Exception as e:
                self.logger.error(f"Error searching Google Places for keyword '{keyword}': {str(e)}")
                continue
                
            if len(all_places) >= limit:
                break
                
        self.logger.info(f"Collected {len(all_places)} places from Google Places")
        return all_places[:limit]
    
    async def collect_comments(
        self, 
        place_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect reviews for a specific Google Place
        Note: Google Places API has limited review data
        
        Args:
            place_id: Google Place ID
            limit: Maximum number of reviews to collect (max 5 from API)
            
        Returns:
            List of review data
        """
        if not self.gmaps_client:
            raise RuntimeError("Google Maps client not authenticated")
            
        all_reviews = []
        
        try:
            # Get place details with reviews
            place_details = self.gmaps_client.place(
                place_id=place_id,
                fields=['reviews'],
                language='vi'
            )
            
            if 'result' in place_details and 'reviews' in place_details['result']:
                reviews = place_details['result']['reviews']
                
                for review in reviews[:limit]:  # Google API returns max 5 reviews
                    review_data = {
                        'review_id': f"{place_id}_{review['author_name']}_{review['time']}",
                        'place_id': place_id,
                        'author_name': review['author_name'],
                        'author_url': review.get('author_url', ''),
                        'profile_photo_url': review.get('profile_photo_url', ''),
                        'rating': review['rating'],
                        'text': review.get('text', ''),
                        'time': review['time'],
                        'relative_time_description': review.get('relative_time_description', ''),
                        'language': review.get('language', 'vi')
                    }
                    
                    all_reviews.append(review_data)
                    
        except Exception as e:
            self.logger.error(f"Error collecting reviews for place {place_id}: {str(e)}")
            
        self.logger.info(f"Collected {len(all_reviews)} reviews for place {place_id}")
        return all_reviews
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        """
        Convert Google Place data to SocialPostCreate schema
        Note: Each place is treated as a "post"
        """
        # Use current time as posted_at since Google Places doesn't provide creation date
        posted_at = datetime.utcnow()
        
        # Create content from place information
        content_parts = [f"📍 {raw_post['name']}"]
        
        if raw_post.get('address'):
            content_parts.append(f"🏠 Địa chỉ: {raw_post['address']}")
            
        if raw_post.get('rating'):
            content_parts.append(f"⭐ Đánh giá: {raw_post['rating']}/5 ({raw_post.get('reviews_count', 0)} reviews)")
            
        if raw_post.get('phone'):
            content_parts.append(f"📞 {raw_post['phone']}")
            
        if raw_post.get('website'):
            content_parts.append(f"🌐 {raw_post['website']}")
        
        content = "\n".join(content_parts)
        
        return SocialPostCreate(
            attraction_id=attraction_id,
            platform="google_reviews",
            platform_post_id=raw_post['place_id'],
            author_name="Google Places",
            author_id="google_places",
            content=content,
            posted_at=posted_at,
            url=raw_post['url'],
            likes_count=0,  # Not applicable for places
            comments_count=raw_post.get('reviews_count', 0),
            shares_count=0,  # Not applicable
            views_count=0,  # Not available
            language=self._detect_language(raw_post['name']),
            metadata={
                'rating': raw_post.get('rating'),
                'reviews_count': raw_post.get('reviews_count', 0),
                'location': raw_post.get('location', {}),
                'types': raw_post.get('types', []),
                'photo_url': raw_post.get('photo_url'),
                'price_level': raw_post.get('price_level'),
                'opening_hours': raw_post.get('opening_hours', {}),
                'keywords': raw_post.get('keywords', [])
            }
        )
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int) -> CommentCreate:
        """
        Convert Google Review data to CommentCreate schema
        """
        # Convert Unix timestamp to datetime
        posted_at = datetime.fromtimestamp(raw_comment['time'])
        
        return CommentCreate(
            post_id=post_id,
            platform="google_reviews",
            platform_comment_id=raw_comment['review_id'],
            author_name=raw_comment['author_name'],
            author_id=raw_comment['author_name'],  # Google doesn't provide author ID
            content=self._clean_text(raw_comment.get('text', '')),
            posted_at=posted_at,
            likes_count=0,  # Not available in Google Reviews API
            language=raw_comment.get('language', 'vi'),
            parent_comment_id=None,  # Google Reviews don't have nested comments
            metadata={
                'rating': raw_comment['rating'],
                'author_url': raw_comment.get('author_url', ''),
                'profile_photo_url': raw_comment.get('profile_photo_url', ''),
                'relative_time_description': raw_comment.get('relative_time_description', ''),
                'original_language': raw_comment.get('language', 'vi')
            }
        )


# Factory function for easy instantiation
def create_google_reviews_collector(api_key: str) -> GoogleReviewsCollector:
    """
    Create and authenticate a Google Reviews collector
    
    Args:
        api_key: Google Maps API key
        
    Returns:
        Authenticated GoogleReviewsCollector instance
    """
    collector = GoogleReviewsCollector(api_key)
    
    if not collector.authenticate():
        raise RuntimeError("Failed to authenticate Google Reviews collector")
        
    return collector