"""
TikTok Collector using Apify
Apify provides scrapers that work without TikTok API approval
"""
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


class TikTokApifyCollector(BaseCollector):
    """
    TikTok collector using Apify scraper
    
    Apify advantages:
    - No TikTok API approval needed (saves 2-4 weeks)
    - Can scrape public videos and hashtags
    - More reliable than TikTok API for public data
    - Handles rate limits and CAPTCHAs
    """
    
    def __init__(self, apify_api_token: str):
        super().__init__("tiktok")
        self.apify_token = apify_api_token
        self.client = None
        
    def authenticate(self, **credentials) -> bool:
        """
        Initialize Apify client
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client = ApifyClient(self.apify_token)
            # Test the token
            user_info = self.client.user().get()
            if user_info:
                self.logger.info(f"Apify authenticated for user: {user_info.get('username', 'Unknown')}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Apify authentication failed: {str(e)}")
            return False
    
    async def collect_posts(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Collect TikTok videos using Apify scraper
        
        Uses: apify/tiktok-scraper or clockworks/tiktok-scraper
        
        Args:
            keywords: List of hashtags or usernames to scrape
            location: Location filter (optional, limited support)
            limit: Maximum number of videos to collect
            
        Returns:
            List of video data
        """
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
            
        all_videos = []
        
        try:
            # Using clockworks/tiktok-scraper (more reliable)
            # Alternative: apify/tiktok-scraper
            actor_id = "clockworks/tiktok-scraper"
            
            # Build search queries - IMPROVED for better results
            # Use short, popular hashtags and Vietnamese terms
            search_queries = []
            for keyword in keywords:
                # Convert attraction names to TikTok-friendly hashtags
                if keyword.startswith('#'):
                    # Already a hashtag
                    search_queries.append({
                        "type": "hashtag",
                        "input": keyword[1:]  # Remove #
                    })
                elif keyword.startswith('@'):
                    # Profile/username
                    search_queries.append({
                        "type": "profile",
                        "input": keyword[1:]  # Remove @
                    })
                else:
                    # Convert to hashtag (remove spaces, special chars)
                    # E.g., "Bà Nà Hills" → "banahills"
                    hashtag = keyword.lower()\
                        .replace(' ', '')\
                        .replace('à', 'a').replace('á', 'a').replace('ả', 'a').replace('ã', 'a').replace('ạ', 'a')\
                        .replace('ă', 'a').replace('ằ', 'a').replace('ắ', 'a').replace('ẳ', 'a').replace('ẵ', 'a').replace('ặ', 'a')\
                        .replace('â', 'a').replace('ầ', 'a').replace('ấ', 'a').replace('ẩ', 'a').replace('ẫ', 'a').replace('ậ', 'a')\
                        .replace('đ', 'd')\
                        .replace('è', 'e').replace('é', 'e').replace('ẻ', 'e').replace('ẽ', 'e').replace('ẹ', 'e')\
                        .replace('ê', 'e').replace('ề', 'e').replace('ế', 'e').replace('ể', 'e').replace('ễ', 'e').replace('ệ', 'e')\
                        .replace('ì', 'i').replace('í', 'i').replace('ỉ', 'i').replace('ĩ', 'i').replace('ị', 'i')\
                        .replace('ò', 'o').replace('ó', 'o').replace('ỏ', 'o').replace('õ', 'o').replace('ọ', 'o')\
                        .replace('ô', 'o').replace('ồ', 'o').replace('ố', 'o').replace('ổ', 'o').replace('ỗ', 'o').replace('ộ', 'o')\
                        .replace('ơ', 'o').replace('ờ', 'o').replace('ớ', 'o').replace('ở', 'o').replace('ỡ', 'o').replace('ợ', 'o')\
                        .replace('ù', 'u').replace('ú', 'u').replace('ủ', 'u').replace('ũ', 'u').replace('ụ', 'u')\
                        .replace('ư', 'u').replace('ừ', 'u').replace('ứ', 'u').replace('ử', 'u').replace('ữ', 'u').replace('ự', 'u')\
                        .replace('ỳ', 'y').replace('ý', 'y').replace('ỷ', 'y').replace('ỹ', 'y').replace('ỵ', 'y')
                    
                    # Remove non-alphanumeric
                    hashtag = ''.join(c for c in hashtag if c.isalnum())
                    
                    if hashtag:
                        search_queries.append({
                            "type": "hashtag",
                            "input": hashtag
                        })
                        self.logger.info(f"📌 TikTok hashtag: #{hashtag} (from: {keyword})")
                    search_queries.append({
                        "type": "hashtag",
                        "input": keyword.replace(' ', '').lower()
                    })
            
            # Run the actor
            run_input = {
                "searchQueries": search_queries[:5],  # Limit to 5 queries
                "resultsPerPage": min(limit, 100),
                "maxProfilesPerQuery": 1,
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False,
                "shouldDownloadSubtitles": False
            }
            
            self.logger.info(f"Starting Apify actor {actor_id} for {len(search_queries)} queries...")
            run = self.client.actor(actor_id).call(run_input=run_input)
            
            # Fetch results
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                all_videos.append({
                    'video_id': item.get('id'),
                    'url': item.get('webVideoUrl') or f"https://www.tiktok.com/@{item.get('authorMeta', {}).get('name', 'unknown')}/video/{item.get('id')}",
                    'description': item.get('text') or item.get('description', ''),
                    'username': item.get('authorMeta', {}).get('name', 'Unknown'),
                    'author_id': item.get('authorMeta', {}).get('id', ''),
                    'create_time': item.get('createTime') or item.get('createTimeISO'),
                    'like_count': item.get('diggCount', 0),
                    'comment_count': item.get('commentCount', 0),
                    'share_count': item.get('shareCount', 0),
                    'play_count': item.get('playCount', 0),
                    'hashtags': [tag.get('name', '') for tag in item.get('hashtags', [])],
                    'duration': item.get('videoMeta', {}).get('duration', 0),
                    'music': item.get('musicMeta', {}).get('musicName', ''),
                    'keywords': keywords
                })
                
                if len(all_videos) >= limit:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error collecting TikTok videos via Apify: {str(e)}")
            
        self.logger.info(f"Collected {len(all_videos)} videos from TikTok via Apify")
        return all_videos[:limit]
    
    async def collect_comments(
        self, 
        video_url: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect comments for a specific TikTok video using Apify
        
        Note: Comment scraping is limited on TikTok due to anti-bot measures
        
        Args:
            video_url: TikTok video URL
            limit: Maximum number of comments to collect
            
        Returns:
            List of comment data
        """
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
            
        all_comments = []
        
        try:
            # TikTok comments scraper
            actor_id = "clockworks/tiktok-scraper"
            
            run_input = {
                "searchQueries": [{
                    "type": "video",
                    "input": video_url
                }],
                "resultsPerPage": min(limit, 100),
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False
            }
            
            self.logger.info(f"Collecting comments for video: {video_url}")
            run = self.client.actor(actor_id).call(run_input=run_input)
            
            # Fetch results - TikTok API might include comments in video data
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                # Check if comments are included
                if item.get('comments'):
                    for comment in item['comments']:
                        all_comments.append({
                            'comment_id': comment.get('id') or comment.get('cid'),
                            'video_id': item.get('id'),
                            'text': comment.get('text', ''),
                            'author_name': comment.get('user', {}).get('uniqueId', 'Unknown'),
                            'author_id': comment.get('user', {}).get('id', ''),
                            'create_time': comment.get('createTime'),
                            'like_count': comment.get('diggCount', 0),
                            'reply_count': comment.get('replyCommentTotal', 0)
                        })
                        
                        if len(all_comments) >= limit:
                            break
                            
        except Exception as e:
            self.logger.error(f"Error collecting TikTok comments via Apify: {str(e)}")
            self.logger.warning("Note: TikTok comment scraping is limited. Consider focusing on video data.")
            
        self.logger.info(f"Collected {len(all_comments)} comments via Apify")
        return all_comments[:limit]
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        """
        Convert Apify TikTok video data to SocialPostCreate schema
        """
        # Parse create time
        created_time = None
        if raw_post.get('create_time'):
            try:
                if isinstance(raw_post['create_time'], str):
                    created_time = datetime.fromisoformat(raw_post['create_time'].replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(raw_post['create_time'])
            except Exception:
                created_time = datetime.utcnow()
        
        # Build content from description and hashtags
        content = raw_post.get('description', '')
        if raw_post.get('hashtags'):
            hashtags_str = ' '.join([f"#{tag}" for tag in raw_post['hashtags']])
            content = f"{content}\n\n{hashtags_str}"
        
        return SocialPostCreate(
            platform=PlatformEnum.TIKTOK,
            platform_post_id=str(raw_post.get('video_id', '')),
            attraction_id=attraction_id,
            content=self._clean_text(content),
            author=raw_post.get('username', 'Unknown'),
            author_id=raw_post.get('author_id', ''),
            post_date=created_time or datetime.utcnow(),
            post_url=raw_post.get('url', '')
        )
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int) -> CommentCreate:
        """
        Convert Apify TikTok comment data to CommentCreate schema
        """
        # Parse create time
        created_time = None
        if raw_comment.get('create_time'):
            try:
                if isinstance(raw_comment['create_time'], str):
                    created_time = datetime.fromisoformat(raw_comment['create_time'].replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(raw_comment['create_time'])
            except Exception:
                created_time = datetime.utcnow()
        
        return CommentCreate(
            platform=PlatformEnum.TIKTOK,
            platform_comment_id=str(raw_comment.get('comment_id', '')),
            post_id=post_id,
            attraction_id=raw_comment.get('attraction_id', 0),
            content=self._clean_text(raw_comment.get('text', '')),
            author=raw_comment.get('author_name', 'Unknown'),
            author_id=raw_comment.get('author_id', ''),
            comment_date=created_time or datetime.utcnow(),
            like_count=raw_comment.get('like_count', 0),
            reply_count=raw_comment.get('reply_count', 0)
        )


# Factory function
def create_tiktok_apify_collector(apify_token: str) -> TikTokApifyCollector:
    """
    Create and authenticate a TikTok Apify collector
    
    Args:
        apify_token: Apify API token
        
    Returns:
        Authenticated TikTokApifyCollector instance
    """
    collector = TikTokApifyCollector(apify_token)
    
    if not collector.authenticate():
        logging.warning("TikTok Apify collector created but authentication failed")
        
    return collector
