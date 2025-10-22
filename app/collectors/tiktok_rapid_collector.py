"""
TikTok Collector using RapidAPI
Alternative to Apify for TikTok data scraping
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import requests

from app.collectors.base_collector import BaseCollector
from app.schemas.post import SocialPostCreate, PlatformEnum
from app.schemas.comment import CommentCreate


class TikTokRapidCollector(BaseCollector):
    """
    TikTok collector using RapidAPI
    
    RapidAPI Endpoint Options:
    1. "tiktok-scraper7" API
    2. "tiktok-video-no-watermark2" API
    3. "tokapi-mobile-version" API
    
    Features:
    - Search videos by hashtags
    - Get user videos
    - Download video info and comments
    - No TikTok API approval needed
    """
    
    # RapidAPI endpoints (using tiktok-scraper7 as example)
    RAPID_API_HOST = "tiktok-scraper7.p.rapidapi.com"
    RAPID_API_BASE = f"https://{RAPID_API_HOST}"
    
    def __init__(self, rapid_api_key: str):
        super().__init__("tiktok")
        self.rapid_api_key = rapid_api_key
        self.headers = {
            "X-RapidAPI-Key": rapid_api_key,
            "X-RapidAPI-Host": self.RAPID_API_HOST
        }
        
    def authenticate(self, **credentials) -> bool:
        """
        Verify RapidAPI key by making a test request
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Test with a simple request
            response = requests.get(
                f"{self.RAPID_API_BASE}/test",
                headers=self.headers,
                timeout=10
            )
            
            # RapidAPI returns 200 even if endpoint doesn't exist
            if response.status_code in [200, 404, 429]:
                self.logger.info("RapidAPI key verified for TikTok")
                return True
            elif response.status_code == 403:
                self.logger.error("RapidAPI key is invalid or doesn't have access")
                return False
            else:
                self.logger.info("RapidAPI key appears valid")
                return True
                
        except Exception as e:
            self.logger.error(f"RapidAPI authentication error: {str(e)}")
            return False
    
    async def collect_posts(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Collect TikTok videos using RapidAPI
        
        Args:
            keywords: List of hashtags or usernames
            location: Location filter (limited support)
            limit: Maximum number of videos to collect
            
        Returns:
            List of video data
        """
        all_videos = []
        
        try:
            for keyword in keywords:
                if len(all_videos) >= limit:
                    break
                
                # Clean keyword (remove # if hashtag)
                clean_keyword = keyword.replace('#', '').replace('@', '')
                
                self.logger.info(f"Searching TikTok for: {keyword}")
                
                # Try hashtag search first
                url = f"{self.RAPID_API_BASE}/hashtag/posts"
                
                params = {
                    "hashtag": clean_keyword,
                    "count": min(limit - len(all_videos), 50)
                }
                
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Parse response (structure varies by API)
                    videos = data.get('data', []) or data.get('itemList', []) or data.get('videos', [])
                    
                    for video in videos:
                        # Handle different response structures
                        video_info = video.get('video', video)
                        author_info = video.get('author', {})
                        stats = video.get('stats', {})
                        
                        all_videos.append({
                            'video_id': video.get('id') or video.get('aweme_id'),
                            'url': video.get('video_url') or f"https://www.tiktok.com/@{author_info.get('uniqueId', 'unknown')}/video/{video.get('id')}",
                            'description': video.get('desc') or video.get('title', ''),
                            'username': author_info.get('uniqueId') or author_info.get('nickname', 'Unknown'),
                            'author_id': author_info.get('id', ''),
                            'create_time': video.get('createTime') or video.get('create_time'),
                            'like_count': stats.get('diggCount', 0),
                            'comment_count': stats.get('commentCount', 0),
                            'share_count': stats.get('shareCount', 0),
                            'play_count': stats.get('playCount', 0),
                            'hashtags': [tag.get('name', '') for tag in video.get('textExtra', []) if tag.get('hashtagName')],
                            'duration': video_info.get('duration', 0),
                            'music': video.get('music', {}).get('title', ''),
                            'keywords': keywords
                        })
                        
                        if len(all_videos) >= limit:
                            break
                            
                elif response.status_code == 429:
                    self.logger.warning("RapidAPI rate limit reached")
                    break
                else:
                    self.logger.error(f"RapidAPI error: {response.status_code} - {response.text[:200]}")
                    
        except Exception as e:
            self.logger.error(f"Error collecting TikTok videos via RapidAPI: {str(e)}")
            
        self.logger.info(f"Collected {len(all_videos)} videos from TikTok via RapidAPI")
        return all_videos[:limit]
    
    async def collect_comments(
        self, 
        video_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect comments for a specific TikTok video using RapidAPI
        
        Args:
            video_id: TikTok video ID or URL
            limit: Maximum number of comments to collect
            
        Returns:
            List of comment data
        """
        all_comments = []
        
        try:
            self.logger.info(f"Collecting comments for video: {video_id}")
            
            # Extract video ID from URL if needed
            clean_video_id = video_id
            if 'tiktok.com' in video_id:
                # Extract ID from URL
                parts = video_id.split('/')
                for i, part in enumerate(parts):
                    if part == 'video' and i + 1 < len(parts):
                        clean_video_id = parts[i + 1].split('?')[0]
                        break
            
            # RapidAPI endpoint for video comments
            url = f"{self.RAPID_API_BASE}/video/comments"
            
            params = {
                "video_id": clean_video_id,
                "count": min(limit, 100)
            }
            
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                comments = data.get('data', []) or data.get('comments', [])
                
                for comment in comments:
                    user_info = comment.get('user', {})
                    
                    all_comments.append({
                        'comment_id': comment.get('cid') or comment.get('id'),
                        'video_id': clean_video_id,
                        'text': comment.get('text', ''),
                        'author_name': user_info.get('uniqueId') or user_info.get('nickname', 'Unknown'),
                        'author_id': user_info.get('id', ''),
                        'create_time': comment.get('create_time') or comment.get('createTime'),
                        'like_count': comment.get('digg_count', 0),
                        'reply_count': comment.get('reply_comment_total', 0)
                    })
                    
                    if len(all_comments) >= limit:
                        break
                        
            elif response.status_code == 429:
                self.logger.warning("RapidAPI rate limit reached")
            else:
                self.logger.error(f"RapidAPI error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error collecting TikTok comments via RapidAPI: {str(e)}")
            
        self.logger.info(f"Collected {len(all_comments)} comments via RapidAPI")
        return all_comments[:limit]
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        """
        Convert RapidAPI TikTok video data to SocialPostCreate schema
        """
        # Parse create time
        created_time = None
        if raw_post.get('create_time'):
            try:
                create_time_value = raw_post['create_time']
                if isinstance(create_time_value, str):
                    created_time = datetime.fromisoformat(create_time_value.replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(create_time_value)
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
        Convert RapidAPI TikTok comment data to CommentCreate schema
        """
        # Parse create time
        created_time = None
        if raw_comment.get('create_time'):
            try:
                create_time_value = raw_comment['create_time']
                if isinstance(create_time_value, str):
                    created_time = datetime.fromisoformat(create_time_value.replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(create_time_value)
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
def create_tiktok_rapid_collector(rapid_api_key: str) -> TikTokRapidCollector:
    """
    Create and authenticate a TikTok RapidAPI collector
    
    Args:
        rapid_api_key: RapidAPI key with TikTok API access
        
    Returns:
        Authenticated TikTokRapidCollector instance
    """
    collector = TikTokRapidCollector(rapid_api_key)
    
    if not collector.authenticate():
        logging.warning("TikTok RapidAPI collector created but authentication failed")
        
    return collector
