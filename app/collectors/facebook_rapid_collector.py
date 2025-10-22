"""
Facebook Collector using RapidAPI
Alternative to Apify for Facebook data scraping
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import requests

from app.collectors.base_collector import BaseCollector
from app.schemas.post import SocialPostCreate, PlatformEnum
from app.schemas.comment import CommentCreate


class FacebookRapidCollector(BaseCollector):
    """
    Facebook collector using RapidAPI
    
    RapidAPI Endpoint Options:
    1. "facebook-data-scraper" API
    2. "facebook-api-scraper" API
    3. "facebook-scraper2" API
    
    Features:
    - Scrape posts from public pages
    - Get comments from posts
    - No Facebook API approval needed
    - Pay-per-use pricing
    """
    
    # RapidAPI endpoints
    RAPID_API_HOST = "facebook-data-scraper.p.rapidapi.com"
    RAPID_API_BASE = f"https://{RAPID_API_HOST}"
    
    def __init__(self, rapid_api_key: str):
        super().__init__("facebook")
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
            # Check if we got rate limit error (means key is valid)
            if response.status_code in [200, 429]:
                self.logger.info("RapidAPI key verified for Facebook")
                return True
            elif response.status_code == 403:
                self.logger.error("RapidAPI key is invalid or doesn't have access")
                return False
            else:
                # Assume valid if we get any response
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
        Collect Facebook posts using RapidAPI
        
        Args:
            keywords: List of page names or search queries
            location: Location filter (optional)
            limit: Maximum number of posts to collect
            
        Returns:
            List of post data
        """
        all_posts = []
        
        try:
            for keyword in keywords:
                if len(all_posts) >= limit:
                    break
                
                self.logger.info(f"Searching Facebook for: {keyword}")
                
                # RapidAPI endpoint for Facebook page posts
                # Note: Actual endpoint varies by chosen API
                url = f"{self.RAPID_API_BASE}/page/posts"
                
                params = {
                    "page": keyword,
                    "limit": min(limit - len(all_posts), 50)
                }
                
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Parse response (structure depends on API)
                    posts = data.get('data', []) or data.get('posts', [])
                    
                    for post in posts:
                        all_posts.append({
                            'post_id': post.get('id') or post.get('post_id'),
                            'page_id': post.get('page_id') or post.get('from', {}).get('id'),
                            'page_name': post.get('page_name') or post.get('from', {}).get('name'),
                            'message': post.get('message') or post.get('text', ''),
                            'created_time': post.get('created_time') or post.get('timestamp'),
                            'post_url': post.get('permalink_url') or f"https://facebook.com/{post.get('id')}",
                            'like_count': post.get('like_count', 0),
                            'comment_count': post.get('comment_count', 0),
                            'share_count': post.get('share_count', 0),
                            'post_type': post.get('type', 'status'),
                            'keywords': keywords
                        })
                        
                        if len(all_posts) >= limit:
                            break
                            
                elif response.status_code == 429:
                    self.logger.warning("RapidAPI rate limit reached")
                    break
                else:
                    self.logger.error(f"RapidAPI error: {response.status_code} - {response.text[:200]}")
                    
        except Exception as e:
            self.logger.error(f"Error collecting Facebook posts via RapidAPI: {str(e)}")
            
        self.logger.info(f"Collected {len(all_posts)} posts from Facebook via RapidAPI")
        return all_posts[:limit]
    
    async def collect_comments(
        self, 
        post_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect comments for a specific Facebook post using RapidAPI
        
        Args:
            post_id: Facebook post ID
            limit: Maximum number of comments to collect
            
        Returns:
            List of comment data
        """
        all_comments = []
        
        try:
            self.logger.info(f"Collecting comments for post: {post_id}")
            
            # RapidAPI endpoint for post comments
            url = f"{self.RAPID_API_BASE}/post/comments"
            
            params = {
                "post_id": post_id,
                "limit": min(limit, 100)
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
                    all_comments.append({
                        'comment_id': comment.get('id') or comment.get('comment_id'),
                        'post_id': post_id,
                        'text': comment.get('message') or comment.get('text', ''),
                        'author_name': comment.get('from', {}).get('name', 'Unknown'),
                        'author_id': comment.get('from', {}).get('id', ''),
                        'created_time': comment.get('created_time') or comment.get('timestamp'),
                        'like_count': comment.get('like_count', 0),
                        'reply_count': comment.get('comment_count', 0)
                    })
                    
                    if len(all_comments) >= limit:
                        break
                        
            elif response.status_code == 429:
                self.logger.warning("RapidAPI rate limit reached")
            else:
                self.logger.error(f"RapidAPI error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error collecting Facebook comments via RapidAPI: {str(e)}")
            
        self.logger.info(f"Collected {len(all_comments)} comments via RapidAPI")
        return all_comments[:limit]
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        """
        Convert RapidAPI Facebook post data to SocialPostCreate schema
        """
        # Parse created time
        created_time = None
        if raw_post.get('created_time'):
            try:
                # Facebook ISO format or timestamp
                time_str = raw_post['created_time']
                if isinstance(time_str, str):
                    created_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(time_str)
            except Exception:
                created_time = datetime.utcnow()
        
        return SocialPostCreate(
            platform=PlatformEnum.FACEBOOK,
            platform_post_id=str(raw_post.get('post_id', '')),
            attraction_id=attraction_id,
            content=self._clean_text(raw_post.get('message', '')),
            author=raw_post.get('page_name', 'Unknown'),
            author_id=raw_post.get('page_id', ''),
            post_date=created_time or datetime.utcnow(),
            post_url=raw_post.get('post_url', ''),
            like_count=raw_post.get('like_count', 0),
            comment_count=raw_post.get('comment_count', 0),
            share_count=raw_post.get('share_count', 0)
        )
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int) -> CommentCreate:
        """
        Convert RapidAPI Facebook comment data to CommentCreate schema
        """
        # Parse created time
        created_time = None
        if raw_comment.get('created_time'):
            try:
                time_str = raw_comment['created_time']
                if isinstance(time_str, str):
                    created_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(time_str)
            except Exception:
                created_time = datetime.utcnow()
        
        return CommentCreate(
            platform=PlatformEnum.FACEBOOK,
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
def create_facebook_rapid_collector(rapid_api_key: str) -> FacebookRapidCollector:
    """
    Create and authenticate a Facebook RapidAPI collector
    
    Args:
        rapid_api_key: RapidAPI key with Facebook API access
        
    Returns:
        Authenticated FacebookRapidCollector instance
    """
    collector = FacebookRapidCollector(rapid_api_key)
    
    if not collector.authenticate():
        logging.warning("Facebook RapidAPI collector created but authentication failed")
        
    return collector
