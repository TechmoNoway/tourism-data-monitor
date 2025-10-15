"""
Facebook Graph API Collector for tourism-related content
Note: Facebook API has strict limitations and requires app review for many endpoints
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    import requests
except ImportError:
    logging.warning("Requests library not installed. Run: pip install requests")

from app.collectors.base_collector import BaseCollector
from app.schemas.post import SocialPostCreate
from app.schemas.comment import CommentCreate


class FacebookCollector(BaseCollector):
    """
    Collector for Facebook posts and comments about tourist attractions
    Note: Limited by Facebook API permissions and policies
    """
    
    def __init__(self, access_token: str, app_id: str, app_secret: str):
        super().__init__("facebook")
        self.access_token = access_token
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "https://graph.facebook.com/v24.0"
        
    def authenticate(self, **credentials) -> bool:
        """
        Verify Facebook access token
        
        Returns:
            True if token is valid, False otherwise
        """
        try:
            # Test the access token by getting basic info
            url = f"{self.base_url}/me"
            params = {
                'access_token': self.access_token,
                'fields': 'id,name'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                user_info = response.json()
                self.logger.info(f"Facebook API authentication successful for user: {user_info.get('name', 'Unknown')}")
                return True
            else:
                self.logger.error(f"Facebook API authentication failed: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Facebook API authentication error: {str(e)}")
            return False
    
    async def collect_posts(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Collect Facebook posts from public pages
        Note: Very limited due to Facebook API restrictions
        
        Args:
            keywords: List of page names or search terms
            location: Location filter (limited support)
            limit: Maximum number of posts to collect
            
        Returns:
            List of post data
        """
        all_posts = []
        
        # Note: Facebook's search API is heavily restricted
        # This implementation focuses on getting posts from known tourism pages
        
        tourism_pages = [
            # Vietnam tourism pages (you need to get page IDs)
            # These are examples - you need actual page IDs and permissions
            "vietnamtourism",
            "dalattourism", 
            "danangtourism",
            "binhthuan.tourism"
        ]
        
        for page_name in tourism_pages:
            try:
                # Get page posts (requires page_posts permission)
                posts = await self._get_page_posts(page_name, limit // len(tourism_pages))
                
                # Filter posts by keywords
                filtered_posts = []
                for post in posts:
                    post_text = (post.get('message', '') + ' ' + post.get('description', '')).lower()
                    if any(keyword.lower() in post_text for keyword in keywords):
                        filtered_posts.append(post)
                
                all_posts.extend(filtered_posts)
                
            except Exception as e:
                self.logger.error(f"Error collecting posts from page {page_name}: {str(e)}")
                continue
                
        self.logger.info(f"Collected {len(all_posts)} posts from Facebook")
        return all_posts[:limit]
    
    async def _get_page_posts(self, page_id: str, limit: int) -> List[Dict[str, Any]]:
        """
        Get posts from a specific Facebook page
        
        Args:
            page_id: Facebook page ID or username
            limit: Maximum number of posts to get
            
        Returns:
            List of post data
        """
        try:
            url = f"{self.base_url}/{page_id}/posts"
            params = {
                'access_token': self.access_token,
                'fields': 'id,message,description,created_time,updated_time,type,link,picture,full_picture,reactions.summary(true),comments.summary(true),shares',
                'limit': min(limit, 100)  # Facebook API limit
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                posts = []
                
                for post in data.get('data', []):
                    # Process post data
                    processed_post = {
                        'post_id': post['id'],
                        'page_id': page_id,
                        'message': post.get('message', ''),
                        'description': post.get('description', ''),
                        'created_time': post['created_time'],
                        'updated_time': post.get('updated_time', post['created_time']),
                        'type': post.get('type', 'status'),
                        'link': post.get('link', ''),
                        'picture': post.get('picture', ''),
                        'full_picture': post.get('full_picture', ''),
                        'reactions_count': post.get('reactions', {}).get('summary', {}).get('total_count', 0),
                        'comments_count': post.get('comments', {}).get('summary', {}).get('total_count', 0),
                        'shares_count': post.get('shares', {}).get('count', 0),
                        'url': f"https://www.facebook.com/{post['id']}"
                    }
                    
                    posts.append(processed_post)
                    
                return posts
                
            else:
                self.logger.error(f"Failed to get posts from page {page_id}: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting posts from page {page_id}: {str(e)}")
            return []
    
    async def collect_comments(
        self, 
        post_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect comments for a specific Facebook post
        
        Args:
            post_id: Facebook post ID
            limit: Maximum number of comments to collect
            
        Returns:
            List of comment data
        """
        all_comments = []
        
        try:
            url = f"{self.base_url}/{post_id}/comments"
            params = {
                'access_token': self.access_token,
                'fields': 'id,message,created_time,from,like_count,comment_count,parent',
                'limit': min(limit, 100)
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                for comment in data.get('data', []):
                    comment_data = {
                        'comment_id': comment['id'],
                        'post_id': post_id,
                        'message': comment.get('message', ''),
                        'created_time': comment['created_time'],
                        'author_name': comment.get('from', {}).get('name', 'Unknown'),
                        'author_id': comment.get('from', {}).get('id', ''),
                        'like_count': comment.get('like_count', 0),
                        'reply_count': comment.get('comment_count', 0),
                        'parent_id': comment.get('parent', {}).get('id') if comment.get('parent') else None
                    }
                    
                    all_comments.append(comment_data)
                    
                    # Get replies if available
                    if comment.get('comment_count', 0) > 0 and len(all_comments) < limit:
                        replies = await self._get_comment_replies(comment['id'], limit - len(all_comments))
                        all_comments.extend(replies)
                
            else:
                self.logger.error(f"Failed to get comments for post {post_id}: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error collecting comments for post {post_id}: {str(e)}")
            
        self.logger.info(f"Collected {len(all_comments)} comments for post {post_id}")
        return all_comments[:limit]
    
    async def _get_comment_replies(self, comment_id: str, limit: int) -> List[Dict[str, Any]]:
        """
        Get replies to a specific comment
        
        Args:
            comment_id: Facebook comment ID
            limit: Maximum number of replies to get
            
        Returns:
            List of reply data
        """
        try:
            url = f"{self.base_url}/{comment_id}/comments"
            params = {
                'access_token': self.access_token,
                'fields': 'id,message,created_time,from,like_count',
                'limit': min(limit, 50)
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                replies = []
                
                for reply in data.get('data', []):
                    reply_data = {
                        'comment_id': reply['id'],
                        'post_id': None,  # Will be set by parent
                        'message': reply.get('message', ''),
                        'created_time': reply['created_time'],
                        'author_name': reply.get('from', {}).get('name', 'Unknown'),
                        'author_id': reply.get('from', {}).get('id', ''),
                        'like_count': reply.get('like_count', 0),
                        'reply_count': 0,
                        'parent_id': comment_id
                    }
                    
                    replies.append(reply_data)
                    
                return replies
                
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting replies for comment {comment_id}: {str(e)}")
            return []
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        """
        Convert Facebook post data to SocialPostCreate schema
        """
        # Parse Facebook datetime format
        created_time = datetime.fromisoformat(raw_post['created_time'].replace('T', ' ').replace('+0000', ''))
        
        # Combine message and description
        content_parts = []
        if raw_post.get('message'):
            content_parts.append(raw_post['message'])
        if raw_post.get('description'):
            content_parts.append(raw_post['description'])
            
        content = '\n\n'.join(content_parts)
        
        return SocialPostCreate(
            attraction_id=attraction_id,
            platform="facebook",
            platform_post_id=raw_post['post_id'],
            author_name=raw_post.get('page_id', 'Unknown Page'),
            author_id=raw_post.get('page_id', ''),
            content=self._clean_text(content),
            posted_at=created_time,
            url=raw_post['url'],
            likes_count=raw_post['reactions_count'],
            comments_count=raw_post['comments_count'],
            shares_count=raw_post['shares_count'],
            views_count=0,  # Not available via API
            language=self._detect_language(content),
            metadata={
                'post_type': raw_post.get('type', 'status'),
                'link': raw_post.get('link', ''),
                'picture': raw_post.get('picture', ''),
                'full_picture': raw_post.get('full_picture', ''),
                'updated_time': raw_post.get('updated_time')
            }
        )
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int) -> CommentCreate:
        """
        Convert Facebook comment data to CommentCreate schema
        """
        # Parse Facebook datetime format
        created_time = datetime.fromisoformat(raw_comment['created_time'].replace('T', ' ').replace('+0000', ''))
        
        return CommentCreate(
            post_id=post_id,
            platform="facebook",
            platform_comment_id=raw_comment['comment_id'],
            author_name=raw_comment['author_name'],
            author_id=raw_comment['author_id'],
            content=self._clean_text(raw_comment['message']),
            posted_at=created_time,
            likes_count=raw_comment['like_count'],
            language=self._detect_language(raw_comment['message']),
            parent_comment_id=raw_comment.get('parent_id'),
            metadata={
                'reply_count': raw_comment.get('reply_count', 0)
            }
        )


# Factory function for easy instantiation
def create_facebook_collector(access_token: str, app_id: str, app_secret: str) -> FacebookCollector:
    """
    Create and authenticate a Facebook collector
    
    Args:
        access_token: Facebook access token
        app_id: Facebook app ID
        app_secret: Facebook app secret
        
    Returns:
        Authenticated FacebookCollector instance
    """
    collector = FacebookCollector(access_token, app_id, app_secret)
    
    if not collector.authenticate():
        raise RuntimeError("Failed to authenticate Facebook collector")
        
    return collector