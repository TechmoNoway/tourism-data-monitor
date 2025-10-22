"""
Facebook Collector using Apify
Apify provides a robust scraper that bypasses Facebook API limitations
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


class FacebookApifyCollector(BaseCollector):
    """
    Facebook collector using Apify scraper
    
    Apify advantages:
    - No Facebook API approval needed
    - Can scrape public posts and pages
    - More reliable than Graph API for public data
    - Handles pagination and rate limits automatically
    """
    
    def __init__(self, apify_api_token: str):
        super().__init__("facebook")
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
            # Test the token by getting user info
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
        Collect Facebook posts using Apify scraper
        
        Uses: apify/facebook-pages-scraper or apify/facebook-posts-scraper
        
        Args:
            keywords: List of page URLs or search queries
            location: Location filter (optional)
            limit: Maximum number of posts to collect
            
        Returns:
            List of post data
        """
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
            
        all_posts = []
        
        try:
            # Apify Facebook Pages Scraper
            # Actor ID: apify/facebook-pages-scraper
            actor_id = "apify/facebook-pages-scraper"
            
            # Build page URLs from keywords
            # You can provide direct Facebook page URLs or search queries
            page_urls = []
            for keyword in keywords:
                # If keyword looks like a URL, use it directly
                if keyword.startswith("http"):
                    page_urls.append(keyword)
                else:
                    # Convert keyword to potential page URL
                    # Example: "Dalat Tourism" -> search or known page
                    # You might need to manually map keywords to page URLs
                    page_urls.append(f"https://www.facebook.com/search/posts/?q={keyword.replace(' ', '%20')}")
            
            # Run the actor
            run_input = {
                "startUrls": [{"url": url} for url in page_urls[:5]],  # Limit to avoid long runs
                "maxPosts": limit,
                "resultsLimit": limit,
                "scrapeAbout": True,
                "scrapeReviews": False,
                "scrapeServices": False,
                "scrapePosts": True
            }
            
            self.logger.info(f"Starting Apify actor {actor_id} for {len(page_urls)} URLs...")
            run = self.client.actor(actor_id).call(run_input=run_input)
            
            # Fetch results
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                # Process each post
                if item.get('posts'):
                    for post in item['posts']:
                        all_posts.append({
                            'post_id': post.get('postId') or post.get('id'),
                            'url': post.get('postUrl'),
                            'text': post.get('text') or post.get('message', ''),
                            'author': post.get('pageTitle') or post.get('pageName', ''),
                            'author_id': post.get('pageId', ''),
                            'created_time': post.get('time') or post.get('createdTime'),
                            'like_count': post.get('likes', 0),
                            'comment_count': post.get('comments', 0),
                            'share_count': post.get('shares', 0),
                            'post_type': post.get('type', 'status'),
                            'media': post.get('images', []) or post.get('video'),
                            'keywords': keywords
                        })
                        
                        if len(all_posts) >= limit:
                            break
                
                if len(all_posts) >= limit:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error collecting Facebook posts via Apify: {str(e)}")
            
        self.logger.info(f"Collected {len(all_posts)} posts from Facebook via Apify")
        return all_posts[:limit]
    
    async def collect_comments(
        self, 
        post_url: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect comments for a specific Facebook post using Apify
        
        Args:
            post_url: Facebook post URL (not just ID)
            limit: Maximum number of comments to collect
            
        Returns:
            List of comment data
        """
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
            
        all_comments = []
        
        try:
            # Apify Facebook Comments Scraper
            actor_id = "apify/facebook-comments-scraper"
            
            run_input = {
                "startUrls": [{"url": post_url}],
                "maxComments": limit,
                "scrapeReplies": True,
                "maxReplies": 10  # Limit replies per comment
            }
            
            self.logger.info(f"Collecting comments for post: {post_url}")
            run = self.client.actor(actor_id).call(run_input=run_input)
            
            # Fetch results
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                all_comments.append({
                    'comment_id': item.get('commentId') or item.get('id'),
                    'post_id': item.get('postId'),
                    'text': item.get('text') or item.get('message', ''),
                    'author_name': item.get('authorName') or item.get('from', {}).get('name', 'Unknown'),
                    'author_id': item.get('authorId') or item.get('from', {}).get('id', ''),
                    'created_time': item.get('createdTime') or item.get('time'),
                    'like_count': item.get('likes', 0),
                    'reply_count': item.get('replies', 0),
                    'parent_id': item.get('parentId')
                })
                
                if len(all_comments) >= limit:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error collecting comments via Apify: {str(e)}")
            
        self.logger.info(f"Collected {len(all_comments)} comments via Apify")
        return all_comments[:limit]
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        """
        Convert Apify Facebook post data to SocialPostCreate schema
        """
        # Parse created time
        created_time = None
        if raw_post.get('created_time'):
            try:
                if isinstance(raw_post['created_time'], str):
                    created_time = datetime.fromisoformat(raw_post['created_time'].replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(raw_post['created_time'])
            except Exception:
                created_time = datetime.utcnow()
        
        return SocialPostCreate(
            platform=PlatformEnum.FACEBOOK,
            platform_post_id=str(raw_post.get('post_id', '')),
            attraction_id=attraction_id,
            content=self._clean_text(raw_post.get('text', '')),
            author=raw_post.get('author', 'Unknown'),
            author_id=raw_post.get('author_id', ''),
            post_date=created_time or datetime.utcnow(),
            post_url=raw_post.get('url', '')
        )
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int) -> CommentCreate:
        """
        Convert Apify Facebook comment data to CommentCreate schema
        """
        # Parse created time
        created_time = None
        if raw_comment.get('created_time'):
            try:
                if isinstance(raw_comment['created_time'], str):
                    created_time = datetime.fromisoformat(raw_comment['created_time'].replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(raw_comment['created_time'])
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
def create_facebook_apify_collector(apify_token: str) -> FacebookApifyCollector:
    """
    Create and authenticate a Facebook Apify collector
    
    Args:
        apify_token: Apify API token
        
    Returns:
        Authenticated FacebookApifyCollector instance
    """
    collector = FacebookApifyCollector(apify_token)
    
    if not collector.authenticate():
        logging.warning("Facebook Apify collector created but authentication failed")
        
    return collector
