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


class FacebookPostsScraper(BaseCollector):
    def __init__(self, apify_api_token: str):
        super().__init__("facebook")
        self.apify_token = apify_api_token
        self.client = None
        
    def authenticate(self, **credentials) -> bool:
        try:
            self.client = ApifyClient(self.apify_token)
            user_info = self.client.user().get()
            if user_info:
                self.logger.info(f"✓ Apify authenticated: {user_info.get('username', 'Unknown')}")
                self.logger.info(f"💰 Balance: ${user_info.get('balance', 0):.2f}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Apify authentication failed: {str(e)}")
            return False
    
    async def collect_posts_with_comments(
        self,
        keywords: List[str],
        location: Optional[str] = None,
        limit: int = 20,
        comments_per_post: int = 50
    ) -> List[Dict[str, Any]]:
        if isinstance(keywords, str):
            self.logger.debug(f"Converting string keywords to list: {keywords}")
            keywords = [keywords]
        
        self.logger.info("🔄 Using 2-actor strategy: Posts + Comments")
        
        posts = await self.collect_posts(keywords=keywords, location=location, limit=limit)
        
        if not posts:
            self.logger.warning("No posts collected, skipping comment collection")
            return posts
        
        post_urls = [post.get("url") for post in posts if post.get("url")]
        
        if not post_urls:
            self.logger.warning("No valid post URLs found, skipping comment collection")
            return posts
        
        self.logger.info(f"📝 Extracted {len(post_urls)} post URLs for comment collection")
        
        comments = await self._scrape_comments_from_urls(post_urls, comments_per_post)
        
        if not comments:
            self.logger.warning("No comments collected")
            for post in posts:
                post["comments"] = []
            return posts
        
        self.logger.info("🔗 Merging comments with posts...")
        
        comments_by_post = {}
        for comment in comments:
            post_url = comment.get("post_url")
            if post_url:
                if post_url not in comments_by_post:
                    comments_by_post[post_url] = []
                comments_by_post[post_url].append(comment)
        
        posts_with_comments = 0
        total_comments = 0
        
        for post in posts:
            post_url = post.get("url")
            if post_url in comments_by_post:
                post["comments"] = comments_by_post[post_url]
                posts_with_comments += 1
                total_comments += len(comments_by_post[post_url])
            else:
                post["comments"] = []
        
        self.logger.info(f"✅ Merged {total_comments} comments into {posts_with_comments}/{len(posts)} posts")
        
        return posts
    
    async def collect_posts(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
        
        if isinstance(keywords, str):
            self.logger.debug(f"Converting string keywords to list: {keywords}")
            keywords = [keywords]
            
        all_posts = []
        
        try:
            page_urls = [k for k in keywords if k.startswith("http")]
            search_keywords = [k for k in keywords if not k.startswith("http")]
            
            # Strategy 1: Scrape from pages 
            if page_urls:
                posts = await self._scrape_from_pages(page_urls, limit)
                all_posts.extend(posts)
                self.logger.info(f"✓ Got {len(posts)} posts from pages")
            
            # Strategy 2: Search if not enough posts
            if len(all_posts) < limit and search_keywords:
                remaining = limit - len(all_posts)
                search_posts = await self._scrape_from_search(search_keywords[:3], remaining)
                all_posts.extend(search_posts)
                self.logger.info(f"✓ Got {len(search_posts)} posts from search")
            
        except Exception as e:
            self.logger.error(f"Error collecting Facebook posts: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return all_posts
    
    async def _scrape_comments_from_urls(self, post_urls: List[str], comments_per_post: int = 50) -> List[Dict[str, Any]]:
        all_comments = []
        
        if not post_urls:
            self.logger.warning("No post URLs provided for comment scraping")
            return all_comments
        
        try:
            actor_id = "apify/facebook-comments-scraper"
            
            self.logger.info(f"💬 Scraping comments from {len(post_urls)} posts using Comments Scraper")
            
            run_input = {
                "startUrls": [{"url": url} for url in post_urls],
                "resultsLimit": comments_per_post,
                "sort": "newest", 
                "maxComments": comments_per_post,
                "proxy": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"]
                }
            }
            
            self.logger.info(f"💰 Running Apify Comments Scraper for {len(post_urls)} posts")
            
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            
            if not dataset_id:
                self.logger.error("❌ No dataset_id returned from Comments Scraper")
                return all_comments
            
            items = self.client.dataset(dataset_id).list_items().items
            self.logger.info(f"💬 Comments Scraper returned {len(items)} comments")
            
            for item in items:
                try:
                    comment_data = {
                        "comment_id": item.get("id") or item.get("feedbackId", ""),
                        "text": item.get("text", ""),
                        "author_name": item.get("profileName", "Unknown"),
                        "author_id": item.get("profileUrl", ""),
                        "like_count": item.get("likesCount", 0),
                        "reply_count": item.get("commentsCount", 0),
                        "published_at": item.get("date"),
                        "comment_url": item.get("commentUrl"),
                        "post_url": item.get("inputUrl") or item.get("facebookUrl"),
                    }
                    all_comments.append(comment_data)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse comment: {e}")
                    continue
            
            self.logger.info(f"✅ Successfully parsed {len(all_comments)} comments")
            
        except Exception as e:
            self.logger.error(f"Error scraping comments: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return all_comments
    
    async def _scrape_from_search(self, search_keywords: List[str], limit: int) -> List[Dict[str, Any]]:
        posts = []
        
        if not search_keywords:
            return posts
        
        try:
            actor_id = "TMBawM4LZpKN15DZX"
            
            query = " ".join(search_keywords)
            
            self.logger.info(f"🔍 Searching Facebook posts with query: '{query}'")
            
            run_input = {
                "query": query,
                "maxPosts": min(limit, 20),
                "searchType": "top"  
            }
            
            self.logger.info("💰 Running Apify Posts Search (keyword-mode)")
            
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            
            if not dataset_id:
                self.logger.error("❌ No dataset_id returned from Posts Search")
                return posts
            
            items = self.client.dataset(dataset_id).list_items().items
            self.logger.info(f"📊 Posts Search returned {len(items)} items")
            
            for i, item in enumerate(items, 1):
                try:
                    post = self._parse_search_post(item)
                    if post:
                        self.logger.info(f"✅ Parsed search result {i}: {post.get('post_id', 'unknown')}")
                        posts.append(post)
                    else:
                        self.logger.warning(f"❌ Failed to parse search result {i}")
                except Exception as e:
                    self.logger.warning(f"Error parsing search result {i}: {e}")
                    continue
            
            self.logger.info(f"✅ Successfully collected {len(posts)} posts from search")
            
        except Exception as e:
            self.logger.error(f"Error in search scraping: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return posts
    
    def _parse_search_post(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse post from facebook-posts-search actor
        Format is different from posts-scraper actor
        """
        try:
            post_id = item.get("postId", "")
            if not post_id:
                self.logger.warning("Search result missing postId")
                return None
            
            timestamp_ms = item.get("timestamp")
            published_at = None
            if timestamp_ms:
                try:
                    published_at = datetime.fromtimestamp(timestamp_ms / 1000)
                except Exception:
                    pass
            
            author = item.get("author", {})
            
            post = {
                "post_id": str(post_id),
                "url": item.get("url", ""),
                "text": item.get("postText", ""),
                "author_name": author.get("name", "Unknown"),
                "author_id": author.get("id", ""),
                "author_url": author.get("profileUrl", ""),
                "like_count": item.get("reactionsCount", 0),
                "comment_count": item.get("commentsCount", 0),
                "published_at": published_at,
                "reactions": item.get("reactions", {}),
                "comments": [],  # Will be filled by comment scraper
                "source": "search"  # Mark as search result
            }
            
            return post
            
        except Exception as e:
            self.logger.error(f"Error parsing search post: {e}")
            return None
    
    async def _scrape_from_pages(self, page_urls: List[str], limit: int) -> List[Dict[str, Any]]:
        posts = []

        if not page_urls:
            return posts

        try:
            actor_id = "apify/facebook-posts-scraper"

            page_urls = page_urls[:3]

            self.logger.info(f"📍 Scraping {len(page_urls)} Facebook pages using Posts Scraper")

            run_input = {
                "startUrls": [{"url": url} for url in page_urls],
                "resultsLimit": min(limit, 50),
                "maxComments": 500, 
                "maxCommentsPerPost": 100,
                "commentsMode": "RANKED_RELEVANT",
                "scrapePostComments": True,
                "scrapeReactions": False,
                "proxyConfiguration": {"useApifyProxy": True},
            }

            self.logger.info("💰 Running Apify Posts Scraper (page-mode)")

            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")

            if not dataset_id:
                self.logger.error("❌ No dataset_id returned from Apify run")
                return posts

            items = self.client.dataset(dataset_id).list_items().items
            self.logger.info(f"📊 Dataset returned {len(items)} items")

            if not items:
                self.logger.warning("⚠️ Dataset is empty! Possible reasons: page private, no posts, or actor couldn't access the page")
                return posts

            for i, item in enumerate(items, 1):
                self.logger.info(f"📝 Processing item {i}/{len(items)}")
                self.logger.info(f"🔑 Raw item keys: {list(item.keys())}")
                if 'postId' in item or 'id' in item:
                    self.logger.info(f"   Post ID: {item.get('postId') or item.get('id')}")
                if 'text' in item:
                    self.logger.info(f"   Text preview: {str(item.get('text'))[:100]}...")
                post = self._parse_apify_post(item)
                if post:
                    self.logger.info(f"✅ Successfully parsed post: {post.get('post_id', 'unknown')}")
                    posts.append(post)
                else:
                    self.logger.warning(f"❌ Failed to parse item {i}")
                    if any(k in item for k in ("pageUrl", "pageName", "facebookUrl", "pageId", "facebookId")):
                        page_meta = {
                            "is_page_meta": True,
                            "page_name": item.get("pageName") or item.get("title") or item.get("page_name"),
                            "page_url": item.get("pageUrl") or item.get("facebookUrl") or item.get("url"),
                            "followers": item.get("followers") or item.get("likes") or item.get("rating") or 0,
                            "intro": item.get("intro") or item.get("info") or item.get("description") or "",
                        }
                        posts.append(page_meta)
                        self.logger.info(f"ℹ️ Page metadata collected for {page_meta.get('page_url')}")
                    else:
                        self.logger.warning(f"Failed to parse item {i}")
                        self.logger.debug(f"Raw item sample: {str(item)[:400]}...")

        except Exception as e:
            msg = str(e)
            if "exceed the memory limit" in msg or "memory limit" in msg:
                self.logger.error("❌ Apify memory limit prevented launching the actor run.")
                self.logger.error("Action needed: free up actor/build memory in your Apify account or upgrade the plan. See: https://console.apify.com/billing/subscription")
                return posts

            self.logger.error(f"Error scraping pages: {msg}")
            import traceback
            self.logger.error(traceback.format_exc())

        return posts
    
    async def _scrape_from_search(self, keywords: List[str], limit: int) -> List[Dict[str, Any]]:
        posts = []
        
        if not keywords:
            return posts
        
        try:
            actor_id = "TMBawM4LZpKN15DZX"  
            
            query = " ".join(keywords)
            
            self.logger.info(f"🔍 Searching Facebook posts with query: '{query}'")
            
            run_input = {
                "query": query,
                "maxPosts": min(limit, 20),
                "searchType": "top" 
            }
            
            self.logger.info("💰 Running Apify Posts Search (keyword-mode)")
            
            run = self.client.actor(actor_id).call(run_input=run_input)
            dataset_id = run.get("defaultDatasetId")
            
            if not dataset_id:
                self.logger.error("❌ No dataset_id returned from Posts Search")
                return posts
            
            items = self.client.dataset(dataset_id).list_items().items
            self.logger.info(f"📊 Posts Search returned {len(items)} items")
            
            for i, item in enumerate(items, 1):
                try:
                    post = self._parse_search_post(item)
                    if post:
                        self.logger.info(f"✅ Parsed search result {i}: {post.get('post_id', 'unknown')}")
                        posts.append(post)
                    else:
                        self.logger.warning(f"❌ Failed to parse search result {i}")
                except Exception as e:
                    self.logger.warning(f"Error parsing search result {i}: {e}")
                    continue
            
            self.logger.info(f"✅ Successfully collected {len(posts)} posts from search")
            
        except Exception as e:
            self.logger.error(f"Error in search scraping: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return posts
    
    def _parse_search_post(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse post from facebook-posts-search actor (TMBawM4LZpKN15DZX)
        Format is different from posts-scraper actor
        """
        try:
            post_id = item.get("postId", "")
            if not post_id:
                self.logger.warning("Search result missing postId")
                return None
            
            # Convert timestamp (milliseconds) to datetime
            timestamp_ms = item.get("timestamp")
            published_at = None
            if timestamp_ms:
                try:
                    published_at = datetime.fromtimestamp(timestamp_ms / 1000)
                except Exception:
                    pass
            
            author = item.get("author", {})
            
            post = {
                "post_id": str(post_id),
                "url": item.get("url", ""),
                "text": item.get("postText", ""),
                "author_name": author.get("name", "Unknown"),
                "author_id": author.get("id", ""),
                "author_url": author.get("profileUrl", ""),
                "like_count": item.get("reactionsCount", 0),
                "comment_count": item.get("commentsCount", 0),
                "published_at": published_at,
                "reactions": item.get("reactions", {}),
                "comments": [],  # Will be filled by comment scraper
                "source": "search"  # Mark as search result for tracking
            }
            
            return post
            
        except Exception as e:
            self.logger.error(f"Error parsing search post: {e}")
            return None
    
    def _parse_apify_post(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Apify facebook-pages-scraper and facebook-posts-scraper output"""
        try:
            post_id = (item.get("postId") or 
                      item.get("id") or 
                      item.get("post_id") or
                      item.get("pagePostId"))
            
            if not post_id:
                self.logger.warning(f"No post ID found in item. Available keys: {list(item.keys())}")
                return None
            
            text = (item.get("text") or 
                   item.get("content") or 
                   item.get("post_text") or
                   item.get("message") or
                   "")
            
            author = (item.get("authorName") or 
                     item.get("author", {}).get("name") if isinstance(item.get("author"), dict) else item.get("author") or
                     item.get("page_name") or
                     "Unknown")
            
            author_id = (item.get("authorId") or 
                        item.get("author", {}).get("id") if isinstance(item.get("author"), dict) else None or
                        item.get("page_id"))
            
            url = (item.get("postUrl") or 
                  item.get("url") or 
                  item.get("post_url") or
                  item.get("link"))
            
            media_urls = item.get("images", []) or item.get("media", []) or []
            if not isinstance(media_urls, list):
                media_urls = []
            
            post_data = {
                "post_id": post_id,
                "url": url,
                "text": text,
                "author": author,
                "author_id": author_id,
                "published_at": item.get("publishedAt") or item.get("time") or item.get("created_time"),
                "like_count": item.get("likes", 0) or item.get("reactions", 0) or 0,
                "comment_count": item.get("comments", 0) or item.get("comment_count", 0) or 0,
                "share_count": item.get("shares", 0) or item.get("share_count", 0) or 0,
                "view_count": item.get("views", 0) or item.get("view_count", 0) or 0,
                "media_urls": media_urls,
                "post_type": item.get("postType", "post") or item.get("type", "post"),
                "comments": [],
                "source": "page"
            }
            
            comments_data = item.get("postComments", []) or item.get("comments", [])
            if not isinstance(comments_data, list):
                comments_data = []
            
            for comment in comments_data:
                if not isinstance(comment, dict):
                    continue
                    
                comment_id = comment.get("commentId") or comment.get("id")
                if not comment_id:
                    continue
                
                comment_info = {
                    "comment_id": comment_id,
                    "text": comment.get("text") or comment.get("content") or "",
                    "author_name": comment.get("authorName") or comment.get("author", {}).get("name", "Unknown"),
                    "author_id": comment.get("authorId") or comment.get("author", {}).get("id"),
                    "published_at": comment.get("publishedAt") or comment.get("time"),
                    "like_count": comment.get("likes", 0),
                    "reply_count": comment.get("replies", 0)
                }
                
                post_data["comments"].append(comment_info)
            
            return post_data
            
        except Exception as e:
            self.logger.error(f"Error parsing post: {str(e)}")
            return None
    
    async def collect_comments(
        self, 
        post_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        self.logger.info("Comments are collected with posts in this scraper")
        return []
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        """Convert raw Facebook post to SocialPostCreate schema"""
        return SocialPostCreate(
            attraction_id=attraction_id,
            platform=PlatformEnum.FACEBOOK,
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
            post_type=raw_post.get("post_type", "post")
        )
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int, attraction_id: int) -> CommentCreate:
        """Convert raw Facebook comment to CommentCreate schema"""
        return CommentCreate(
            post_id=post_id,
            attraction_id=attraction_id,
            platform=PlatformEnum.FACEBOOK,
            platform_comment_id=raw_comment["comment_id"],
            content=raw_comment.get("text", ""),
            author=raw_comment.get("author_name", "Unknown"),
            author_id=raw_comment.get("author_id"),
            comment_date=self._parse_datetime(raw_comment.get("published_at")),
            like_count=raw_comment.get("like_count", 0),
            reply_count=raw_comment.get("reply_count", 0)
        )
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception:
            pass
        
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except Exception:
            return None
