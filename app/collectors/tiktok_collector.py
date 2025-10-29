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
    def __init__(self, apify_api_token: str):
        super().__init__("tiktok")
        self.apify_token = apify_api_token
        self.client = None
        
    def authenticate(self, **credentials) -> bool:
        try:
            self.client = ApifyClient(self.apify_token)
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
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
            
        all_videos = []
        
        try:
           
            actor_id = "OtzYfK1ndEGdwWFKQ"
            
            search_queries = []
            for keyword in keywords:
                if keyword.startswith('#'):
                    search_queries.append({
                        "type": "hashtag",
                        "input": keyword[1:]
                    })
                elif keyword.startswith('@'):
                    # Profile/username
                    search_queries.append({
                        "type": "profile",
                        "input": keyword[1:]
                    })
                else:
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
            
            hashtags_list = []
            for query in search_queries[:5]:
                if query["type"] == "hashtag":
                    hashtags_list.append(f"#{query['input']}")
                elif query["type"] == "profile":
                    hashtags_list.append(f"@{query['input']}")
            
            run_input = {
                "hashtags": hashtags_list, 
                "resultsPerPage": min(limit, 50),
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False,
                "shouldDownloadSubtitles": False,
                "shouldDownloadSlideshowImages": False
            }
            
            self.logger.info(f"🚀 Starting Apify actor {actor_id}")
            self.logger.info(f"📌 Hashtags: {', '.join(hashtags_list)}")
            self.logger.info(f"📊 Max results: {min(limit, 50)} per hashtag")
            
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
        Collect comments for a TikTok video using specialized Comments Scraper
        
        Uses: clockworks/tiktok-comments-scraper (18K users, 4.8 rating)
        This actor is specifically designed for extracting TikTok comments
        """
        if not self.client:
            raise RuntimeError("Apify client not authenticated")
            
        all_comments = []
        
        try:
            # Use specialized TikTok Comments Scraper actor
            # Much better than the general scraper for comments specifically
            actor_id = "clockworks/tiktok-comments-scraper"
            
            # Input format for comments scraper - expects video URLs
            # NOTE: Field name is "postURLs" (capital U) not "postUrls"
            run_input = {
                "postURLs": [video_url],  # Must be "postURLs" with capital U
                "maxComments": min(limit, 500),  # Max comments to collect (up to 500)
                "maxReplies": 10,  # Increased from 5 to 10 replies per comment for better thread context
                "commentsPerPage": 30,  # How many to load per page
            }
            
            self.logger.info(f"💬 Collecting comments using specialized Comments Scraper")
            self.logger.info(f"📹 Video: {video_url}")
            self.logger.info(f"🎯 Target: {min(limit, 500)} comments")
            
            run = self.client.actor(actor_id).call(run_input=run_input)
            
            # Process results from comments scraper
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                # Comments scraper returns comments directly, not nested in video data
                comment_data = {
                    'comment_id': item.get('id') or item.get('cid'),
                    'video_id': item.get('videoId') or video_url.split('/')[-1],
                    'text': item.get('text', ''),
                    'author_name': item.get('uniqueId') or item.get('nickname', 'Unknown'),
                    'author_id': item.get('authorId', ''),
                    'create_time': item.get('createTime') or item.get('createTimeISO'),
                    'like_count': item.get('diggCount', 0),
                    'reply_count': item.get('replyCommentTotal', 0)
                }
                
                all_comments.append(comment_data)
                
                if len(all_comments) >= limit:
                    break
                            
        except Exception as e:
            self.logger.error(f"❌ Error collecting TikTok comments: {str(e)}")
            self.logger.info("💡 Tip: Make sure the video URL is valid and public")
            import traceback
            self.logger.debug(traceback.format_exc())
            
        if all_comments:
            self.logger.info(f"✅ Collected {len(all_comments)} comments from TikTok")
        else:
            self.logger.warning(f"⚠️  No comments collected for {video_url}")
            self.logger.info("💡 This could be because:")
            self.logger.info("   1. Video has no comments yet")
            self.logger.info("   2. Comments are disabled for this video")
            self.logger.info("   3. TikTok's anti-bot measures blocked the scraper")
            
        return all_comments[:limit]
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        created_time = None
        if raw_post.get('create_time'):
            try:
                if isinstance(raw_post['create_time'], str):
                    created_time = datetime.fromisoformat(raw_post['create_time'].replace('Z', '+00:00'))
                else:
                    created_time = datetime.fromtimestamp(raw_post['create_time'])
            except Exception:
                created_time = datetime.utcnow()
        
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
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int, attraction_id: int) -> CommentCreate:
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
            attraction_id=attraction_id,
            content=self._clean_text(raw_comment.get('text', '')),
            author=raw_comment.get('author_name', 'Unknown'),
            author_id=raw_comment.get('author_id', ''),
            comment_date=created_time or datetime.utcnow(),
            like_count=raw_comment.get('like_count', 0),
            reply_count=raw_comment.get('reply_count', 0)
        )


# Factory function
def create_tiktok_collector(apify_token: str) -> TikTokApifyCollector:
    """Factory function to create TikTok collector with Apify"""
    collector = TikTokApifyCollector(apify_token)
    
    if not collector.authenticate():
        logging.warning("TikTok collector created but authentication failed")
        
    return collector
