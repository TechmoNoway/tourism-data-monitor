from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    logging.warning("Google API client not installed. Run: pip install google-api-python-client")

from app.collectors.base_collector import BaseCollector
from app.schemas.post import SocialPostCreate
from app.schemas.comment import CommentCreate


class YouTubeCollector(BaseCollector):
 
    def __init__(self, api_key: str):
        super().__init__("youtube")
        self.api_key = api_key
        self.youtube_service = None
        
    def authenticate(self, **credentials) -> bool:
        try:
            self.youtube_service = build('youtube', 'v3', developerKey=self.api_key)
            self.logger.info("YouTube API authentication successful")
            return True
        except Exception as e:
            self.logger.error(f"YouTube API authentication failed: {str(e)}")
            return False
    
    async def collect_posts(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        if not self.youtube_service:
            raise RuntimeError("YouTube service not authenticated")
            
        all_videos = []
        
        for keyword in keywords:
            try:
                search_query = f"{keyword} du lịch review"
                
                search_response = self.youtube_service.search().list(
                    q=search_query,
                    part='id,snippet',
                    type='video',
                    maxResults=min(limit // len(keywords), 50),
                    order='relevance',
                    relevanceLanguage='vi',
                    regionCode='VN'
                ).execute()
                
                for item in search_response.get('items', []):
                    video_id = item['id']['videoId']
                    
                    video_details = self.youtube_service.videos().list(
                        part='snippet,statistics,contentDetails',
                        id=video_id
                    ).execute()
                    
                    if video_details['items']:
                        video_data = video_details['items'][0]
                        
                        try:
                            comments_response = self.youtube_service.commentThreads().list(
                                part='snippet',
                                videoId=video_id,
                                maxResults=1
                            ).execute()
                            comments_available = len(comments_response.get('items', [])) > 0
                        except HttpError:
                            comments_available = False
                        
                        processed_video = {
                            'video_id': video_id,
                            'title': video_data['snippet']['title'],
                            'description': video_data['snippet'].get('description', ''),
                            'channel_title': video_data['snippet']['channelTitle'],
                            'channel_id': video_data['snippet']['channelId'],
                            'published_at': video_data['snippet']['publishedAt'],
                            'view_count': int(video_data['statistics'].get('viewCount', 0)),
                            'like_count': int(video_data['statistics'].get('likeCount', 0)),
                            'comment_count': int(video_data['statistics'].get('commentCount', 0)),
                            'duration': video_data['contentDetails']['duration'],
                            'keywords': [keyword],
                            'thumbnail_url': video_data['snippet']['thumbnails'].get('high', {}).get('url', ''),
                            'comments_available': comments_available,
                            'url': f"https://www.youtube.com/watch?v={video_id}"
                        }
                        
                        all_videos.append(processed_video)
                        
                        self.logger.info(f"Collected video: {video_data['snippet']['title']}")
                        
            except HttpError as e:
                # Check if quota exceeded
                if e.resp.status == 403 and 'quotaExceeded' in str(e):
                    self.logger.warning(f"⚠️ YouTube API quota exceeded! Will skip YouTube for now.")
                    self.logger.info("💡 Tip: YouTube quota resets at midnight PT. Try again tomorrow.")
                    # Return empty list to let pipeline continue with other platforms
                    return []
                else:
                    self.logger.error(f"Error searching YouTube for keyword '{keyword}': {str(e)}")
                    continue
            except Exception as e:
                self.logger.error(f"Unexpected error searching YouTube: {str(e)}")
                continue
                
        self.logger.info(f"Collected {len(all_videos)} videos from YouTube")
        return all_videos[:limit]
    
    async def collect_comments(
        self, 
        video_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        if not self.youtube_service:
            raise RuntimeError("YouTube service not authenticated")
            
        all_comments = []
        
        try:
            comments_response = self.youtube_service.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=min(limit, 100),
                order='relevance'
            ).execute()
            
            for item in comments_response.get('items', []):
                top_comment = item['snippet']['topLevelComment']['snippet']
                
                comment_data = {
                    'comment_id': item['snippet']['topLevelComment']['id'],
                    'video_id': video_id,
                    'author_name': top_comment['authorDisplayName'],
                    'author_channel_id': top_comment.get('authorChannelId', {}).get('value', ''),
                    'text': top_comment['textDisplay'],
                    'like_count': top_comment.get('likeCount', 0),
                    'published_at': top_comment['publishedAt'],
                    'updated_at': top_comment.get('updatedAt', top_comment['publishedAt']),
                    'parent_id': None,  # Top-level comment
                    'reply_count': item['snippet'].get('totalReplyCount', 0)
                }
                
                all_comments.append(comment_data)
                
                if 'replies' in item and len(all_comments) < limit:
                    for reply in item['replies']['comments']:
                        if len(all_comments) >= limit:
                            break
                            
                        reply_snippet = reply['snippet']
                        
                        reply_data = {
                            'comment_id': reply['id'],
                            'video_id': video_id,
                            'author_name': reply_snippet['authorDisplayName'],
                            'author_channel_id': reply_snippet.get('authorChannelId', {}).get('value', ''),
                            'text': reply_snippet['textDisplay'],
                            'like_count': reply_snippet.get('likeCount', 0),
                            'published_at': reply_snippet['publishedAt'],
                            'updated_at': reply_snippet.get('updatedAt', reply_snippet['publishedAt']),
                            'parent_id': comment_data['comment_id'],  # Reply to top-level comment
                            'reply_count': 0
                        }
                        
                        all_comments.append(reply_data)
                
                if len(all_comments) >= limit:
                    break
                    
        except HttpError as e:
            if e.resp.status == 403:
                self.logger.warning(f"Comments disabled for video {video_id}")
            else:
                self.logger.error(f"Error collecting comments for video {video_id}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error collecting comments: {str(e)}")
            
        self.logger.info(f"Collected {len(all_comments)} comments for video {video_id}")
        return all_comments
    
    def _convert_raw_post(self, raw_post: Dict[str, Any], attraction_id: int) -> SocialPostCreate:
        published_at = datetime.fromisoformat(raw_post['published_at'].replace('Z', '+00:00'))
        
        return SocialPostCreate(
            attraction_id=attraction_id,
            platform="youtube",
            platform_post_id=raw_post['video_id'],
            author_name=raw_post['channel_title'],
            author_id=raw_post['channel_id'],
            content=f"{raw_post['title']}\n\n{raw_post['description'][:500]}...", 
            posted_at=published_at,
            url=raw_post['url'],
            likes_count=raw_post['like_count'],
            comments_count=raw_post['comment_count'],
            shares_count=0,
            views_count=raw_post['view_count'],
            language=self._detect_language(raw_post['title'] + ' ' + raw_post['description']),
            metadata={
                'duration': raw_post['duration'],
                'thumbnail_url': raw_post['thumbnail_url'],
                'keywords': raw_post['keywords'],
                'comments_available': raw_post['comments_available']
            }
        )
    
    def _convert_raw_comment(self, raw_comment: Dict[str, Any], post_id: int, attraction_id: int) -> CommentCreate:
        published_at = datetime.fromisoformat(raw_comment['published_at'].replace('Z', '+00:00'))
        
        return CommentCreate(
            post_id=post_id,
            attraction_id=attraction_id,
            platform="youtube",
            platform_comment_id=raw_comment['comment_id'],
            author=raw_comment['author_name'],
            author_id=raw_comment['author_channel_id'],
            content=self._clean_text(raw_comment['text']),
            comment_date=published_at,
            like_count=raw_comment['like_count'],
            reply_count=raw_comment['reply_count']
        )


# Factory function for easy instantiation
def create_youtube_collector(api_key: str) -> YouTubeCollector:
    collector = YouTubeCollector(api_key)
    
    if not collector.authenticate():
        raise RuntimeError("Failed to authenticate YouTube collector")
        
    return collector