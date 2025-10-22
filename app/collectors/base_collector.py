from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.database.connection import get_db
from app.models.social_post import SocialPost
from app.models.comment import Comment
from app.models.analysis_log import AnalysisLog
from app.schemas.post import SocialPostCreate
from app.schemas.comment import CommentCreate
from app.collectors.relevance_filter import RelevanceFilter, PlatformKeywordOptimizer


class BaseCollector(ABC):

    def __init__(self, platform_name: str, min_relevance_score: float = 0.0):
        """
        Initialize base collector.
        
        Args:
            platform_name: Name of the platform (e.g., 'facebook', 'youtube')
            min_relevance_score: Minimum relevance score to keep posts (0-1).
                                 Default 0.0 (disabled) for testing.
                                 Increase to 0.3+ for production filtering.
        """
        self.platform_name = platform_name
        self.logger = logging.getLogger(f"collector.{platform_name}")
        self.relevance_filter = RelevanceFilter(min_relevance_score=min_relevance_score)
        self.keyword_optimizer = PlatformKeywordOptimizer()
        self.min_relevance_score = min_relevance_score

    @abstractmethod
    async def collect_posts(
        self, keywords: List[str], location: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def collect_comments(
        self, platform_post_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def authenticate(self, **credentials) -> bool:
        pass

    def generate_search_keywords(
        self, attraction_name: str, province_name: str
    ) -> List[str]:
        """
        Generate optimized search keywords for the attraction.
        
        Args:
            attraction_name: Name of tourist attraction
            province_name: Name of province
            
        Returns:
            List of search keywords optimized for this platform
        """
        keywords = self.relevance_filter.generate_keywords(
            attraction_name, province_name
        )
        
        # Apply platform-specific optimization
        if self.platform_name.lower() == 'youtube':
            keywords = self.keyword_optimizer.optimize_for_youtube(keywords)
        elif self.platform_name.lower() == 'facebook':
            keywords = self.keyword_optimizer.optimize_for_facebook(keywords)
        elif self.platform_name.lower() == 'tiktok':
            keywords = self.keyword_optimizer.optimize_for_tiktok(keywords)
        
        self.logger.info(f"Generated {len(keywords)} keywords for {attraction_name}, {province_name}")
        return keywords

    def filter_relevant_posts(
        self, 
        raw_posts: List[Dict[str, Any]], 
        attraction_name: str, 
        province_name: str
    ) -> List[Dict[str, Any]]:
        """
        Filter posts to keep only relevant ones.
        
        Args:
            raw_posts: List of raw post data from API
            attraction_name: Name of tourist attraction
            province_name: Name of province
            
        Returns:
            Filtered list of relevant posts
        """
        if not raw_posts:
            return []
        
        initial_count = len(raw_posts)
        
        # Filter by relevance score
        filtered_posts = self.relevance_filter.filter_posts(
            raw_posts, attraction_name, province_name
        )
        
        # Filter by date (keep last 90 days by default)
        filtered_posts = self.relevance_filter.filter_by_date(filtered_posts, days_back=90)
        
        # Remove duplicates
        filtered_posts = self.relevance_filter.remove_duplicates(filtered_posts)
        
        final_count = len(filtered_posts)
        filtered_count = initial_count - final_count
        
        self.logger.info(
            f"Filtered {filtered_count}/{initial_count} posts. "
            f"Kept {final_count} relevant posts (min_score={self.min_relevance_score})"
        )
        
        return filtered_posts

    def process_and_store_posts(
        self, raw_posts: List[Dict[str, Any]], attraction_id: int
    ) -> Dict[str, int]:
        """
        Process and store posts in database.
        
        Returns:
            Dict mapping platform_post_id to database post ID.
            Includes BOTH new posts and existing posts, enabling comment collection for all.
        """
        db = next(get_db())
        post_id_mapping = {}  # platform_post_id -> db_post_id

        try:
            for raw_post in raw_posts:
                post_data = self._convert_raw_post(raw_post, attraction_id)

                existing_post = (
                    db.query(SocialPost)
                    .filter(
                        SocialPost.platform_post_id == post_data.platform_post_id,
                        SocialPost.platform == self.platform_name,
                    )
                    .first()
                )

                if existing_post:
                    # Include existing post in mapping so we can collect comments for it
                    post_id_mapping[post_data.platform_post_id] = existing_post.id
                    self.logger.info(
                        f"Post {post_data.platform_post_id} already exists (ID={existing_post.id}), will collect comments"
                    )
                else:
                    # Create new post
                    db_post = SocialPost(**post_data.model_dump())
                    db.add(db_post)
                    db.flush()  

                    post_id_mapping[post_data.platform_post_id] = db_post.id
                    self.logger.info(f"Created post {db_post.id} from {self.platform_name}")

            db.commit()

        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing posts: {str(e)}")
            raise
        finally:
            db.close()

        return post_id_mapping

    def process_and_store_comments(
        self, raw_comments: List[Dict[str, Any]], post_id: int, attraction_id: int
    ) -> List[int]:
        db = next(get_db())
        created_comments = []

        try:
            for raw_comment in raw_comments:
                comment_data = self._convert_raw_comment(raw_comment, post_id, attraction_id)

                existing_comment = (
                    db.query(Comment)
                    .filter(
                        Comment.platform_comment_id == comment_data.platform_comment_id,
                        Comment.platform == self.platform_name,
                    )
                    .first()
                )

                if existing_comment:
                    self.logger.info(
                        f"Comment {comment_data.platform_comment_id} already exists, skipping"
                    )
                    continue

                db_comment = Comment(**comment_data.dict())
                db.add(db_comment)
                db.flush()

                created_comments.append(db_comment.id)
                self.logger.info(
                    f"Created comment {db_comment.id} from {self.platform_name}"
                )

            db.commit()

        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing comments: {str(e)}")
            raise
        finally:
            db.close()

        return created_comments

    def log_collection_activity(
        self,
        attraction_id: int,
        posts_collected: int,
        comments_collected: int,
        status: str = "success",
        error_message: Optional[str] = None,
        posts_filtered: int = 0,
        initial_posts: int = 0,
    ):
        import json
        db = next(get_db())

        try:
            log_entry = AnalysisLog(
                attraction_id=attraction_id,
                total_comments=comments_collected,
                trending_aspects=json.dumps({
                    "platform": self.platform_name,
                    "posts_collected": posts_collected,
                    "comments_collected": comments_collected,
                    "posts_filtered": posts_filtered,
                    "initial_posts": initial_posts
                }),
                activity_score=float(posts_collected)
            )

            db.add(log_entry)
            db.commit()

            self.logger.info(
                f"Logged collection activity for attraction {attraction_id}: "
                f"{posts_collected} posts stored, {posts_filtered} filtered out"
            )

        except Exception as e:
            db.rollback()
            self.logger.error(f"Error logging collection activity: {str(e)}")
        finally:
            db.close()

    @abstractmethod
    def _convert_raw_post(
        self, raw_post: Dict[str, Any], attraction_id: int
    ) -> SocialPostCreate:
        pass

    @abstractmethod
    def _convert_raw_comment(
        self, raw_comment: Dict[str, Any], post_id: int, attraction_id: int
    ) -> CommentCreate:
        pass

    def _detect_language(self, text: str) -> str:
        """
        Detect language of text content

        Args:
            text: Text content to analyze

        Returns:
            Language code (default: 'vi' for Vietnamese)
        """
        # Simple language detection - can be enhanced with proper library
        vietnamese_keywords = [
            "là",
            "và",
            "của",
            "có",
            "để",
            "với",
            "từ",
            "về",
            "này",
            "đó",
        ]

        if any(keyword in text.lower() for keyword in vietnamese_keywords):
            return "vi"
        else:
            return "en"  # Default to English if not Vietnamese

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        cleaned = " ".join(text.split())
        
        return cleaned
