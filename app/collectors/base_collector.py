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


class BaseCollector(ABC):
    """
    Abstract base class for all social media data collectors
    """

    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.logger = logging.getLogger(f"collector.{platform_name}")

    @abstractmethod
    async def collect_posts(
        self, keywords: List[str], location: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect posts from the platform

        Args:
            keywords: List of keywords to search for
            location: Optional location filter
            limit: Maximum number of posts to collect

        Returns:
            List of raw post data
        """
        pass

    @abstractmethod
    async def collect_comments(
        self, platform_post_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect comments for a specific post

        Args:
            platform_post_id: Platform-specific post ID
            limit: Maximum number of comments to collect

        Returns:
            List of raw comment data
        """
        pass

    @abstractmethod
    def authenticate(self, **credentials) -> bool:
        """
        Authenticate with the platform API

        Returns:
            True if authentication successful, False otherwise
        """
        pass

    def process_and_store_posts(
        self, raw_posts: List[Dict[str, Any]], attraction_id: int
    ) -> List[int]:
        """
        Process raw post data and store in database

        Args:
            raw_posts: List of raw post data from platform
            attraction_id: ID of the tourist attraction

        Returns:
            List of created post IDs
        """
        db = next(get_db())
        created_posts = []

        try:
            for raw_post in raw_posts:
                # Convert raw data to Pydantic schema
                post_data = self._convert_raw_post(raw_post, attraction_id)

                # Check if post already exists
                existing_post = (
                    db.query(SocialPost)
                    .filter(
                        SocialPost.platform_post_id == post_data.platform_post_id,
                        SocialPost.platform == self.platform_name,
                    )
                    .first()
                )

                if existing_post:
                    self.logger.info(
                        f"Post {post_data.platform_post_id} already exists, skipping"
                    )
                    continue

                # Create new post
                db_post = SocialPost(**post_data.dict())
                db.add(db_post)
                db.flush()  # Get the ID without committing

                created_posts.append(db_post.id)
                self.logger.info(f"Created post {db_post.id} from {self.platform_name}")

            db.commit()

        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing posts: {str(e)}")
            raise
        finally:
            db.close()

        return created_posts

    def process_and_store_comments(
        self, raw_comments: List[Dict[str, Any]], post_id: int
    ) -> List[int]:
        """
        Process raw comment data and store in database

        Args:
            raw_comments: List of raw comment data from platform
            post_id: Database ID of the social post

        Returns:
            List of created comment IDs
        """
        db = next(get_db())
        created_comments = []

        try:
            for raw_comment in raw_comments:
                # Convert raw data to Pydantic schema
                comment_data = self._convert_raw_comment(raw_comment, post_id)

                # Check if comment already exists
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

                # Create new comment
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
    ):
        """
        Log collection activity to analysis_logs table

        Args:
            attraction_id: ID of the tourist attraction
            posts_collected: Number of posts collected
            comments_collected: Number of comments collected
            status: Collection status (success/error)
            error_message: Optional error message if status is error
        """
        db = next(get_db())

        try:
            log_entry = AnalysisLog(
                attraction_id=attraction_id,
                analysis_type=f"{self.platform_name}_collection",
                analysis_parameters={
                    "platform": self.platform_name,
                    "posts_collected": posts_collected,
                    "comments_collected": comments_collected,
                    "collection_time": datetime.utcnow().isoformat(),
                },
                results={
                    "status": status,
                    "posts_count": posts_collected,
                    "comments_count": comments_collected,
                    "error_message": error_message,
                },
                created_at=datetime.utcnow(),
            )

            db.add(log_entry)
            db.commit()

            self.logger.info(
                f"Logged collection activity for attraction {attraction_id}"
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
        """
        Convert platform-specific raw post data to SocialPostCreate schema

        Args:
            raw_post: Raw post data from platform
            attraction_id: ID of the tourist attraction

        Returns:
            SocialPostCreate schema instance
        """
        pass

    @abstractmethod
    def _convert_raw_comment(
        self, raw_comment: Dict[str, Any], post_id: int
    ) -> CommentCreate:
        """
        Convert platform-specific raw comment data to CommentCreate schema

        Args:
            raw_comment: Raw comment data from platform
            post_id: Database ID of the social post

        Returns:
            CommentCreate schema instance
        """
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
        """
        Clean and normalize text content

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespaces
        cleaned = " ".join(text.split())

        # Remove or replace special characters if needed
        # Add more cleaning rules as needed

        return cleaned
