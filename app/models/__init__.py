from app.models.base import Base
from app.models.province import Province
from app.models.tourist_attraction import TouristAttraction
from app.models.social_post import SocialPost
from app.models.comment import Comment
from app.models.analysis_log import AnalysisLog

__all__ = [
    "Base",
    "Province",
    "TouristAttraction",
    "SocialPost",
    "Comment",
    "AnalysisLog"
]
