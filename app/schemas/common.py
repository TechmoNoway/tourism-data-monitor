
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
    timestamp: datetime = datetime.now()

class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
    details: Optional[str] = None

class PaginatedResponse(BaseModel): 
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool

class FilterParams(BaseModel):
    page: int = 1
    size: int = 20
    sort_by: Optional[str] = "created_at"
    sort_order: Optional[str] = "desc"  # or "asc"

class DateRangeFilter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class SentimentFilter(BaseModel):
    sentiment: Optional[str] = None  # "positive", "negative", "neutral"
    min_confidence: Optional[float] = None
    exclude_bots: bool = True

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    page: int = 1
    size: int = 20

class SearchResult(BaseModel):
    query: str
    results: List[Dict]
    total_found: int
    search_time: float
    suggestions: Optional[List[str]] = None 