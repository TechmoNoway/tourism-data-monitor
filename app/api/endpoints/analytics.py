"""Analytics endpoints for trends and statistics"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from datetime import datetime, timedelta
from typing import List, Optional
from app.database.connection import get_db
from app.models.comment import Comment
from app.models.social_post import SocialPost
from app.models.tourist_attraction import TouristAttraction

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/collection-trends")
async def get_collection_trends(
    attraction_id: Optional[int] = Query(None, description="Filter by attraction ID"),
    province_id: Optional[int] = Query(None, description="Filter by province ID"),
    period: str = Query("2weeks", description="Time period: 1week, 2weeks, 1month, 3months, 6months, 1year"),
    db: Session = Depends(get_db)
):
    """
    Get collection trends showing comment counts over time
    Returns data grouped by collection period (every 2 weeks by default)
    """
    
    # Determine time range
    period_days = {
        "1week": 7,
        "2weeks": 14,
        "1month": 30,
        "3months": 90,
        "6months": 180,
        "1year": 365
    }
    
    days = period_days.get(period, 180)
    start_date = datetime.now() - timedelta(days=days)
    
    # Build query
    query = db.query(
        func.date_trunc('week', Comment.scraped_at).label('collection_week'),
        func.count(Comment.id).label('comment_count'),
        func.count(func.distinct(Comment.attraction_id)).label('attraction_count')
    ).filter(
        Comment.scraped_at >= start_date
    )
    
    # Apply filters
    if attraction_id:
        query = query.filter(Comment.attraction_id == attraction_id)
    elif province_id:
        query = query.join(
            TouristAttraction,
            Comment.attraction_id == TouristAttraction.id
        ).filter(
            TouristAttraction.province_id == province_id
        )
    
    # Group by week
    results = query.group_by('collection_week').order_by('collection_week').all()
    
    # Format response
    trends = []
    for row in results:
        trends.append({
            "date": row.collection_week.strftime("%Y-%m-%d") if row.collection_week else None,
            "comment_count": row.comment_count,
            "attraction_count": row.attraction_count
        })
    
    # Calculate statistics
    total_comments = sum(t['comment_count'] for t in trends)
    avg_per_period = total_comments / len(trends) if trends else 0
    max_period = max(trends, key=lambda x: x['comment_count']) if trends else None
    
    return {
        "period": period,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "trends": trends,
        "statistics": {
            "total_comments": total_comments,
            "total_periods": len(trends),
            "avg_comments_per_period": round(avg_per_period, 2),
            "peak_period": {
                "date": max_period['date'] if max_period else None,
                "count": max_period['comment_count'] if max_period else 0
            }
        }
    }


@router.get("/sentiment-trends")
async def get_sentiment_trends(
    attraction_id: Optional[int] = Query(None, description="Filter by attraction ID"),
    province_id: Optional[int] = Query(None, description="Filter by province ID"),
    period: str = Query("3months", description="Time period"),
    db: Session = Depends(get_db)
):
    """
    Get sentiment trends over time
    Shows positive/negative/neutral comment counts per period
    """
    
    period_days = {
        "1week": 7,
        "2weeks": 14,
        "1month": 30,
        "3months": 90,
        "6months": 180,
        "1year": 365
    }
    
    days = period_days.get(period, 90)
    start_date = datetime.now() - timedelta(days=days)
    
    # Build query
    query = db.query(
        func.date_trunc('week', Comment.scraped_at).label('period'),
        Comment.sentiment,
        func.count(Comment.id).label('count')
    ).filter(
        Comment.scraped_at >= start_date,
        Comment.sentiment.isnot(None)
    )
    
    # Apply filters
    if attraction_id:
        query = query.filter(Comment.attraction_id == attraction_id)
    elif province_id:
        query = query.join(
            TouristAttraction,
            Comment.attraction_id == TouristAttraction.id
        ).filter(
            TouristAttraction.province_id == province_id
        )
    
    results = query.group_by('period', Comment.sentiment).order_by('period').all()
    
    # Organize by period
    periods = {}
    for row in results:
        date_key = row.period.strftime("%Y-%m-%d") if row.period else "unknown"
        if date_key not in periods:
            periods[date_key] = {
                "date": date_key,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "total": 0
            }
        
        sentiment = row.sentiment.lower() if row.sentiment else "neutral"
        if sentiment in ["positive", "negative", "neutral"]:
            periods[date_key][sentiment] = row.count
            periods[date_key]["total"] += row.count
    
    # Calculate percentages
    trends = []
    for date_key in sorted(periods.keys()):
        data = periods[date_key]
        total = data["total"]
        trends.append({
            "date": data["date"],
            "positive": data["positive"],
            "negative": data["negative"],
            "neutral": data["neutral"],
            "total": total,
            "positive_percentage": round((data["positive"] / total * 100) if total > 0 else 0, 1),
            "negative_percentage": round((data["negative"] / total * 100) if total > 0 else 0, 1),
            "neutral_percentage": round((data["neutral"] / total * 100) if total > 0 else 0, 1)
        })
    
    return {
        "period": period,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "trends": trends
    }


@router.get("/platform-distribution")
async def get_platform_distribution(
    attraction_id: Optional[int] = Query(None, description="Filter by attraction ID"),
    province_id: Optional[int] = Query(None, description="Filter by province ID"),
    db: Session = Depends(get_db)
):
    """
    Get distribution of comments across platforms
    """
    
    query = db.query(
        Comment.platform,
        func.count(Comment.id).label('count')
    ).filter(
        Comment.platform.isnot(None)
    )
    
    # Apply filters
    if attraction_id:
        query = query.filter(Comment.attraction_id == attraction_id)
    elif province_id:
        query = query.join(
            TouristAttraction,
            Comment.attraction_id == TouristAttraction.id
        ).filter(
            TouristAttraction.province_id == province_id
        )
    
    results = query.group_by(Comment.platform).all()
    
    total = sum(r.count for r in results)
    distribution = [
        {
            "platform": row.platform,
            "count": row.count,
            "percentage": round((row.count / total * 100) if total > 0 else 0, 1)
        }
        for row in results
    ]
    
    return {
        "total_comments": total,
        "distribution": sorted(distribution, key=lambda x: x['count'], reverse=True)
    }


@router.get("/growth-metrics")
async def get_growth_metrics(
    db: Session = Depends(get_db)
):
    """
    Get overall growth metrics for the platform
    """
    
    now = datetime.now()
    last_week = now - timedelta(days=7)
    last_month = now - timedelta(days=30)
    
    # Comments growth
    total_comments = db.query(func.count(Comment.id)).scalar()
    comments_last_week = db.query(func.count(Comment.id)).filter(
        Comment.scraped_at >= last_week
    ).scalar()
    comments_last_month = db.query(func.count(Comment.id)).filter(
        Comment.scraped_at >= last_month
    ).scalar()
    
    # Attractions with data
    total_attractions = db.query(func.count(TouristAttraction.id)).filter(
        TouristAttraction.is_active.is_(True)
    ).scalar()
    
    attractions_with_comments = db.query(
        func.count(func.distinct(Comment.attraction_id))
    ).scalar()
    
    # Sentiment analysis coverage
    analyzed_comments = db.query(func.count(Comment.id)).filter(
        Comment.sentiment.isnot(None)
    ).scalar()
    
    analysis_percentage = round((analyzed_comments / total_comments * 100) if total_comments > 0 else 0, 1)
    
    return {
        "comments": {
            "total": total_comments,
            "last_week": comments_last_week,
            "last_month": comments_last_month,
            "analyzed": analyzed_comments,
            "analysis_coverage": analysis_percentage
        },
        "attractions": {
            "total": total_attractions,
            "with_data": attractions_with_comments,
            "coverage": round((attractions_with_comments / total_attractions * 100) if total_attractions > 0 else 0, 1)
        }
    }
