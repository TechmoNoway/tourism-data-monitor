from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta, timezone

from app.database.connection import get_db
from app.services.demand_service import DemandIndexService
from app.schemas.demand import (
    DemandIndexResponse,
    ProvinceDemandIndexResponse,
    TopAttraction,
    TopProvince,
    DemandAnalytics,
    ComparativeAnalysis
)
from app.models.demand_index import DemandIndex, ProvinceDemandIndex
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province
from sqlalchemy import func, desc

router = APIRouter()


@router.get("/demand/attractions/top", response_model=List[TopAttraction])
def get_top_attractions(
    period: str = Query("week", description="week, month, quarter"),
    province_id: Optional[int] = None,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    period_days = {"week": 7, "month": 30, "quarter": 90}
    days = period_days.get(period, 7)
    
    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=days)
    
    results = DemandIndexService.get_top_attractions(
        db, period_start, period_end, limit, province_id, period
    )
    
    return results


@router.get("/demand/provinces/top", response_model=List[TopProvince])
def get_top_provinces(
    period: str = Query("week", description="week, month, quarter"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    period_days = {"week": 7, "month": 30, "quarter": 90}
    days = period_days.get(period, 7)
    
    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=days)
    
    results = DemandIndexService.get_top_provinces(
        db, period_start, period_end, limit, period
    )
    
    return results


@router.get("/demand/attractions/{attraction_id}", response_model=DemandIndexResponse)
def get_attraction_demand(
    attraction_id: int,
    period: str = Query("week", description="week, month, quarter"),
    db: Session = Depends(get_db)
):
    period_days = {"week": 7, "month": 30, "quarter": 90}
    days = period_days.get(period, 7)
    
    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=days)
    
    # Get the most recent period for the given period_type
    latest_period = db.query(DemandIndex).filter(
        DemandIndex.period_type == period
    ).order_by(desc(DemandIndex.period_end)).first()
    
    if latest_period:
        demand = db.query(DemandIndex).filter(
            DemandIndex.attraction_id == attraction_id,
            DemandIndex.period_start == latest_period.period_start,
            DemandIndex.period_end == latest_period.period_end,
            DemandIndex.period_type == period
        ).first()
    else:
        demand = None
    
    if not demand:
        index_data = DemandIndexService.calculate_demand_index(
            db, attraction_id, period_start, period_end, period
        )
        demand = DemandIndex(**index_data)
        db.add(demand)
        db.commit()
        db.refresh(demand)
    
    attraction = db.query(TouristAttraction).filter(
        TouristAttraction.id == attraction_id
    ).first()
    province = db.query(Province).filter(
        Province.id == demand.province_id
    ).first()
    
    response = DemandIndexResponse.model_validate(demand)
    response.attraction_name = attraction.name if attraction else None
    response.province_name = province.name if province else None
    
    return response


@router.get("/demand/provinces/{province_id}", response_model=ProvinceDemandIndexResponse)
def get_province_demand(
    province_id: int,
    period: str = Query("week", description="week, month, quarter"),
    db: Session = Depends(get_db)
):
    period_days = {"week": 7, "month": 30, "quarter": 90}
    days = period_days.get(period, 7)
    
    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=days)
    
    # Get the most recent period for the given period_type
    latest_period = db.query(ProvinceDemandIndex).filter(
        ProvinceDemandIndex.period_type == period
    ).order_by(desc(ProvinceDemandIndex.period_end)).first()
    
    if latest_period:
        demand = db.query(ProvinceDemandIndex).filter(
            ProvinceDemandIndex.province_id == province_id,
            ProvinceDemandIndex.period_start == latest_period.period_start,
            ProvinceDemandIndex.period_end == latest_period.period_end,
            ProvinceDemandIndex.period_type == period
        ).first()
    else:
        demand = None
    
    if not demand:
        index_data = DemandIndexService.calculate_province_demand_index(
            db, province_id, period_start, period_end, period
        )
        if not index_data:
            raise HTTPException(status_code=404, detail="No data for province")
        demand = ProvinceDemandIndex(**index_data)
        db.add(demand)
        db.commit()
        db.refresh(demand)
    
    province = db.query(Province).filter(Province.id == province_id).first()
    
    # Convert to dict and add province_name before validation
    demand_dict = {
        "id": demand.id,
        "province_id": demand.province_id,
        "province_name": province.name if province else "Unknown",
        "period_start": demand.period_start,
        "period_end": demand.period_end,
        "period_type": demand.period_type,
        "overall_index": demand.overall_index,
        "comment_volume_score": demand.comment_volume_score,
        "sentiment_score": demand.sentiment_score,
        "engagement_score": demand.engagement_score,
        "growth_score": demand.growth_score,
        "total_comments": demand.total_comments,
        "total_attractions": demand.total_attractions,
        "active_attractions": demand.active_attractions,
        "positive_rate": demand.positive_rate,
        "avg_sentiment": demand.avg_sentiment,
        "growth_rate": demand.growth_rate,
        "rank_national": demand.rank_national,
        "calculated_at": demand.calculated_at
    }
    
    return ProvinceDemandIndexResponse(**demand_dict)


@router.get("/demand/attractions/{attraction_id}/analytics", response_model=DemandAnalytics)
def get_attraction_analytics(
    attraction_id: int,
    period: str = Query("week", description="week, month, quarter"),
    db: Session = Depends(get_db)
):
    period_days = {"week": 7, "month": 30, "quarter": 90}
    days = period_days.get(period, 7)
    
    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=days)
    
    current = db.query(DemandIndex).filter(
        DemandIndex.attraction_id == attraction_id,
        DemandIndex.period_start == period_start,
        DemandIndex.period_end == period_end
    ).first()
    
    if not current:
        index_data = DemandIndexService.calculate_demand_index(
            db, attraction_id, period_start, period_end, period
        )
        current = DemandIndex(**index_data)
        db.add(current)
        db.commit()
        db.refresh(current)
    
    prev_period_end = period_start
    prev_period_start = prev_period_end - timedelta(days=days)
    
    previous = db.query(DemandIndex).filter(
        DemandIndex.attraction_id == attraction_id,
        DemandIndex.period_start == prev_period_start,
        DemandIndex.period_end == prev_period_end
    ).first()
    
    if not previous:
        prev_index_data = DemandIndexService.calculate_demand_index(
            db, attraction_id, prev_period_start, prev_period_end, period
        )
        previous = DemandIndex(**prev_index_data)
    
    change = current.overall_index - previous.overall_index
    change_pct = (change / previous.overall_index * 100) if previous.overall_index > 0 else 0
    
    if change_pct > 10:
        trend = "increasing"
    elif change_pct < -10:
        trend = "decreasing"
    else:
        trend = "stable"
    
    trend_data = DemandIndexService.get_demand_trend(
        db, attraction_id, num_periods=12, period_type=period
    )
    
    from app.models.comment import Comment
    from sqlalchemy import func
    
    top_topics_query = db.query(
        func.jsonb_array_elements_text(Comment.topics).label('topic'),
        func.count().label('count')
    ).filter(
        Comment.attraction_id == attraction_id,
        Comment.comment_date >= period_start,
        Comment.comment_date < period_end,
        Comment.topics.isnot(None)
    ).group_by('topic').order_by(desc('count')).limit(5)
    
    top_topics = [{'topic': row.topic, 'count': row.count} for row in top_topics_query.all()]
    
    return DemandAnalytics(
        current_index=current.overall_index,
        previous_index=previous.overall_index,
        change_percentage=round(change_pct, 2),
        trend=trend,
        total_comments=current.total_comments,
        positive_rate=current.positive_rate,
        negative_rate=current.negative_rate,
        neutral_rate=current.neutral_rate,
        top_topics=top_topics,
        trend_data=trend_data
    )


@router.get("/demand/attractions/{attraction_id}/compare", response_model=ComparativeAnalysis)
def compare_attraction(
    attraction_id: int,
    period: str = Query("week", description="week, month, quarter"),
    db: Session = Depends(get_db)
):
    period_days = {"week": 7, "month": 30, "quarter": 90}
    days = period_days.get(period, 7)
    
    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=days)
    
    # Get the most recent period for the given period_type
    latest_period = db.query(DemandIndex).filter(
        DemandIndex.period_type == period
    ).order_by(desc(DemandIndex.period_end)).first()
    
    if not latest_period:
        raise HTTPException(status_code=404, detail="No demand data available for this period")
    
    current = db.query(DemandIndex).filter(
        DemandIndex.attraction_id == attraction_id,
        DemandIndex.period_start == latest_period.period_start,
        DemandIndex.period_end == latest_period.period_end,
        DemandIndex.period_type == period
    ).first()
    
    if not current:
        raise HTTPException(status_code=404, detail="Demand data not found for this attraction")
    
    attraction = db.query(TouristAttraction).filter(
        TouristAttraction.id == attraction_id
    ).first()
    province = db.query(Province).filter(Province.id == current.province_id).first()
    
    avg_province = db.query(func.avg(DemandIndex.overall_index)).filter(
        DemandIndex.province_id == current.province_id,
        DemandIndex.period_start == latest_period.period_start,
        DemandIndex.period_end == latest_period.period_end,
        DemandIndex.period_type == period
    ).scalar() or 0
    
    avg_national = db.query(func.avg(DemandIndex.overall_index)).filter(
        DemandIndex.period_start == latest_period.period_start,
        DemandIndex.period_end == latest_period.period_end,
        DemandIndex.period_type == period
    ).scalar() or 0
    
    diff_province = ((current.overall_index - avg_province) / avg_province * 100) if avg_province > 0 else 0
    diff_national = ((current.overall_index - avg_national) / avg_national * 100) if avg_national > 0 else 0
    
    if current.overall_index > avg_province * 1.2:
        performance = "excellent"
    elif current.overall_index > avg_province:
        performance = "good"
    elif current.overall_index > avg_province * 0.8:
        performance = "average"
    else:
        performance = "below_average"
    
    return ComparativeAnalysis(
        attraction_id=attraction_id,
        attraction_name=attraction.name,
        province_name=province.name,
        current_index=current.overall_index,
        rank_in_province=current.rank_in_province,
        rank_national=current.rank_national,
        compared_to_avg_province=round(diff_province, 2),
        compared_to_avg_national=round(diff_national, 2),
        performance=performance
    )


@router.post("/demand/calculate")
def calculate_demand_indexes(
    period: str = Query("week", description="week, month, quarter"),
    db: Session = Depends(get_db)
):
    period_days = {"week": 7, "month": 30, "quarter": 90}
    days = period_days.get(period, 7)
    
    period_end = datetime.now(timezone.utc)
    period_start = period_end - timedelta(days=days)
    
    result = DemandIndexService.calculate_and_store_all_indexes(
        db, period_start, period_end, period
    )
    
    return {
        "success": True,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "attractions_calculated": result['attractions'],
        "provinces_calculated": result['provinces']
    }
