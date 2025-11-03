from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from sqlalchemy import and_

from app.database.connection import get_db
from app.services.tourist_attraction_service import TouristAttractionService
from app.schemas.attraction import AttractionDetailStats, AspectSentiment
from app.models.comment import Comment

router = APIRouter()


@router.get("/{attraction_id}/detail-stats", response_model=AttractionDetailStats)
async def get_attraction_detail_stats(
    attraction_id: int,
    months: int = Query(6, ge=1, le=24, description="Number of months for statistics (default 6 months)"),
    db: Session = Depends(get_db)
):
    service = TouristAttractionService(db)
    attraction = service.get_attraction_with_province(attraction_id)
    if not attraction:
        raise HTTPException(status_code=404, detail="Attraction not found")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    comments_query = db.query(Comment).filter(
        and_(
            Comment.attraction_id == attraction_id,
            Comment.is_meaningful == True,  
            Comment.scraped_at >= start_date, 
            Comment.sentiment.isnot(None)
        )
    )
    
    total_comments = comments_query.count()
    all_comments = comments_query.all()
    
    sentiment_counts = {
        'positive': sum(1 for c in all_comments if c.sentiment == 'positive'),
        'negative': sum(1 for c in all_comments if c.sentiment == 'negative'),
        'neutral': sum(1 for c in all_comments if c.sentiment == 'neutral')
    }
    
    avg_score = 0
    if total_comments > 0:
        avg_score = (sentiment_counts['positive'] - sentiment_counts['negative']) / total_comments * 100
    
    overall_sentiment = {
        **sentiment_counts,
        'total': total_comments,
        'avg_score': round(avg_score, 2),
        'positive_percentage': round(sentiment_counts['positive'] / total_comments * 100, 1) if total_comments > 0 else 0
    }
    
    aspects_list = ['scenery', 'activities', 'food', 'facilities', 'accessibility', 'pricing', 'service']
    aspect_results = []
    
    for aspect in aspects_list:
        aspect_comments = [c for c in all_comments if c.topics and aspect in c.topics]
        
        if not aspect_comments:
            continue
            
        pos = sum(1 for c in aspect_comments if c.aspect_sentiments and 
                 c.aspect_sentiments.get(aspect) == 'positive')
        neg = sum(1 for c in aspect_comments if c.aspect_sentiments and 
                 c.aspect_sentiments.get(aspect) == 'negative')
        neu = sum(1 for c in aspect_comments if c.aspect_sentiments and 
                 c.aspect_sentiments.get(aspect) == 'neutral')
        
        total = len(aspect_comments)
        pos_pct = (pos / total * 100) if total > 0 else 0
        score = ((pos - neg) / total * 100) if total > 0 else 0
        
        aspect_results.append(AspectSentiment(
            aspect=aspect,
            total_mentions=total,
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            positive_percentage=round(pos_pct, 1),
            sentiment_score=round(score, 1)
        ))
    
    aspect_results.sort(key=lambda x: x.total_mentions, reverse=True)
    
    positive_comments = [c for c in all_comments if c.sentiment == 'positive']
    negative_comments = [c for c in all_comments if c.sentiment == 'negative']
    
    positive_comments.sort(key=lambda x: x.quality_score or 0, reverse=True)
    negative_comments.sort(key=lambda x: x.quality_score or 0, reverse=True)
    
    top_positive = [
        {
            'id': c.id,
            'text': c.cleaned_content or c.content, 
            'platform': c.platform,
            'author_name': c.author,  
            'comment_date': c.comment_date.isoformat() if c.comment_date else None,
            'scraped_at': c.scraped_at.isoformat(),
            'quality_score': c.quality_score,
            'quality_tier': c.quality_tier,
            'topics': c.topics,
            'aspect_sentiments': c.aspect_sentiments
        }
        for c in positive_comments[:5]
    ]
    
    top_negative = [
        {
            'id': c.id,
            'text': c.cleaned_content or c.content,
            'platform': c.platform,
            'author_name': c.author,
            'comment_date': c.comment_date.isoformat() if c.comment_date else None,
            'scraped_at': c.scraped_at.isoformat(),
            'quality_score': c.quality_score,
            'quality_tier': c.quality_tier,
            'topics': c.topics,
            'aspect_sentiments': c.aspect_sentiments
        }
        for c in negative_comments[:5]
    ]
    
    return AttractionDetailStats(
        attraction=attraction,
        total_comments_6months=total_comments,
        meaningful_comments_6months=total_comments, 
        overall_sentiment=overall_sentiment,
        aspects=aspect_results,
        top_positive_comments=top_positive,
        top_negative_comments=top_negative
    )


@router.get("/{attraction_id}/comments-by-aspect")
async def get_comments_by_aspect(
    attraction_id: int,
    aspect: str = Query(..., description="Aspect: scenery, activities, food, facilities, accessibility, pricing, service"),
    sentiment: str = Query(None, description="Filter by sentiment: positive, negative, neutral (optional)"),
    months: int = Query(6, ge=1, le=24, description="Number of months for statistics"),
    limit: int = Query(50, ge=1, le=200, description="Number of comments to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db)
):
    valid_aspects = ['scenery', 'activities', 'food', 'facilities', 'accessibility', 'pricing', 'service']
    if aspect not in valid_aspects:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid aspect. Must be one of: {', '.join(valid_aspects)}"
        )
    
    if sentiment and sentiment not in ['positive', 'negative', 'neutral']:
        raise HTTPException(
            status_code=400,
            detail="Invalid sentiment. Must be: positive, negative, or neutral"
        )
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    query = db.query(Comment).filter(
        and_(
            Comment.attraction_id == attraction_id,
            Comment.is_meaningful == True,
            Comment.scraped_at >= start_date,
            Comment.sentiment.isnot(None),
            Comment.topics.isnot(None)
        )
    )
    
    query = query.filter(Comment.topics.contains([aspect]))
    
    if sentiment:
        query = query.filter(Comment.sentiment == sentiment)
    
    total_count = query.count()
    
    comments = query.order_by(
        Comment.quality_score.desc(),
        Comment.scraped_at.desc()
    ).limit(limit).offset(offset).all()
    
    result = {
        'attraction_id': attraction_id,
        'aspect': aspect,
        'sentiment_filter': sentiment,
        'total_count': total_count,
        'returned_count': len(comments),
        'limit': limit,
        'offset': offset,
        'has_more': (offset + len(comments)) < total_count,
        'comments': [
            {
                'id': c.id,
                'text': c.cleaned_content or c.content,
                'platform': c.platform,
                'author': c.author,
                'comment_date': c.comment_date.isoformat() if c.comment_date else None,
                'scraped_at': c.scraped_at.isoformat(),
                'sentiment': c.sentiment,
                'quality_tier': c.quality_tier,
                'quality_score': c.quality_score,
                'like_count': c.like_count,
                'topics': c.topics,
                'aspect_sentiments': c.aspect_sentiments,
                'language': c.language
            }
            for c in comments
        ]
    }
    
    return result
