from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models.comment import Comment
from app.models.tourist_attraction import TouristAttraction

router = APIRouter()


@router.get("/aspects/summary")
async def get_aspect_summary(
    attraction_id: Optional[int] = Query(None, description="Filter by attraction"),
    province_id: Optional[int] = Query(None, description="Filter by province"),
    db: Session = Depends(get_db)
):
    query = db.query(Comment).filter(
        Comment.is_meaningful == True,
        Comment.topics.isnot(None)
    )
    
    if attraction_id:
        query = query.filter(Comment.attraction_id == attraction_id)
    
    if province_id:
        query = query.join(TouristAttraction).filter(
            TouristAttraction.province_id == province_id
        )
    
    comments = query.all()
    
    aspect_stats = {}
    
    for comment in comments:
        if not comment.aspect_sentiments:
            continue
            
        for aspect, sentiment in comment.aspect_sentiments.items():
            if aspect not in aspect_stats:
                aspect_stats[aspect] = {
                    'total': 0,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            
            aspect_stats[aspect]['total'] += 1
            aspect_stats[aspect][sentiment] += 1
    
    result = {}
    for aspect, stats in aspect_stats.items():
        total = stats['total']
        result[aspect] = {
            'total_comments': total,
            'positive': {
                'count': stats['positive'],
                'percentage': round(stats['positive'] / total * 100, 1)
            },
            'negative': {
                'count': stats['negative'],
                'percentage': round(stats['negative'] / total * 100, 1)
            },
            'neutral': {
                'count': stats['neutral'],
                'percentage': round(stats['neutral'] / total * 100, 1)
            },
            'sentiment_score': round(
                (stats['positive'] - stats['negative']) / total * 100, 1
            )
        }
    
    result = dict(sorted(
        result.items(), 
        key=lambda x: x[1]['total_comments'], 
        reverse=True
    ))
    
    return {
        'summary': result,
        'total_comments_analyzed': len(comments)
    }


@router.get("/aspects/details")
async def get_aspect_details(
    aspect: str = Query(..., description="Aspect name (scenery, food, service, etc.)"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment (positive/negative/neutral)"),
    attraction_id: Optional[int] = Query(None, description="Filter by attraction"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(Comment).filter(
        Comment.is_meaningful == True,
        Comment.aspect_sentiments.contains({aspect: sentiment} if sentiment else {aspect: None})
    )
    
    if attraction_id:
        query = query.filter(Comment.attraction_id == attraction_id)
    
    total = query.count()
    comments = query.order_by(Comment.comment_date.desc()).offset(offset).limit(limit).all()
    
    return {
        'aspect': aspect,
        'sentiment_filter': sentiment,
        'total': total,
        'comments': [
            {
                'id': c.id,
                'content': c.content,
                'sentiment': c.aspect_sentiments.get(aspect) if c.aspect_sentiments else None,
                'overall_sentiment': c.sentiment,
                'attraction_id': c.attraction_id,
                'platform': c.platform,
                'comment_date': c.comment_date,
                'like_count': c.like_count,
                'quality_score': c.quality_score
            }
            for c in comments
        ],
        'pagination': {
            'limit': limit,
            'offset': offset,
            'total': total,
            'has_more': offset + limit < total
        }
    }


@router.get("/aspects/comparison")
async def compare_attractions_by_aspect(
    attraction_ids: str = Query(..., description="Comma-separated attraction IDs"),
    db: Session = Depends(get_db)
):
    ids = [int(x.strip()) for x in attraction_ids.split(',')]
    
    results = {}
    
    for attraction_id in ids:
        attraction = db.query(TouristAttraction).filter(
            TouristAttraction.id == attraction_id
        ).first()
        
        if not attraction:
            continue
        
        comments = db.query(Comment).filter(
            Comment.attraction_id == attraction_id,
            Comment.is_meaningful == True,
            Comment.aspect_sentiments.isnot(None)
        ).all()
        
        aspect_stats = {}
        for comment in comments:
            if not comment.aspect_sentiments:
                continue
                
            for aspect, sentiment in comment.aspect_sentiments.items():
                if aspect not in aspect_stats:
                    aspect_stats[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0}
                aspect_stats[aspect][sentiment] += 1
        
        aspect_scores = {}
        for aspect, stats in aspect_stats.items():
            total = sum(stats.values())
            aspect_scores[aspect] = {
                'score': round((stats['positive'] - stats['negative']) / total * 100, 1),
                'total': total,
                'positive_pct': round(stats['positive'] / total * 100, 1),
                'negative_pct': round(stats['negative'] / total * 100, 1)
            }
        
        results[attraction.name] = {
            'id': attraction_id,
            'aspects': aspect_scores,
            'total_comments': len(comments)
        }
    
    return {
        'comparison': results,
        'attractions_count': len(results)
    }


@router.get("/aspects/trending")
async def get_trending_aspects(
    days: int = Query(30, description="Look back period in days"),
    province_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    from datetime import datetime, timedelta
    
    since_date = datetime.now() - timedelta(days=days)
    
    query = db.query(Comment).filter(
        Comment.is_meaningful == True,
        Comment.topics.isnot(None),
        Comment.comment_date >= since_date
    )
    
    if province_id:
        query = query.join(TouristAttraction).filter(
            TouristAttraction.province_id == province_id
        )
    
    comments = query.all()
    
    topic_counts = {}
    topic_sentiments = {}
    
    for comment in comments:
        if not comment.topics:
            continue
            
        for topic in comment.topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            if topic not in topic_sentiments:
                topic_sentiments[topic] = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            # Get sentiment for this topic
            if comment.aspect_sentiments and topic in comment.aspect_sentiments:
                sentiment = comment.aspect_sentiments[topic]
                topic_sentiments[topic][sentiment] += 1
    
    results = []
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        sentiments = topic_sentiments[topic]
        total = sum(sentiments.values())
        
        results.append({
            'aspect': topic,
            'mention_count': count,
            'sentiment_breakdown': {
                'positive': {
                    'count': sentiments['positive'],
                    'percentage': round(sentiments['positive'] / total * 100, 1) if total > 0 else 0
                },
                'negative': {
                    'count': sentiments['negative'],
                    'percentage': round(sentiments['negative'] / total * 100, 1) if total > 0 else 0
                },
                'neutral': {
                    'count': sentiments['neutral'],
                    'percentage': round(sentiments['neutral'] / total * 100, 1) if total > 0 else 0
                }
            },
            'overall_sentiment': 'positive' if sentiments['positive'] > sentiments['negative'] else 'negative' if sentiments['negative'] > sentiments['positive'] else 'neutral'
        })
    
    return {
        'period_days': days,
        'trending_aspects': results,
        'total_comments_analyzed': len(comments)
    }
