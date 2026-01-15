from sqlalchemy.orm import Session
from sqlalchemy import func, case, and_, desc
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import logging

from app.models.demand_index import DemandIndex, ProvinceDemandIndex
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province
from app.models.comment import Comment

logger = logging.getLogger(__name__)


class DemandIndexService:
    
    @staticmethod
    def calculate_demand_index(
        db: Session,
        attraction_id: int,
        period_start: datetime,
        period_end: datetime,
        period_type: str = "week"
    ) -> Dict:
        
        attraction = db.query(TouristAttraction).filter(
            TouristAttraction.id == attraction_id
        ).first()
        
        if not attraction:
            raise ValueError(f"Attraction {attraction_id} not found")
        
        stats = db.query(
            func.count(Comment.id).label('total_comments'),
            func.sum(case((Comment.sentiment == 'positive', 1), else_=0)).label('positive'),
            func.sum(case((Comment.sentiment == 'negative', 1), else_=0)).label('negative'),
            func.sum(case((Comment.sentiment == 'neutral', 1), else_=0)).label('neutral'),
            func.avg(Comment.sentiment_score).label('avg_sentiment'),
            func.sum(Comment.like_count + Comment.reply_count).label('total_engagement')
        ).filter(
            Comment.attraction_id == attraction_id,
            Comment.comment_date >= period_start,
            Comment.comment_date < period_end,
            Comment.is_meaningful == True
        ).first()
        
        total_comments = stats.total_comments or 0
        positive = stats.positive or 0
        negative = stats.negative or 0
        neutral = stats.neutral or 0
        avg_sentiment = float(stats.avg_sentiment or 0)
        total_engagement = stats.total_engagement or 0
        
        positive_rate = (positive / total_comments * 100) if total_comments > 0 else 0
        negative_rate = (negative / total_comments * 100) if total_comments > 0 else 0
        neutral_rate = (neutral / total_comments * 100) if total_comments > 0 else 0
        
        prev_period_start = period_start - (period_end - period_start)
        prev_period_end = period_start
        
        prev_stats = db.query(
            func.count(Comment.id).label('prev_total')
        ).filter(
            Comment.attraction_id == attraction_id,
            Comment.comment_date >= prev_period_start,
            Comment.comment_date < prev_period_end,
            Comment.is_meaningful == True
        ).first()
        
        prev_total = prev_stats.prev_total or 0
        growth_rate = ((total_comments - prev_total) / prev_total * 100) if prev_total > 0 else 0
        
        attraction_comment_counts = db.query(
            Comment.attraction_id,
            func.count(Comment.id).label('comment_count')
        ).filter(
            Comment.comment_date >= period_start,
            Comment.comment_date < period_end,
            Comment.is_meaningful == True
        ).group_by(Comment.attraction_id).all()
        
        max_comments = max([count.comment_count for count in attraction_comment_counts]) if attraction_comment_counts else 1
        
        comment_volume_score = min((total_comments / max(max_comments, 1)) * 100, 100)
        
        sentiment_score = ((positive_rate * 1.0) + (neutral_rate * 0.5) + (negative_rate * 0.0))
        
        avg_engagement_per_comment = (total_engagement / total_comments) if total_comments > 0 else 0
        engagement_score = min(avg_engagement_per_comment * 2, 100)
        
        growth_score = min(max(growth_rate + 50, 0), 100)
        
        overall_index = (
            comment_volume_score * 0.40 +
            sentiment_score * 0.30 +
            engagement_score * 0.20 +
            growth_score * 0.10
        )
        
        return {
            'attraction_id': attraction_id,
            'province_id': attraction.province_id,
            'period_start': period_start,
            'period_end': period_end,
            'period_type': period_type,
            'overall_index': round(overall_index, 2),
            'comment_volume_score': round(comment_volume_score, 2),
            'sentiment_score': round(sentiment_score, 2),
            'engagement_score': round(engagement_score, 2),
            'growth_score': round(growth_score, 2),
            'total_comments': total_comments,
            'positive_rate': round(positive_rate, 2),
            'negative_rate': round(negative_rate, 2),
            'neutral_rate': round(neutral_rate, 2),
            'avg_sentiment': round(avg_sentiment, 2),
            'total_engagement': total_engagement,
            'growth_rate': round(growth_rate, 2)
        }
    
    @staticmethod
    def calculate_province_demand_index(
        db: Session,
        province_id: int,
        period_start: datetime,
        period_end: datetime,
        period_type: str = "week"
    ) -> Dict:
        
        province = db.query(Province).filter(Province.id == province_id).first()
        if not province:
            raise ValueError(f"Province {province_id} not found")
        
        attraction_ids = db.query(TouristAttraction.id).filter(
            TouristAttraction.province_id == province_id,
            TouristAttraction.is_active == True
        ).all()
        attraction_ids = [a.id for a in attraction_ids]
        
        if not attraction_ids:
            return None
        
        stats = db.query(
            func.count(Comment.id).label('total_comments'),
            func.sum(case((Comment.sentiment == 'positive', 1), else_=0)).label('positive'),
            func.avg(Comment.sentiment_score).label('avg_sentiment'),
            func.sum(Comment.like_count + Comment.reply_count).label('total_engagement')
        ).filter(
            Comment.attraction_id.in_(attraction_ids),
            Comment.comment_date >= period_start,
            Comment.comment_date < period_end,
            Comment.is_meaningful == True
        ).first()
        
        total_comments = stats.total_comments or 0
        positive = stats.positive or 0
        avg_sentiment = float(stats.avg_sentiment or 0)
        total_engagement = stats.total_engagement or 0
        
        positive_rate = (positive / total_comments * 100) if total_comments > 0 else 0
        
        prev_period_start = period_start - (period_end - period_start)
        prev_period_end = period_start
        
        prev_stats = db.query(
            func.count(Comment.id).label('prev_total')
        ).filter(
            Comment.attraction_id.in_(attraction_ids),
            Comment.comment_date >= prev_period_start,
            Comment.comment_date < prev_period_end,
            Comment.is_meaningful == True
        ).first()
        
        prev_total = prev_stats.prev_total or 0
        growth_rate = ((total_comments - prev_total) / prev_total * 100) if prev_total > 0 else 0
        
        total_attractions = len(attraction_ids)
        active_attractions = db.query(func.count(func.distinct(Comment.attraction_id))).filter(
            Comment.attraction_id.in_(attraction_ids),
            Comment.comment_date >= period_start,
            Comment.comment_date < period_end
        ).scalar() or 0
        
        comment_volume_score = min((total_comments / (total_attractions * 10)) * 100, 100)
        sentiment_score = positive_rate
        engagement_score = min((total_engagement / max(total_comments, 1)) * 2, 100)
        growth_score = min(max(growth_rate + 50, 0), 100)
        
        overall_index = (
            comment_volume_score * 0.40 +
            sentiment_score * 0.30 +
            engagement_score * 0.20 +
            growth_score * 0.10
        )
        
        return {
            'province_id': province_id,
            'period_start': period_start,
            'period_end': period_end,
            'period_type': period_type,
            'overall_index': round(overall_index, 2),
            'comment_volume_score': round(comment_volume_score, 2),
            'sentiment_score': round(sentiment_score, 2),
            'engagement_score': round(engagement_score, 2),
            'growth_score': round(growth_score, 2),
            'total_comments': total_comments,
            'total_attractions': total_attractions,
            'active_attractions': active_attractions,
            'positive_rate': round(positive_rate, 2),
            'avg_sentiment': round(avg_sentiment, 2),
            'growth_rate': round(growth_rate, 2)
        }
    
    @staticmethod
    def calculate_and_store_all_indexes(
        db: Session,
        period_start: datetime,
        period_end: datetime,
        period_type: str = "week"
    ):
        
        logger.info(f"Calculating demand indexes for period {period_start} to {period_end}")
        
        existing_attractions = db.query(DemandIndex).filter(
            DemandIndex.period_start == period_start,
            DemandIndex.period_end == period_end,
            DemandIndex.period_type == period_type
        ).delete()
        
        existing_provinces = db.query(ProvinceDemandIndex).filter(
            ProvinceDemandIndex.period_start == period_start,
            ProvinceDemandIndex.period_end == period_end,
            ProvinceDemandIndex.period_type == period_type
        ).delete()
        
        db.commit()
        
        logger.info(f"Deleted {existing_attractions} old attraction indexes and {existing_provinces} old province indexes")
        
        attractions = db.query(TouristAttraction).filter(
            TouristAttraction.is_active == True
        ).all()
        
        attraction_count = 0
        for attraction in attractions:
            try:
                index_data = DemandIndexService.calculate_demand_index(
                    db, attraction.id, period_start, period_end, period_type
                )
                
                demand_index = DemandIndex(**index_data)
                db.add(demand_index)
                db.commit()
                attraction_count += 1
                
            except Exception as e:
                logger.error(f"Error calculating index for attraction {attraction.id}: {e}")
                db.rollback()
                continue
        
        logger.info(f"Calculated {attraction_count} attraction indexes")
        
        attraction_indexes = db.query(DemandIndex).filter(
            DemandIndex.period_start == period_start,
            DemandIndex.period_end == period_end,
            DemandIndex.period_type == period_type
        ).order_by(desc(DemandIndex.overall_index)).all()
        
        for rank, idx in enumerate(attraction_indexes, 1):
            idx.rank_national = rank
        
        province_groups = {}
        for idx in attraction_indexes:
            if idx.province_id not in province_groups:
                province_groups[idx.province_id] = []
            province_groups[idx.province_id].append(idx)
        
        for province_id, indexes in province_groups.items():
            indexes.sort(key=lambda x: x.overall_index, reverse=True)
            for rank, idx in enumerate(indexes, 1):
                idx.rank_in_province = rank
        
        db.commit()
        logger.info("Updated attraction rankings")
        
        provinces = db.query(Province).all()
        province_count = 0
        
        for province in provinces:
            try:
                index_data = DemandIndexService.calculate_province_demand_index(
                    db, province.id, period_start, period_end, period_type
                )
                
                if index_data:
                    prov_index = ProvinceDemandIndex(**index_data)
                    db.add(prov_index)
                    db.commit()
                    province_count += 1
                    
            except Exception as e:
                logger.error(f"Error calculating index for province {province.id}: {e}")
                db.rollback()
                continue
        
        logger.info(f"Calculated {province_count} province indexes")
        
        province_indexes = db.query(ProvinceDemandIndex).filter(
            ProvinceDemandIndex.period_start == period_start,
            ProvinceDemandIndex.period_end == period_end,
            ProvinceDemandIndex.period_type == period_type
        ).order_by(desc(ProvinceDemandIndex.overall_index)).all()
        
        for rank, idx in enumerate(province_indexes, 1):
            idx.rank_national = rank
        
        db.commit()
        logger.info("Updated province rankings")
        
        logger.info(f"Successfully calculated {attraction_count} attractions and {province_count} provinces")
        
        return {
            'attractions': attraction_count,
            'provinces': province_count
        }
        
        logger.info(f"Calculated {len(attraction_indexes)} attraction indexes and {len(province_indexes)} province indexes")
        
        return {
            'attractions': len(attraction_indexes),
            'provinces': len(province_indexes)
        }
    
    @staticmethod
    def get_top_attractions(
        db: Session,
        period_start: datetime,
        period_end: datetime,
        limit: int = 10,
        province_id: Optional[int] = None,
        period_type: str = "month"
    ) -> List[Dict]:
        
        # Get the most recent period for the given period_type
        latest_period = db.query(DemandIndex).filter(
            DemandIndex.period_type == period_type
        ).order_by(desc(DemandIndex.period_end)).first()
        
        if not latest_period:
            return []
        
        query = db.query(
            DemandIndex,
            TouristAttraction.name,
            Province.name.label('province_name')
        ).join(
            TouristAttraction, DemandIndex.attraction_id == TouristAttraction.id
        ).join(
            Province, DemandIndex.province_id == Province.id
        ).filter(
            DemandIndex.period_start == latest_period.period_start,
            DemandIndex.period_end == latest_period.period_end,
            DemandIndex.period_type == period_type
        )
        
        if province_id:
            query = query.filter(DemandIndex.province_id == province_id)
        
        query = query.order_by(desc(DemandIndex.overall_index)).limit(limit)
        
        results = []
        for idx, attr_name, prov_name in query.all():
            results.append({
                'id': idx.attraction_id,
                'name': attr_name,
                'province_name': prov_name,
                'overall_index': idx.overall_index,
                'total_comments': idx.total_comments,
                'positive_rate': idx.positive_rate,
                'growth_rate': idx.growth_rate,
                'rank': (idx.rank_in_province if province_id else idx.rank_national) or 0
            })
        
        return results
    
    @staticmethod
    def get_top_provinces(
        db: Session,
        period_start: datetime,
        period_end: datetime,
        limit: int = 10,
        period_type: str = "month"
    ) -> List[Dict]:
        
        # Get the most recent period for the given period_type
        latest_period = db.query(ProvinceDemandIndex).filter(
            ProvinceDemandIndex.period_type == period_type
        ).order_by(desc(ProvinceDemandIndex.period_end)).first()
        
        if not latest_period:
            return []
        
        query = db.query(
            ProvinceDemandIndex,
            Province.name
        ).join(
            Province, ProvinceDemandIndex.province_id == Province.id
        ).filter(
            ProvinceDemandIndex.period_start == latest_period.period_start,
            ProvinceDemandIndex.period_end == latest_period.period_end,
            ProvinceDemandIndex.period_type == period_type
        ).order_by(
            desc(ProvinceDemandIndex.overall_index)
        ).limit(limit)
        
        results = []
        for idx, prov_name in query.all():
            results.append({
                'id': idx.province_id,
                'name': prov_name,
                'overall_index': idx.overall_index,
                'total_comments': idx.total_comments,
                'active_attractions': idx.active_attractions,
                'positive_rate': idx.positive_rate,
                'growth_rate': idx.growth_rate,
                'rank': idx.rank_national or 0
            })
        
        return results
    
    @staticmethod
    def get_demand_trend(
        db: Session,
        attraction_id: int,
        num_periods: int = 12,
        period_type: str = "week"
    ) -> List[Dict]:
        
        query = db.query(DemandIndex).filter(
            DemandIndex.attraction_id == attraction_id,
            DemandIndex.period_type == period_type
        ).order_by(
            desc(DemandIndex.period_start)
        ).limit(num_periods)
        
        results = []
        for idx in query.all():
            results.append({
                'date': idx.period_start,
                'overall_index': idx.overall_index,
                'comment_volume': idx.total_comments,
                'sentiment_score': idx.sentiment_score
            })
        
        return list(reversed(results))
