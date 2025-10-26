from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.province import Province
from app.models.tourist_attraction import TouristAttraction
from app.schemas.province import Province as ProvinceSchema, ProvinceWithStats
from app.schemas.attraction import TouristAttraction as AttractionSchema


class ProvinceService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_all_provinces(self) -> List[ProvinceSchema]:
        provinces = self.db.query(Province).all()
        return [ProvinceSchema.from_orm(province) for province in provinces]
    
    def get_province_by_id(self, province_id: int) -> Optional[ProvinceSchema]:
        province = self.db.query(Province).filter(Province.id == province_id).first()
        return ProvinceSchema.from_orm(province) if province else None
    
    def get_province_by_code(self, province_code: str) -> Optional[ProvinceSchema]:
        province = self.db.query(Province).filter(Province.code == province_code).first()
        return ProvinceSchema.from_orm(province) if province else None
    
    def get_attractions_by_province_id(self, province_id: int, active_only: bool = True) -> List[AttractionSchema]:
        query = self.db.query(TouristAttraction).filter(TouristAttraction.province_id == province_id)
        
        if active_only:
            query = query.filter(TouristAttraction.is_active.is_(True))
        
        attractions = query.all()
        return [AttractionSchema.from_orm(attraction) for attraction in attractions]
    
    def get_attractions_by_province_code(self, province_code: str, active_only: bool = True) -> List[AttractionSchema]:
        province = self.get_province_by_code(province_code)
        if not province:
            return []
        
        return self.get_attractions_by_province_id(province.id, active_only)
    
    def get_attractions_with_province_info(self, province_id: int, active_only: bool = True) -> List[AttractionSchema]:
        query = self.db.query(TouristAttraction).join(Province).filter(
            TouristAttraction.province_id == province_id
        )
        
        if active_only:
            query = query.filter(TouristAttraction.is_active.is_(True))
        
        attractions = query.all()
        return [AttractionSchema.from_orm(attraction) for attraction in attractions]
    
    def get_province_with_stats(self, province_id: int) -> Optional[ProvinceWithStats]:
        province = self.db.query(Province).filter(Province.id == province_id).first()
        if not province:
            return None
        
        total_attractions = self.db.query(TouristAttraction).filter(
            TouristAttraction.province_id == province_id
        ).count()
        
        active_attractions = self.db.query(TouristAttraction).filter(
            TouristAttraction.province_id == province_id,
            TouristAttraction.is_active.is_(True)
        ).count()
        
        total_comments = 0
        avg_sentiment = 0.0
        
        return ProvinceWithStats(
            id=province.id,
            name=province.name,
            code=province.code,
            main_city=province.main_city,
            created_at=province.created_at,
            total_attractions=total_attractions,
            active_attractions=active_attractions,
            total_comments=total_comments,
            avg_sentiment=avg_sentiment
        )
    
    def search_attractions_by_name(self, province_id: int, search_term: str) -> List[AttractionSchema]:
        attractions = self.db.query(TouristAttraction).filter(
            TouristAttraction.province_id == province_id,
            TouristAttraction.name.ilike(f"%{search_term}%"),
            TouristAttraction.is_active.is_(True)
        ).all()
        
        return [AttractionSchema.from_orm(attraction) for attraction in attractions]
    
    def get_attractions_by_category(self, province_id: int, category: str) -> List[AttractionSchema]:
        attractions = self.db.query(TouristAttraction).filter(
            TouristAttraction.province_id == province_id,
            TouristAttraction.category == category,
            TouristAttraction.is_active.is_(True)
        ).all()
        
        return [AttractionSchema.from_orm(attraction) for attraction in attractions]