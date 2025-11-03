from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from app.models.tourist_attraction import TouristAttraction
from app.models.province import Province
from app.schemas.attraction import (
    TouristAttraction as AttractionSchema,
    TouristAttractionCreate,
    TouristAttractionUpdate,
    TouristAttractionWithProvince
)


class TouristAttractionService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_all_attractions(self, active_only: bool = True) -> List[AttractionSchema]:
        query = self.db.query(TouristAttraction)
        
        if active_only:
            query = query.filter(TouristAttraction.is_active.is_(True))
        
        attractions = query.all()
        return [AttractionSchema.model_validate(attraction) for attraction in attractions]
    
    def get_attraction_by_id(self, attraction_id: int) -> Optional[AttractionSchema]:
        attraction = self.db.query(TouristAttraction).filter(
            TouristAttraction.id == attraction_id
        ).first()
        return AttractionSchema.model_validate(attraction) if attraction else None
    
    def get_attraction_with_province(self, attraction_id: int) -> Optional[TouristAttractionWithProvince]:
        attraction = self.db.query(TouristAttraction).join(Province).filter(
            TouristAttraction.id == attraction_id
        ).first()
        
        if not attraction:
            return None
            
        return TouristAttractionWithProvince(
            **AttractionSchema.model_validate(attraction).model_dump(),
            province=attraction.province
        )
    
    def create_attraction(self, attraction_data: TouristAttractionCreate) -> AttractionSchema:
        db_attraction = TouristAttraction(**attraction_data.model_dump())
        self.db.add(db_attraction)
        self.db.commit()
        self.db.refresh(db_attraction)
        return AttractionSchema.model_validate(db_attraction)
    
    def update_attraction(self, attraction_id: int, update_data: TouristAttractionUpdate) -> Optional[AttractionSchema]:
        attraction = self.db.query(TouristAttraction).filter(
            TouristAttraction.id == attraction_id
        ).first()
        
        if not attraction:
            return None
        
        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(attraction, field, value)
        
        self.db.commit()
        self.db.refresh(attraction)
        return AttractionSchema.from_orm(attraction)
    
    def deactivate_attraction(self, attraction_id: int) -> bool:
        attraction = self.db.query(TouristAttraction).filter(
            TouristAttraction.id == attraction_id
        ).first()
        
        if not attraction:
            return False
        
        attraction.is_active = False
        self.db.commit()
        return True
    
    def search_attractions(self, 
                         search_term: Optional[str] = None,
                         province_id: Optional[int] = None,
                         category: Optional[str] = None,
                         min_rating: Optional[float] = None,
                         active_only: bool = True) -> List[AttractionSchema]:
        query = self.db.query(TouristAttraction)
        
        conditions = []
        
        if active_only:
            conditions.append(TouristAttraction.is_active.is_(True))
        
        if search_term:
            search_condition = or_(
                TouristAttraction.name.ilike(f"%{search_term}%"),
                TouristAttraction.description.ilike(f"%{search_term}%"),
                TouristAttraction.address.ilike(f"%{search_term}%")
            )
            conditions.append(search_condition)
        
        if province_id:
            conditions.append(TouristAttraction.province_id == province_id)
        
        if category:
            conditions.append(TouristAttraction.category == category)
        
        if min_rating:
            conditions.append(TouristAttraction.rating >= min_rating)
        
        if conditions:
            query = query.filter(and_(*conditions))
        
        attractions = query.all()
        return [AttractionSchema.from_orm(attraction) for attraction in attractions]
    
    def get_popular_attractions(self, province_id: Optional[int] = None, limit: int = 10) -> List[AttractionSchema]:
        query = self.db.query(TouristAttraction).filter(
            TouristAttraction.is_active.is_(True)
        )
        
        if province_id:
            query = query.filter(TouristAttraction.province_id == province_id)
        
        attractions = query.order_by(
            TouristAttraction.rating.desc(),
            TouristAttraction.total_reviews.desc()
        ).limit(limit).all()
        
        return [AttractionSchema.from_orm(attraction) for attraction in attractions]
    
    def get_attractions_by_category(self, category: str, province_id: Optional[int] = None) -> List[AttractionSchema]:
        query = self.db.query(TouristAttraction).filter(
            TouristAttraction.category == category,
            TouristAttraction.is_active.is_(True)
        )
        
        if province_id:
            query = query.filter(TouristAttraction.province_id == province_id)
        
        attractions = query.all()
        return [AttractionSchema.from_orm(attraction) for attraction in attractions]
    
    def get_categories(self, province_id: Optional[int] = None) -> List[str]:
        query = self.db.query(TouristAttraction.category).filter(
            TouristAttraction.category.isnot(None),
            TouristAttraction.is_active.is_(True)
        ).distinct()
        
        if province_id:
            query = query.filter(TouristAttraction.province_id == province_id)
        
        categories = query.all()
        return [category[0] for category in categories if category[0]]
    
    def update_rating(self, attraction_id: int, new_rating: float, new_review_count: int) -> Optional[AttractionSchema]:
        attraction = self.db.query(TouristAttraction).filter(
            TouristAttraction.id == attraction_id
        ).first()
        
        if not attraction:
            return None
        
        attraction.rating = new_rating
        attraction.total_reviews = new_review_count
        
        self.db.commit()
        self.db.refresh(attraction)
        return AttractionSchema.from_orm(attraction)