from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.services.tourist_attraction_service import TouristAttractionService
from app.schemas.attraction import (
    TouristAttraction,
    TouristAttractionCreate,
    TouristAttractionUpdate,
    TouristAttractionWithProvince,
)
from app.schemas.common import ApiResponse

router = APIRouter()


@router.get("/", response_model=List[TouristAttraction])
async def get_attractions(
    province_id: Optional[int] = Query(None, description="Filter by province"),
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search by name/description"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Minimum rating"),
    active_only: bool = Query(True, description="Only active attractions"),
    limit: int = Query(100, le=1000, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    service = TouristAttractionService(db)
    
    attractions = service.search_attractions(
        search_term=search,
        province_id=province_id,
        category=category,
        min_rating=min_rating,
        active_only=active_only
    )
    
    return attractions[:limit]


@router.get("/popular", response_model=List[TouristAttraction])
async def get_popular_attractions(
    province_id: Optional[int] = Query(None, description="Filter by province"),
    limit: int = Query(10, le=50, description="Number of results"),
    db: Session = Depends(get_db)
):
    service = TouristAttractionService(db)
    return service.get_popular_attractions(province_id=province_id, limit=limit)


@router.get("/categories", response_model=List[str])
async def get_attraction_categories(
    province_id: Optional[int] = Query(None, description="Filter by province"),
    db: Session = Depends(get_db)
):
    service = TouristAttractionService(db)
    return service.get_categories(province_id=province_id)


@router.get("/{attraction_id}", response_model=TouristAttractionWithProvince)
async def get_attraction(attraction_id: int, db: Session = Depends(get_db)):
    service = TouristAttractionService(db)
    attraction = service.get_attraction_with_province(attraction_id)
    
    if not attraction:
        raise HTTPException(status_code=404, detail="Tourist attraction not found")
    
    return attraction


@router.post("/", response_model=TouristAttraction)
async def create_attraction(
    attraction: TouristAttractionCreate, 
    db: Session = Depends(get_db)
):
    service = TouristAttractionService(db)
    
    try:
        new_attraction = service.create_attraction(attraction)
        return new_attraction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{attraction_id}", response_model=TouristAttraction)
async def update_attraction(
    attraction_id: int,
    update_data: TouristAttractionUpdate,
    db: Session = Depends(get_db)
):
    service = TouristAttractionService(db)
    
    updated_attraction = service.update_attraction(attraction_id, update_data)
    if not updated_attraction:
        raise HTTPException(status_code=404, detail="Tourist attraction not found")
    
    return updated_attraction


@router.delete("/{attraction_id}", response_model=ApiResponse)
async def deactivate_attraction(attraction_id: int, db: Session = Depends(get_db)):
    service = TouristAttractionService(db)
    
    success = service.deactivate_attraction(attraction_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tourist attraction not found")
    
    return ApiResponse(
        success=True,
        message=f"Tourist attraction {attraction_id} has been deactivated"
    )


@router.patch("/{attraction_id}/rating", response_model=TouristAttraction)
async def update_attraction_rating(
    attraction_id: int,
    rating: float = Query(..., ge=0, le=5, description="New rating (0-5)"),
    review_count: int = Query(..., ge=0, description="Total review count"),
    db: Session = Depends(get_db)
):
    service = TouristAttractionService(db)
    
    updated_attraction = service.update_rating(attraction_id, rating, review_count)
    if not updated_attraction:
        raise HTTPException(status_code=404, detail="Tourist attraction not found")
    
    return updated_attraction


@router.get("/category/{category}", response_model=List[TouristAttraction])
async def get_attractions_by_category(
    category: str,
    province_id: Optional[int] = Query(None, description="Filter by province"),
    db: Session = Depends(get_db)
):
    service = TouristAttractionService(db)
    return service.get_attractions_by_category(category, province_id)