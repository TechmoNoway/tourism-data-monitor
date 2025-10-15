from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.services.tourist_attraction_service import TouristAttractionService
from app.schemas.attraction import (
    TouristAttraction,
    TouristAttractionCreate,
    TouristAttractionUpdate,
    TouristAttractionWithProvince
)
from app.schemas.common import ApiResponse

router = APIRouter()


@router.get("/", response_model=List[TouristAttraction])
async def get_attractions(
    province_id: Optional[int] = Query(None, description="Lọc theo tỉnh"),
    category: Optional[str] = Query(None, description="Lọc theo danh mục"),
    search: Optional[str] = Query(None, description="Tìm kiếm theo tên/mô tả"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Rating tối thiểu"),
    active_only: bool = Query(True, description="Chỉ lấy điểm du lịch hoạt động"),
    limit: int = Query(100, le=1000, description="Số lượng kết quả tối đa"),
    db: Session = Depends(get_db)
):
    """Lấy danh sách khu du lịch với các bộ lọc"""
    service = TouristAttractionService(db)
    
    attractions = service.search_attractions(
        search_term=search,
        province_id=province_id,
        category=category,
        min_rating=min_rating,
        active_only=active_only
    )
    
    # Limit results
    return attractions[:limit]


@router.get("/popular", response_model=List[TouristAttraction])
async def get_popular_attractions(
    province_id: Optional[int] = Query(None, description="Lọc theo tỉnh"),
    limit: int = Query(10, le=50, description="Số lượng kết quả"),
    db: Session = Depends(get_db)
):
    """Lấy các khu du lịch phổ biến (rating cao, nhiều review)"""
    service = TouristAttractionService(db)
    return service.get_popular_attractions(province_id=province_id, limit=limit)


@router.get("/categories", response_model=List[str])
async def get_attraction_categories(
    province_id: Optional[int] = Query(None, description="Lọc theo tỉnh"),
    db: Session = Depends(get_db)
):
    """Lấy danh sách các danh mục khu du lịch có sẵn"""
    service = TouristAttractionService(db)
    return service.get_categories(province_id=province_id)


@router.get("/{attraction_id}", response_model=TouristAttractionWithProvince)
async def get_attraction(attraction_id: int, db: Session = Depends(get_db)):
    """Lấy thông tin chi tiết một khu du lịch"""
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
    """Tạo khu du lịch mới"""
    # TODO: Add authentication/authorization
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
    """Cập nhật thông tin khu du lịch"""
    # TODO: Add authentication/authorization
    service = TouristAttractionService(db)
    
    updated_attraction = service.update_attraction(attraction_id, update_data)
    if not updated_attraction:
        raise HTTPException(status_code=404, detail="Tourist attraction not found")
    
    return updated_attraction


@router.delete("/{attraction_id}", response_model=ApiResponse)
async def deactivate_attraction(attraction_id: int, db: Session = Depends(get_db)):
    """Vô hiệu hóa khu du lịch (soft delete)"""
    # TODO: Add authentication/authorization
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
    rating: float = Query(..., ge=0, le=5, description="Rating mới (0-5)"),
    review_count: int = Query(..., ge=0, description="Tổng số review"),
    db: Session = Depends(get_db)
):
    """Cập nhật rating và số lượng review của khu du lịch"""
    service = TouristAttractionService(db)
    
    updated_attraction = service.update_rating(attraction_id, rating, review_count)
    if not updated_attraction:
        raise HTTPException(status_code=404, detail="Tourist attraction not found")
    
    return updated_attraction


@router.get("/category/{category}", response_model=List[TouristAttraction])
async def get_attractions_by_category(
    category: str,
    province_id: Optional[int] = Query(None, description="Lọc theo tỉnh"),
    db: Session = Depends(get_db)
):
    """Lấy khu du lịch theo danh mục cụ thể"""
    service = TouristAttractionService(db)
    return service.get_attractions_by_category(category, province_id)