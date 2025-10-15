from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.services.province_service import ProvinceService
from app.schemas.province import Province, ProvinceCreate, ProvinceWithStats
from app.schemas.attraction import TouristAttraction
from app.schemas.common import ApiResponse

router = APIRouter()


@router.get("/", response_model=List[Province])
async def get_provinces(db: Session = Depends(get_db)):
    """Lấy danh sách tất cả các tỉnh thành"""
    service = ProvinceService(db)
    provinces = service.get_all_provinces()
    return provinces


@router.get("/{province_id}", response_model=Province)
async def get_province(province_id: int, db: Session = Depends(get_db)):
    """Lấy thông tin chi tiết một tỉnh"""
    service = ProvinceService(db)
    province = service.get_province_by_id(province_id)
    
    if not province:
        raise HTTPException(status_code=404, detail="Province not found")
    
    return province


@router.get("/code/{province_code}", response_model=Province)
async def get_province_by_code(province_code: str, db: Session = Depends(get_db)):
    """Lấy thông tin tỉnh theo mã tỉnh (LD, DN, BT)"""
    service = ProvinceService(db)
    province = service.get_province_by_code(province_code)
    
    if not province:
        raise HTTPException(status_code=404, detail="Province not found")
    
    return province


@router.get("/{province_id}/stats", response_model=ProvinceWithStats)
async def get_province_stats(province_id: int, db: Session = Depends(get_db)):
    """Lấy thống kê chi tiết của một tỉnh"""
    service = ProvinceService(db)
    province_stats = service.get_province_with_stats(province_id)
    
    if not province_stats:
        raise HTTPException(status_code=404, detail="Province not found")
    
    return province_stats


@router.get("/{province_id}/attractions", response_model=List[TouristAttraction])
async def get_province_attractions(
    province_id: int,
    active_only: bool = Query(True, description="Chỉ lấy các điểm du lịch đang hoạt động"),
    category: Optional[str] = Query(None, description="Lọc theo danh mục"),
    search: Optional[str] = Query(None, description="Tìm kiếm theo tên"),
    db: Session = Depends(get_db)
):
    """Lấy tất cả khu du lịch của một tỉnh"""
    service = ProvinceService(db)
    
    # Kiểm tra tỉnh có tồn tại không
    province = service.get_province_by_id(province_id)
    if not province:
        raise HTTPException(status_code=404, detail="Province not found")
    
    # Lấy danh sách attractions
    attractions = service.get_attractions_by_province_id(province_id, active_only)
    
    # Lọc theo category nếu có
    if category:
        attractions = [a for a in attractions if a.category == category]
    
    # Tìm kiếm theo tên nếu có
    if search:
        search_lower = search.lower()
        attractions = [
            a for a in attractions 
            if search_lower in a.name.lower() or (a.description and search_lower in a.description.lower())
        ]
    
    return attractions


@router.get("/code/{province_code}/attractions", response_model=List[TouristAttraction])
async def get_province_attractions_by_code(
    province_code: str,
    active_only: bool = Query(True, description="Chỉ lấy các điểm du lịch đang hoạt động"),
    category: Optional[str] = Query(None, description="Lọc theo danh mục"),
    search: Optional[str] = Query(None, description="Tìm kiếm theo tên"),
    db: Session = Depends(get_db)
):
    """Lấy tất cả khu du lịch của một tỉnh theo mã tỉnh"""
    service = ProvinceService(db)
    
    # Kiểm tra tỉnh có tồn tại không
    province = service.get_province_by_code(province_code)
    if not province:
        raise HTTPException(status_code=404, detail="Province not found")
    
    # Lấy attractions thông qua province_id
    return await get_province_attractions(province.id, active_only, category, search, db)


@router.post("/", response_model=Province)
async def create_province(province: ProvinceCreate, db: Session = Depends(get_db)):
    """Tạo tỉnh thành mới (Admin only)"""
    # TODO: Add authentication/authorization
    service = ProvinceService(db)
    
    # Kiểm tra mã tỉnh đã tồn tại chưa
    existing = service.get_province_by_code(province.code)
    if existing:
        raise HTTPException(
            status_code=400, 
            detail=f"Province with code '{province.code}' already exists"
        )
    
    # Tạo tỉnh mới (cần implement trong service)
    # new_province = service.create_province(province)
    # return new_province
    
    raise HTTPException(status_code=501, detail="Create province not implemented yet")


@router.post("/init-sample-data", response_model=ApiResponse)
async def init_sample_provinces(db: Session = Depends(get_db)):
    """Khởi tạo dữ liệu mẫu cho 3 tỉnh thành"""
    # TODO: Implement sample data creation
    sample_provinces = [
        {"name": "Lâm Đồng", "code": "LD", "main_city": "Đà Lạt"},
        {"name": "Đà Nẵng", "code": "DN", "main_city": "Đà Nẵng"},
        {"name": "Bình Thuận", "code": "BT", "main_city": "Phan Thiết"}
    ]
    
    return ApiResponse(
        success=True,
        message="Sample data initialization endpoint ready",
        data={"provinces": sample_provinces}
    )