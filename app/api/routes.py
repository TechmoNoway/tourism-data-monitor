from fastapi import APIRouter
from app.api.endpoints import provinces, attractions, collection

router = APIRouter()

# Include các endpoint routes
router.include_router(provinces.router, prefix="/provinces", tags=["Provinces"])
router.include_router(attractions.router, prefix="/attractions", tags=["Tourist Attractions"])
router.include_router(collection.router, tags=["Data Collection"])

# Health check endpoint
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Tourism Data Monitor API is running",
        "endpoints": {
            "provinces": "/api/v1/provinces",
            "attractions": "/api/v1/attractions"
        }
    }