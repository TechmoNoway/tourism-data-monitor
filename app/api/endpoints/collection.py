from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.collectors.scheduler import get_scheduler, initialize_scheduler


router = APIRouter(prefix="/api/v1/collection", tags=["Data Collection"])


# Request/Response Models
class CollectionRequest(BaseModel):
    attraction_ids: Optional[List[int]] = None
    province_ids: Optional[List[int]] = None
    platforms: Optional[List[str]] = None
    limit_per_platform: int = 20

class ManualCollectionRequest(BaseModel):
    collection_type: str  # "province", "attraction", "all_provinces"
    target_id: Optional[int] = None
    platforms: Optional[List[str]] = None

class ScheduleRequest(BaseModel):
    job_type: str  # "daily", "hourly", "weekly"
    hour: Optional[int] = 2
    minute: Optional[int] = 0
    day_of_week: Optional[int] = 6  # For weekly jobs
    attraction_ids: Optional[List[int]] = None
    province_ids: Optional[List[int]] = None
    platforms: Optional[List[str]] = None

class APICredentials(BaseModel):
    youtube_api_key: Optional[str] = None
    google_maps_api_key: Optional[str] = None
    facebook_access_token: Optional[str] = None
    facebook_app_id: Optional[str] = None
    facebook_app_secret: Optional[str] = None


@router.post("/initialize")
async def initialize_collection_system(credentials: APICredentials):
    """
    Initialize the data collection system with API credentials
    """
    try:
        scheduler = initialize_scheduler(credentials.dict())
        return {
            "message": "Collection system initialized successfully",
            "available_platforms": scheduler.get_available_platforms(),
            "scheduler_stats": scheduler.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize collection system: {str(e)}")


@router.get("/status")
async def get_collection_status():
    """
    Get current status of the collection system
    """
    try:
        scheduler = get_scheduler()
        return {
            "status": "active" if scheduler.is_running else "inactive",
            "stats": scheduler.get_stats(),
            "available_platforms": scheduler.get_available_platforms()
        }
    except RuntimeError:
        return {
            "status": "not_initialized",
            "message": "Collection system not initialized. Use /initialize endpoint first."
        }


@router.post("/manual")
async def run_manual_collection(request: ManualCollectionRequest, background_tasks: BackgroundTasks):
    """
    Run manual data collection immediately
    """
    try:
        scheduler = get_scheduler()
        
        def run_collection():
            try:
                result = scheduler.run_manual_collection(
                    request.collection_type,
                    request.target_id,
                    request.platforms
                )
                return result
            except Exception as e:
                raise e
        
        background_tasks.add_task(run_collection)
        
        return {
            "message": "Manual collection started",
            "collection_type": request.collection_type,
            "target_id": request.target_id,
            "platforms": request.platforms or "all available"
        }
        
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Collection system not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start manual collection: {str(e)}")


@router.post("/schedule")
async def schedule_collection_job(request: ScheduleRequest):
    """
    Schedule automated data collection jobs
    """
    try:
        scheduler = get_scheduler()
        
        if request.job_type == "daily":
            scheduler.schedule_daily_collection(
                hour=request.hour or 2,
                minute=request.minute or 0,
                provinces=request.province_ids,
                platforms=request.platforms
            )
        elif request.job_type == "hourly":
            if not request.attraction_ids:
                raise HTTPException(status_code=400, detail="attraction_ids required for hourly jobs")
            scheduler.schedule_hourly_collection(
                attraction_ids=request.attraction_ids,
                platforms=request.platforms
            )
        elif request.job_type == "weekly":
            scheduler.schedule_weekly_full_collection(
                day_of_week=request.day_of_week or 6,
                hour=request.hour or 1,
                minute=request.minute or 0
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid job_type. Use 'daily', 'hourly', or 'weekly'")
        
        if not scheduler.is_running:
            scheduler.start()
        
        return {
            "message": f"{request.job_type.title()} collection job scheduled successfully",
            "job_type": request.job_type,
            "scheduler_stats": scheduler.get_stats()
        }
        
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Collection system not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.get("/jobs")
async def get_scheduled_jobs():
    """
    Get list of all scheduled collection jobs
    """
    try:
        scheduler = get_scheduler()
        return {
            "jobs": scheduler.get_stats()["scheduled_jobs"],
            "scheduler_running": scheduler.is_running
        }
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Collection system not initialized")


@router.get("/jobs/{job_id}")
async def get_job_details(job_id: str):
    """
    Get details of a specific scheduled job
    """
    try:
        scheduler = get_scheduler()
        job_info = scheduler.get_job_info(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return job_info
        
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Collection system not initialized")


@router.delete("/jobs/{job_id}")
async def remove_scheduled_job(job_id: str):
    """
    Remove a scheduled collection job
    """
    try:
        scheduler = get_scheduler()
        success = scheduler.remove_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or could not be removed")
        
        return {
            "message": f"Job {job_id} removed successfully"
        }
        
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Collection system not initialized")


@router.post("/start")
async def start_scheduler():
    """
    Start the collection scheduler
    """
    try:
        scheduler = get_scheduler()
        scheduler.start()
        
        return {
            "message": "Collection scheduler started",
            "status": "active"
        }
        
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Collection system not initialized")


@router.post("/stop")
async def stop_scheduler():
    """
    Stop the collection scheduler
    """
    try:
        scheduler = get_scheduler()
        scheduler.stop()
        
        return {
            "message": "Collection scheduler stopped",
            "status": "inactive"
        }
        
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Collection system not initialized")


@router.get("/platforms")
async def get_available_platforms():
    """
    Get list of available data collection platforms
    """
    try:
        scheduler = get_scheduler()
        platforms = scheduler.get_available_platforms()
        
        return {
            "available_platforms": platforms,
            "platform_info": {
                "youtube": {
                    "name": "YouTube",
                    "description": "Collect videos and comments about tourist attractions",
                    "data_types": ["videos", "comments"]
                },
                "google_reviews": {
                    "name": "Google Reviews",
                    "description": "Collect place information and reviews from Google Places",
                    "data_types": ["places", "reviews"]
                },
                "facebook": {
                    "name": "Facebook",
                    "description": "Collect posts and comments from tourism pages",
                    "data_types": ["posts", "comments"]
                }
            }
        }
        
    except RuntimeError:
        return {
            "available_platforms": [],
            "message": "Collection system not initialized",
            "platform_info": {
                "youtube": {
                    "name": "YouTube",
                    "description": "Collect videos and comments about tourist attractions",
                    "data_types": ["videos", "comments"],
                    "requires": "YouTube Data API key"
                },
                "google_reviews": {
                    "name": "Google Reviews", 
                    "description": "Collect place information and reviews from Google Places",
                    "data_types": ["places", "reviews"],
                    "requires": "Google Maps API key"
                },
                "facebook": {
                    "name": "Facebook",
                    "description": "Collect posts and comments from tourism pages", 
                    "data_types": ["posts", "comments"],
                    "requires": "Facebook App credentials"
                }
            }
        }


@router.get("/stats")
async def get_collection_statistics():
    """
    Get detailed collection statistics
    """
    try:
        scheduler = get_scheduler()
        stats = scheduler.get_stats()
        
        return {
            "collection_stats": stats,
            "performance_metrics": {
                "success_rate": stats['successful_runs'] / max(stats['total_runs'], 1) * 100,
                "total_data_collected": stats['total_posts_collected'] + stats['total_comments_collected'],
                "average_posts_per_run": stats['total_posts_collected'] / max(stats['successful_runs'], 1),
                "average_comments_per_run": stats['total_comments_collected'] / max(stats['successful_runs'], 1)
            }
        }
        
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Collection system not initialized")