import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import logging

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError:
    logging.warning("APScheduler not installed. Run: pip install apscheduler")

from app.collectors.data_pipeline import create_data_pipeline


class CollectionScheduler:
   
    def __init__(self, api_credentials: Dict[str, Any]):
        self.logger = logging.getLogger("collection_scheduler")
        self.scheduler = AsyncIOScheduler()
        self.pipeline = create_data_pipeline(**api_credentials)
        self.is_running = False
        
        # Collection statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run': None,
            'last_success': None,
            'total_posts_collected': 0,
            'total_comments_collected': 0
        }
    
    def start(self):
        """Start the scheduler"""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            self.logger.info("Collection scheduler started")
        
    def stop(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            self.logger.info("Collection scheduler stopped")
    
    def schedule_daily_collection(
        self, 
        hour: int = 2, 
        minute: int = 0,
        provinces: Optional[List[int]] = None,
        platforms: Optional[List[str]] = None
    ):
        """
        Schedule daily data collection
        
        Args:
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            provinces: List of province IDs to collect (None = all)
            platforms: List of platforms to use (None = all available)
        """
        trigger = CronTrigger(hour=hour, minute=minute)
        
        self.scheduler.add_job(
            self._daily_collection_job,
            trigger=trigger,
            args=[provinces, platforms],
            id='daily_collection',
            name='Daily Tourism Data Collection',
            replace_existing=True
        )
        
        self.logger.info(f"Scheduled daily collection at {hour:02d}:{minute:02d}")
    
    def schedule_hourly_collection(
        self, 
        attraction_ids: List[int],
        platforms: Optional[List[str]] = None,
        limit_per_platform: int = 10
    ):
        trigger = IntervalTrigger(hours=1)
        
        self.scheduler.add_job(
            self._hourly_collection_job,
            trigger=trigger,
            args=[attraction_ids, platforms, limit_per_platform],
            id='hourly_collection',
            name='Hourly Hot Attractions Monitoring',
            replace_existing=True
        )
        
        self.logger.info(f"Scheduled hourly collection for {len(attraction_ids)} attractions")
    
    def schedule_weekly_full_collection(
        self, 
        day_of_week: int = 6,  # Sunday
        hour: int = 1,
        minute: int = 0
    ):
        trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
        
        self.scheduler.add_job(
            self._weekly_full_collection_job,
            trigger=trigger,
            id='weekly_full_collection',
            name='Weekly Full Data Collection',
            replace_existing=True
        )
        
        self.logger.info(f"Scheduled weekly full collection on day {day_of_week} at {hour:02d}:{minute:02d}")
    
    async def _daily_collection_job(
        self, 
        provinces: Optional[List[int]] = None,
        platforms: Optional[List[str]] = None
    ):
        self.logger.info("Starting daily collection job")
        self.stats['total_runs'] += 1
        self.stats['last_run'] = datetime.utcnow().isoformat()
        
        try:
            if provinces:
                results = {
                    'total_posts': 0,
                    'total_comments': 0,
                    'provinces_processed': 0
                }
                
                for province_id in provinces:
                    try:
                        province_result = await self.pipeline.collect_for_province(
                            province_id, 
                            platforms, 
                            limit_per_attraction=20
                        )
                        
                        results['total_posts'] += province_result['total_posts']
                        results['total_comments'] += province_result['total_comments']
                        results['provinces_processed'] += 1
                        
                        await asyncio.sleep(5)
                        
                    except Exception as e:
                        self.logger.error(f"Error in daily collection for province {province_id}: {str(e)}")
            else:
                results = await self.pipeline.collect_all_provinces(
                    platforms, 
                    limit_per_attraction=15
                )
            
            self.stats['successful_runs'] += 1
            self.stats['last_success'] = datetime.now(timezone.utc).isoformat()
            self.stats['total_posts_collected'] += results['total_posts']
            self.stats['total_comments_collected'] += results['total_comments']
            
            self.logger.info(f"Daily collection completed: {results['total_posts']} posts, {results['total_comments']} comments")
            
        except Exception as e:
            self.stats['failed_runs'] += 1
            self.logger.error(f"Daily collection job failed: {str(e)}")
    
    async def _hourly_collection_job(
        self, 
        attraction_ids: List[int],
        platforms: Optional[List[str]] = None,
        limit_per_platform: int = 10
    ):
        self.logger.info(f"Starting hourly collection for {len(attraction_ids)} attractions")
        
        try:
            total_posts = 0
            total_comments = 0
            
            for attraction_id in attraction_ids:
                try:
                    result = await self.pipeline.collect_for_attraction(
                        attraction_id, 
                        platforms, 
                        limit_per_platform
                    )
                    
                    total_posts += result['total_posts']
                    total_comments += result['total_comments']
                    
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Error in hourly collection for attraction {attraction_id}: {str(e)}")
            
            self.stats['total_posts_collected'] += total_posts
            self.stats['total_comments_collected'] += total_comments
            
            self.logger.info(f"Hourly collection completed: {total_posts} posts, {total_comments} comments")
            
        except Exception as e:
            self.logger.error(f"Hourly collection job failed: {str(e)}")
    
    async def _weekly_full_collection_job(self):
        self.logger.info("Starting weekly full collection job")
        
        try:
            results = await self.pipeline.collect_all_provinces(
                platforms=None,  
                limit_per_attraction=50 
            )
            
            # Update statistics
            self.stats['total_posts_collected'] += results['total_posts']
            self.stats['total_comments_collected'] += results['total_comments']
            
            self.logger.info(f"Weekly full collection completed: {results['total_posts']} posts, {results['total_comments']} comments")
            
        except Exception as e:
            self.logger.error(f"Weekly full collection job failed: {str(e)}")
    
    def run_manual_collection(
        self, 
        collection_type: str = "all_provinces",
        target_id: Optional[int] = None,
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        async def _run_collection():
            if collection_type == "province" and target_id:
                return await self.pipeline.collect_for_province(target_id, platforms, 30)
            elif collection_type == "attraction" and target_id:
                return await self.pipeline.collect_for_attraction(target_id, platforms, 50)
            elif collection_type == "all_provinces":
                return await self.pipeline.collect_all_provinces(platforms, 20)
            else:
                raise ValueError(f"Invalid collection type: {collection_type}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_run_collection())
            self.stats['total_posts_collected'] += result.get('total_posts', 0)
            self.stats['total_comments_collected'] += result.get('total_comments', 0)
            return result
        except Exception as e:
            self.logger.error(f"Manual collection failed: {str(e)}")
            raise
        finally:
            loop.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            **self.stats,
            'is_running': self.is_running,
            'available_platforms': self.pipeline.get_available_platforms(),
            'scheduled_jobs': [
                {
                    'id': job.id,
                    'name': job.name,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in self.scheduler.get_jobs()
            ]
        }
    
    def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific job"""
        job = self.scheduler.get_job(job_id)
        if job:
            return {
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            }
        return None
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job"""
        try:
            self.scheduler.remove_job(job_id)
            self.logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove job {job_id}: {str(e)}")
            return False


# Global scheduler instance
_scheduler = None

def get_scheduler() -> CollectionScheduler:
    """Get the global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized. Call initialize_scheduler() first.")
    return _scheduler

def initialize_scheduler(api_credentials: Dict[str, Any]) -> CollectionScheduler:
    """Initialize the global scheduler"""
    global _scheduler
    _scheduler = CollectionScheduler(api_credentials)
    return _scheduler