import platform
import logging
import asyncio
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from app.core.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchedulerService:
    """
    Background scheduler service for automated data collection
    """
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.is_running = False
        self.platform = platform.system()
        
    def _job_listener(self, event):
        """Log job execution results"""
        if event.exception:
            logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            logger.info(f"Job {event.job_id} completed successfully")
    
    async def _weekly_collection(self):
        """Hourly collection task - collects from all attractions"""
        logger.info("="*70)
        logger.info("HOURLY DATA COLLECTION STARTED (All attractions)")
        logger.info("="*70)
        
        try:
            # Import here to avoid circular imports
            import subprocess
            import sys
            from pathlib import Path
            
            script_path = Path(__file__).parent.parent.parent / "scripts" / "collect_data_comprehensive.py"
            
            # Run collection for ALL attractions (batch processing handled by script)
            result = subprocess.run(
                [sys.executable, str(script_path), 
                 "--all"],  # Process all attractions
                capture_output=True,
                text=True,
                encoding='utf-8',  # Force UTF-8 encoding for Vietnamese text
                errors='replace',  # Replace any problematic characters
                timeout=1800  # 30 minutes timeout for full collection
            )
            
            if result.returncode == 0:
                logger.info("[SUCCESS] Hourly collection completed")
                logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            else:
                logger.error(f"[FAILED] Hourly collection failed with code {result.returncode}")
                logger.error(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                
        except subprocess.TimeoutExpired:
            logger.error("[FAILED] Demo collection timeout (>30 min)")
        except Exception as e:
            logger.error(f"[FAILED] Hourly collection error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _monthly_discovery(self):
        """Monthly attraction discovery task"""
        logger.info("="*70)
        logger.info("MONTHLY AUTO-DISCOVERY STARTED")
        logger.info("="*70)
        
        try:
            import subprocess
            import sys
            from pathlib import Path
            
            script_path = Path(__file__).parent.parent.parent / "scripts" / "auto_discover_with_collector.py"
            
            result = subprocess.run(
                [sys.executable, str(script_path), "--limit", "3"],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("[SUCCESS] Monthly discovery completed")
                logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            else:
                logger.error(f"[FAILED] Monthly discovery failed with code {result.returncode}")
                logger.error(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                
        except subprocess.TimeoutExpired:
            logger.error("[FAILED] Monthly discovery timeout (>30 min)")
        except Exception as e:
            logger.error(f"[FAILED] Monthly discovery error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        if not settings.USE_SCHEDULER_SERVICE:
            logger.info("Scheduler disabled (USE_SCHEDULER_SERVICE=false)")
            return
        
        logger.info("="*70)
        logger.info("STARTING SCHEDULER SERVICE")
        logger.info("="*70)
        logger.info(f"Platform: {self.platform}")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.scheduler = AsyncIOScheduler()
        
        self.scheduler.add_listener(
            self._job_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
        
        # Schedule collection to run immediately on startup, then every 1 hour
        self.scheduler.add_job(
            self._weekly_collection,
            'interval',
            hours=1,
            id='hourly_collection',
            name='Hourly Data Collection (All attractions)',
            replace_existing=True,
            next_run_time=datetime.now()  # Run immediately on startup
        )
        logger.info("[OK] Scheduled: Hourly collection (Immediate first run, then every 1 hour)")
        
        # Monthly discovery disabled for demo mode
        # self.scheduler.add_job(
        #     self._monthly_discovery,
        #     CronTrigger(day_of_week='sun', hour=1, minute=0),
        #     id='monthly_discovery',
        #     name='Monthly Auto-Discovery',
        #     replace_existing=True
        # )
        # logger.info("[OK] Scheduled: Monthly discovery (First Sunday 1:00 AM)")
        
        self.scheduler.start()
        self.is_running = True
        
        logger.info("")
        logger.info("Next scheduled runs:")
        for job in self.scheduler.get_jobs():
            job_detail = self.scheduler.get_job(job.id)
            if job_detail:
                logger.info(f"  -> {job.name}: {job_detail.next_run_time}")
        
        logger.info("="*70)
        logger.info("[OK] Scheduler service started successfully")
        logger.info("="*70)
    
    async def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return
        
        logger.info("Stopping scheduler service...")
        
        try:
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown(wait=False)
        except Exception as e:
            logger.debug(f"Scheduler shutdown issue (ignored): {e}")
        
        self.is_running = False
        logger.info("[OK] Scheduler service stopped")
    
    def get_status(self) -> dict:
        """Get scheduler status"""
        if not self.is_running or not self.scheduler:
            return {
                "running": False,
                "platform": self.platform,
                "enabled": settings.USE_SCHEDULER_SERVICE
            }
        
        jobs = []
        for job in self.scheduler.get_jobs():
            job_detail = self.scheduler.get_job(job.id)
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job_detail.next_run_time) if job_detail else None
            })
        
        return {
            "running": True,
            "platform": self.platform,
            "enabled": settings.USE_SCHEDULER_SERVICE,
            "jobs": jobs
        }


scheduler_service = SchedulerService()
