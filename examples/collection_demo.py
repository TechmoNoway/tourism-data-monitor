"""
Data Collection Usage Examples
"""
import asyncio
from app.collectors.data_pipeline import create_data_pipeline
from app.collectors.scheduler import initialize_scheduler


async def demo_collection():
    """
    Demo script showing how to use the data collection system
    """
    
    # 1. API Credentials (replace with your actual keys)
    api_credentials = {
        'youtube_api_key': 'YOUR_YOUTUBE_API_KEY',
        'google_maps_api_key': 'YOUR_GOOGLE_MAPS_API_KEY',
        'facebook_access_token': 'YOUR_FACEBOOK_ACCESS_TOKEN',
        'facebook_app_id': 'YOUR_FACEBOOK_APP_ID',
        'facebook_app_secret': 'YOUR_FACEBOOK_APP_SECRET'
    }
    
    # 2. Create data pipeline
    print("Creating data collection pipeline...")
    pipeline = create_data_pipeline(**api_credentials)
    
    print(f"Available platforms: {pipeline.get_available_platforms()}")
    
    # 3. Collect data for a specific attraction
    try:
        print("\n--- Manual Collection for Attraction ---")
        attraction_id = 1  # Replace with actual attraction ID
        
        result = await pipeline.collect_for_attraction(
            attraction_id=attraction_id,
            platforms=['youtube'],  # Only YouTube for demo
            limit_per_platform=5
        )
        
        print(f"Collection completed for attraction {attraction_id}:")
        print(f"- Posts collected: {result['total_posts']}")
        print(f"- Comments collected: {result['total_comments']}")
        print(f"- Platforms: {list(result['platforms'].keys())}")
        
    except Exception as e:
        print(f"Error in manual collection: {str(e)}")
    
    # 4. Initialize scheduler for automated collection
    print("\n--- Setting up Automated Collection ---")
    try:
        scheduler = initialize_scheduler(api_credentials)
        
        # Schedule daily collection at 2:00 AM
        scheduler.schedule_daily_collection(
            hour=2, 
            minute=0,
            platforms=['youtube', 'google_reviews']
        )
        
        # Schedule hourly collection for popular attractions
        popular_attractions = [1, 2, 3]  # Replace with actual IDs
        scheduler.schedule_hourly_collection(
            attraction_ids=popular_attractions,
            platforms=['youtube']
        )
        
        # Schedule weekly full collection
        scheduler.schedule_weekly_full_collection(
            day_of_week=6,  # Sunday
            hour=1,
            minute=0
        )
        
        print("Scheduled jobs:")
        for job in scheduler.get_stats()['scheduled_jobs']:
            print(f"- {job['name']}: {job['next_run']}")
        
        # Start the scheduler
        scheduler.start()
        print("Scheduler started!")
        
        # Run for a short time (in production, this would run continuously)
        await asyncio.sleep(10)
        
        scheduler.stop()
        print("Scheduler stopped.")
        
    except Exception as e:
        print(f"Error in scheduler setup: {str(e)}")


def demo_api_usage():
    """
    Demo showing how to use the collection API endpoints
    """
    
    api_examples = """
    # Data Collection API Usage Examples
    
    ## 1. Initialize the collection system
    POST /api/v1/collection/initialize
    {
        "youtube_api_key": "YOUR_KEY",
        "google_maps_api_key": "YOUR_KEY",
        "facebook_access_token": "YOUR_TOKEN",
        "facebook_app_id": "YOUR_APP_ID",
        "facebook_app_secret": "YOUR_SECRET"
    }
    
    ## 2. Check system status
    GET /api/v1/collection/status
    
    ## 3. Run manual collection
    POST /api/v1/collection/manual
    {
        "collection_type": "attraction",
        "target_id": 1,
        "platforms": ["youtube", "google_reviews"]
    }
    
    ## 4. Schedule automated collection
    POST /api/v1/collection/schedule
    {
        "job_type": "daily",
        "hour": 2,
        "minute": 0,
        "platforms": ["youtube", "google_reviews"]
    }
    
    ## 5. Get collection statistics
    GET /api/v1/collection/stats
    
    ## 6. View scheduled jobs
    GET /api/v1/collection/jobs
    
    ## 7. Start/Stop scheduler
    POST /api/v1/collection/start
    POST /api/v1/collection/stop
    
    ## 8. Get available platforms
    GET /api/v1/collection/platforms
    """
    
    print(api_examples)


if __name__ == "__main__":
    print("=== Tourism Data Collection Demo ===\n")
    
    print("1. API Usage Examples:")
    demo_api_usage()
    
    print("\n2. Running Collection Demo:")
    print("Note: This requires valid API keys to work properly")
    
    # Uncomment the line below to run the actual demo
    # asyncio.run(demo_collection())
    
    print("\nDemo completed!")
    print("\nNext steps:")
    print("1. Get API keys for YouTube, Google Maps, Facebook")
    print("2. Update api_credentials in the demo")
    print("3. Populate database with tourist attractions")
    print("4. Run the collection system")
    print("5. Start building the NLP analysis pipeline")